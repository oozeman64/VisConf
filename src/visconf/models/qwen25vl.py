"""Single-device Qwen2.5-VL facade for predictor-aligned decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers.cache_utils import Cache, DynamicCache

from visconf.config import ModelSettings
from visconf.models.instrumentation import (
    QwenInstrumentation,
    StepObservations,
    discover_decoder_layers,
)
from visconf.models.token_positions import build_prompt_token_groups
from visconf.types import PromptRecord, TokenGroups


class QwenFacadeError(RuntimeError):
    """Raised when Qwen model state violates the facade contract."""


@dataclass(slots=True)
class PreparedQwenPrompt:
    model_inputs: dict[str, torch.Tensor]
    prompt_record: PromptRecord
    token_groups: TokenGroups


@dataclass(frozen=True, slots=True)
class PrefillResult:
    raw_logits: torch.Tensor
    cache: Cache
    observations: StepObservations
    attention_mask: torch.Tensor
    rope_deltas: torch.Tensor


@dataclass(frozen=True, slots=True)
class DecodeResult:
    raw_logits: torch.Tensor
    cache: Cache
    observations: StepObservations
    attention_mask: torch.Tensor


class QwenModelFacade:
    _VISION_KEYS = (
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "second_per_grid_ts",
    )

    def __init__(
        self,
        model: Qwen2_5_VLForConditionalGeneration,
        processor: Any,
        device: torch.device,
    ) -> None:
        self.model = model
        self.processor = processor
        self.device = device
        self.tokenizer = processor.tokenizer

    @classmethod
    def load(cls, config: ModelSettings) -> "QwenModelFacade":
        device = torch.device(config.device)
        if device.type != "cuda" or device.index is None:
            raise QwenFacadeError("Qwen requires one explicit CUDA device")
        processor = AutoProcessor.from_pretrained(
            config.tokenizer_path or config.model_path,
            revision=config.revision,
            trust_remote_code=config.trust_remote_code,
            local_files_only=True,
            use_fast=config.processor_use_fast,
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.model_path,
            revision=config.revision,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            trust_remote_code=config.trust_remote_code,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        model.eval()

        text_config = model.config.text_config
        if text_config.num_hidden_layers != 36:
            raise QwenFacadeError("checkpoint must contain exactly 36 layers")
        if text_config._attn_implementation != "eager":
            raise QwenFacadeError("checkpoint attention must be eager")
        parameter_devices = {parameter.device for parameter in model.parameters()}
        if parameter_devices != {device}:
            raise QwenFacadeError("model must reside on one CUDA device")
        discover_decoder_layers(model)
        return cls(model=model, processor=processor, device=device)

    def prepare_example(
        self,
        messages: list[dict[str, object]],
    ) -> PreparedQwenPrompt:
        rendered_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        batch = self.processor(
            text=[rendered_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt",
        ).to(self.device)
        if "attention_mask" not in batch:
            batch["attention_mask"] = torch.ones_like(batch["input_ids"])

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        valid_ids = input_ids[0][attention_mask[0] == 1]
        prompt_record = PromptRecord(
            rendered_prompt=rendered_prompt,
            prompt_token_ids=tuple(int(value) for value in valid_ids.tolist()),
            prompt_token_count=int(valid_ids.numel()),
        )
        token_groups = build_prompt_token_groups(
            input_ids[0],
            attention_mask[0],
            self.tokenizer,
        )
        return PreparedQwenPrompt(
            model_inputs={
                key: value
                for key, value in batch.items()
                if isinstance(value, torch.Tensor)
            },
            prompt_record=prompt_record,
            token_groups=token_groups,
        )

    @torch.inference_mode()
    def prefill(
        self,
        prepared: PreparedQwenPrompt,
        instrumentation: QwenInstrumentation,
    ) -> PrefillResult:
        inputs = dict(prepared.model_inputs)
        sequence_length = inputs["input_ids"].shape[1]
        cache_position = torch.arange(
            sequence_length,
            device=self.device,
            dtype=torch.long,
        )
        model_inputs = self.model.prepare_inputs_for_generation(
            **inputs,
            cache_position=cache_position,
            use_cache=True,
        )
        instrumentation.begin_prefill(
            prepared.token_groups,
            inputs["attention_mask"],
        )
        try:
            output = self.model(
                **model_inputs,
                output_attentions=False,
                return_dict=True,
                logits_to_keep=1,
            )
            observations = instrumentation.finish_step()
        except BaseException:
            instrumentation.cancel_step()
            raise
        finally:
            for key in self._VISION_KEYS:
                prepared.model_inputs.pop(key, None)

        if output.past_key_values is None or output.rope_deltas is None:
            raise QwenFacadeError("prefill did not return cache and rope deltas")
        rope_deltas = output.rope_deltas.detach().clone()
        self.model.model.rope_deltas = rope_deltas
        return PrefillResult(
            raw_logits=output.logits[0, -1, :].detach(),
            cache=output.past_key_values,
            observations=observations,
            attention_mask=inputs["attention_mask"].detach().clone(),
            rope_deltas=rope_deltas,
        )

    def repeat_cache(self, base_cache: Cache, batch_size: int) -> Cache:
        if batch_size < 1:
            raise QwenFacadeError("cache batch size must be positive")
        if not hasattr(base_cache, "to_legacy_cache"):
            raise QwenFacadeError("unsupported Transformers cache type")
        legacy = base_cache.to_legacy_cache()
        cloned = tuple(
            (key.detach().clone(), value.detach().clone())
            for key, value in legacy
        )
        cache = DynamicCache.from_legacy_cache(cloned)
        cache.batch_repeat_interleave(batch_size)
        return cache

    def select_cache_rows(
        self,
        cache: Cache,
        indices: torch.Tensor,
    ) -> Cache:
        cache.batch_select_indices(
            indices.to(device=self.device, dtype=torch.long)
        )
        return cache

    @torch.inference_mode()
    def decode_step(
        self,
        selected_token_ids: torch.Tensor,
        cache: Cache,
        attention_mask: torch.Tensor,
        rope_deltas: torch.Tensor,
        prompt_groups: TokenGroups,
        instrumentation: QwenInstrumentation,
    ) -> DecodeResult:
        token_ids = selected_token_ids.to(
            device=self.device,
            dtype=torch.long,
        ).reshape(-1, 1)
        batch_size = token_ids.shape[0]
        if attention_mask.shape[0] == 1 and batch_size > 1:
            attention_mask = attention_mask.repeat(batch_size, 1)
        if attention_mask.shape[0] != batch_size:
            raise QwenFacadeError("attention mask batch differs from tokens")
        attention_mask = torch.cat(
            (
                attention_mask.to(self.device),
                torch.ones(
                    (batch_size, 1),
                    dtype=attention_mask.dtype,
                    device=self.device,
                ),
            ),
            dim=1,
        )
        cache_position = torch.tensor(
            [cache.get_seq_length()],
            dtype=torch.long,
            device=self.device,
        )
        self.model.model.rope_deltas = rope_deltas
        model_inputs = self.model.prepare_inputs_for_generation(
            token_ids,
            past_key_values=cache,
            attention_mask=attention_mask,
            cache_position=cache_position,
            use_cache=True,
        )
        instrumentation.begin_decode(
            prompt_groups,
            batch_size,
            attention_mask,
        )
        try:
            output = self.model(
                **model_inputs,
                output_attentions=False,
                return_dict=True,
                logits_to_keep=1,
            )
            observations = instrumentation.finish_step()
        except BaseException:
            instrumentation.cancel_step()
            raise
        if output.past_key_values is None:
            raise QwenFacadeError("decode did not return a cache")
        return DecodeResult(
            raw_logits=output.logits[:, -1, :].detach(),
            cache=output.past_key_values,
            observations=observations,
            attention_mask=attention_mask,
        )

    def decode_retained(self, token_ids: tuple[int, ...]) -> str:
        return self.tokenizer.decode(
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def token_piece(self, token_id: int) -> str:
        return str(self.tokenizer.convert_ids_to_tokens(token_id))

    def token_text(self, token_id: int) -> str:
        return self.tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def stop_token_ids(self) -> frozenset[int]:
        return frozenset(int(value) for value in self.stop_token_id_names())

    def stop_token_id_names(self) -> dict[str, tuple[str, ...]]:
        eos_id = self.tokenizer.eos_token_id
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if eos_id is None or im_end_id is None or im_end_id < 0:
            raise QwenFacadeError("tokenizer stop tokens are unresolved")
        names: dict[str, list[str]] = {}
        for token_id, symbolic_name in (
            (int(eos_id), "eos_token"),
            (int(im_end_id), "<|im_end|>"),
        ):
            names.setdefault(str(token_id), []).append(symbolic_name)
        return {
            token_id: tuple(symbolic_names)
            for token_id, symbolic_names in names.items()
        }

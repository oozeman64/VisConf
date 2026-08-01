"""Importable answer extraction and comparison for math benchmarks."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any


_FRAC_RE = re.compile(
    r"\\[dt]?frac\s*\{\s*(-?\d*\.?\d+)\s*\}"
    r"\s*\{\s*(-?\d*\.?\d+)\s*\}"
)
_RATIO_RE = re.compile(r"(-?\d*\.?\d+)\s*/\s*(-?\d*\.?\d+)")
_BOXED_LETTER_RE = re.compile(
    r"(?:<boxed>|(?<![A-Za-z])\\?boxed\s*\{)"
    r"\s*[\(\[]?\s*([A-Ha-h])\b"
)


def extract_boxed_answer(text: str) -> str | None:
    matches = list(re.finditer(r"\\boxed\{", text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    position = start
    while position < len(text) and depth:
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
        position += 1
    return text[start : position - 1].strip() if depth == 0 else None


def strip_latex_units(value: Any) -> str:
    text = str(value)
    text = re.sub(
        r"\\(?:text|mathrm|mbox|operatorname)\s*\{([^{}]*)\}",
        r"\1",
        text,
    )
    text = re.sub(r"\\left|\\right|\\,|\\;|\\!|\\ ", " ", text)
    return text.replace("$", "").strip()


def extract_number_str(value: Any) -> str | None:
    """Return the first numeric substring for diagnostics only."""
    cleaned = strip_latex_units(str(value).lower()).replace(",", "")
    match = re.search(r"[-+]?\d*\.?\d+", cleaned)
    return match.group() if match else None


def _clean_numeric(value: Any) -> str:
    text = strip_latex_units(value).lower().replace(",", "")
    text = text.replace("\\%", "").replace("%", "")
    text = re.sub(r"\^\{?\\circ\}?|\\circ|\\degree|°|~", "", text)
    return re.sub(r"^[a-z][a-z\s]*=", "", text).strip()


def _fraction_match(text: str):
    match = _FRAC_RE.fullmatch(text) or _RATIO_RE.fullmatch(text)
    if match:
        return match
    stripped = re.sub(r"[a-z]+$", "", text).strip()
    if stripped != text:
        return _FRAC_RE.fullmatch(stripped) or _RATIO_RE.fullmatch(stripped)
    return None


def to_quantity(value: Any) -> Decimal | None:
    text = _clean_numeric(value)
    fraction = _fraction_match(text)
    if fraction:
        try:
            numerator = Decimal(fraction.group(1))
            denominator = Decimal(fraction.group(2))
            return numerator / denominator if denominator else None
        except (InvalidOperation, ValueError, ZeroDivisionError):
            return None
    if re.search(r"\\[a-zA-Z]", text):
        return None
    text = re.sub(r"[a-z]+$", "", text).strip()
    if not re.fullmatch(r"[-+]?\d*\.?\d+", text):
        return None
    try:
        return Decimal(text)
    except (InvalidOperation, ValueError):
        return None


def numeric_equal(prediction: Any, gold: Any) -> bool | None:
    predicted = to_quantity(prediction)
    expected = to_quantity(gold)
    if predicted is None or expected is None:
        return None
    if looks_like_fraction(prediction) or looks_like_fraction(gold):
        return predicted == expected
    decimals = gold_decimals(gold)
    if decimals:
        quantum = Decimal(1).scaleb(-decimals)
        return predicted.quantize(
            quantum,
            rounding=ROUND_HALF_UP,
        ) == expected.quantize(quantum, rounding=ROUND_HALF_UP)
    return predicted == expected


def looks_like_fraction(value: Any) -> bool:
    return bool(_fraction_match(_clean_numeric(value)))


def gold_decimals(gold: Any) -> int:
    cleaned = strip_latex_units(str(gold)).replace(",", "")
    match = re.search(r"-?\d*\.(\d+)", cleaned)
    return len(match.group(1)) if match else 0


def normalize_answer(value: Any) -> str:
    if value is None:
        return ""
    text = strip_latex_units(str(value).strip().lower())
    text = re.sub(r"[$%,]", "", text)
    try:
        return f"{float(text):g}"
    except Exception:
        return text.strip()


def normalize_choice_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    for old, new in {
        "$": "",
        "\\(": "",
        "\\)": "",
        "\\[": "",
        "\\]": "",
        "\\left": "",
        "\\right": "",
        "\\mathrm": "",
        "\\text": "",
        "\\dfrac": "\\frac",
        "°": "^\\circ",
    }.items():
        text = text.replace(old, new)
    text = re.sub(r"\\boxed\{(.+?)\}", r"\1", text)
    text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", text)
    return re.sub(r"[{}\s,，。.;；:：]", "", text).strip("()[]")


def parse_mcq_choices(question: str, max_choices: int = 8) -> dict[str, str]:
    letters = "".join(chr(65 + index) for index in range(max_choices))
    matches = list(
        re.finditer(
            rf"^\s*[\(\[]?\s*([{letters}])\s*[\.\)\]\．、:]\s*",
            question,
            re.MULTILINE,
        )
    )
    choices: dict[str, str] = {}
    for index, match in enumerate(matches):
        letter = match.group(1).upper()
        start = match.end()
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(question)
        )
        choices[letter] = question[start:end].strip()
    return choices


def _last_sentence(text: str) -> str:
    lines = [line.strip() for line in text.rstrip().split("\n") if line.strip()]
    line = lines[-1] if lines else text
    parts = [
        part.strip()
        for part in re.split(
            r"(?<![0-9])\.\s+(?=[A-Z])|[!?。]\s*",
            line,
        )
        if part.strip()
    ]
    return parts[-1] if parts else line


def _last_line(text: str) -> str:
    lines = [
        line.strip()
        for line in str(text).rstrip().split("\n")
        if line.strip()
    ]
    return lines[-1] if lines else str(text).strip()


def _boxed_letter_lenient(text: str) -> str:
    matches = _BOXED_LETTER_RE.findall(text)
    return matches[-1].upper() if matches else ""


def _extract_choice(text: str, raw_boxed: str, question: str):
    boxed = strip_latex_units(raw_boxed).strip().upper()
    match = re.match(r"^[\(\[]?\s*([A-H])\b", boxed)
    if match:
        return match.group(1), "boxed_letter"
    if raw_boxed:
        normalized = normalize_choice_text(raw_boxed)
        for letter, choice in parse_mcq_choices(question).items():
            if normalized and normalized == normalize_choice_text(choice):
                return letter, "boxed_option_text"

    malformed = _boxed_letter_lenient(text)
    if malformed:
        return malformed, "boxed_letter_lenient"

    final = _last_sentence(text)
    for pattern in (
        r"(?:answer|choice)\s*(?:is\b\s*)?:?\s*[\(\[]?\s*([A-H])\b",
        r"\boption\s*(?:is\b\s*)?:?\s*[\(\[]?\s*([A-H])\b",
    ):
        match = re.search(pattern, final, re.IGNORECASE)
        if match:
            return match.group(1).upper(), "explicit_answer_letter"

    last_line = _last_line(text)
    match = re.match(r"^[\(\[]?\s*([A-H])\s*(?:[.\):\]]|$)", last_line)
    if match:
        letter = match.group(1).upper()
        choices = parse_mcq_choices(question)
        if not choices or letter in choices:
            return letter, "final_line_letter"
    return "", "boxed_unmapped" if raw_boxed else "none"


def score_generation(
    generated_text: str,
    question: str,
    ground_truth: str,
    answer_type: str = "multiple_choice",
) -> dict[str, Any]:
    raw_final = extract_boxed_answer(generated_text)
    gold = str(ground_truth or "").strip().upper()
    if answer_type == "multiple_choice":
        extracted, method = _extract_choice(
            generated_text,
            raw_final or "",
            question,
        )
        gold_match = re.match(
            r"^[\(\[]?\s*([A-H])\b",
            strip_latex_units(gold),
        )
        gold_letter = gold_match.group(1) if gold_match else gold
        correct = (
            extracted.strip().upper() == gold_letter
            if extracted
            else None
        )
    else:
        if raw_final:
            extracted, method = raw_final, "boxed_freeform"
        else:
            match = re.search(
                r"(?:answer|result|value\b[^.]{0,20}?)\s+"
                r"(?:is|=|:)\s*"
                r"([-+]?[\d,]+(?:\.\d+)?"
                r"(?:\s*/\s*[\d,]+(?:\.\d+)?)?)",
                _last_sentence(generated_text),
                re.IGNORECASE,
            )
            extracted = (
                re.sub(r"[,\s]", "", match.group(1)) if match else ""
            )
            method = "explicit_answer_freeform" if match else "none"
        numerical = numeric_equal(extracted, ground_truth) if extracted else None
        correct = None if not extracted else (
            numerical
            if numerical is not None
            else normalize_answer(extracted) == normalize_answer(ground_truth)
        )
    return {
        "raw_final_answer": raw_final,
        "extracted_answer": extracted or None,
        "is_correct": correct,
        "scorer_method": method,
    }

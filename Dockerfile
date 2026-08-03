# syntax=docker/dockerfile:1.7

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

LABEL org.opencontainers.image.title="VisConf Qwen2.5-VL metrics"
LABEL org.opencontainers.image.description="RunPod image for predictor-aligned visual-confidence experiments"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HF_HOME=/workspace/.cache/huggingface \
    HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers \
    PIP_CACHE_DIR=/workspace/.cache/pip \
    VISCONF_IMAGE=runpod-qwen3vl-metrics \
    VISCONF_MAX_THREADS=8

RUN python -c "import sys; assert sys.version_info[:2] == (3, 11), sys.version" \
    && python -c "import torch; assert torch.__version__ == '2.5.1+cu124', torch.__version__; assert torch.version.cuda == '12.4', torch.version.cuda"

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git openssh-server \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/run/sshd

WORKDIR /opt/visconf
COPY . /opt/visconf

RUN python -m pip install --no-cache-dir \
        "setuptools==82.0.1" \
        "wheel==0.47.0" \
    && python -m pip install --no-cache-dir \
        "Pillow==12.2.0" \
        "pyarrow==24.0.0" \
        "pydantic==2.13.4" \
        "PyYAML==6.0.3" \
        "qwen-vl-utils==0.0.8" \
        "transformers==4.57.0" \
        "pytest==8.4.2" \
    && python -m pip install --no-cache-dir --no-deps --no-build-isolation -e /opt/visconf \
    && python -m pip uninstall -y ninja \
    && python -m pip check

COPY docker/entrypoint.sh /usr/local/bin/visconf-entrypoint
RUN chmod 0755 /usr/local/bin/visconf-entrypoint

# Installed into site-packages so the thread cap applies to every interpreter,
# including RunPod SSH sessions, which bypass sshd/PAM and start a non-login
# shell -- no environment set by the entrypoint reaches them.
COPY docker/sitecustomize.py /tmp/sitecustomize.py
RUN python -c "import shutil, sysconfig; shutil.copy('/tmp/sitecustomize.py', sysconfig.get_paths()['purelib'] + '/sitecustomize.py')" \
    && rm /tmp/sitecustomize.py \
    && python -c "import os; value = os.environ.get('OMP_NUM_THREADS'); assert value and int(value) >= 1, value" \
    && OMP_NUM_THREADS=3 python -c "import os; assert os.environ['OMP_NUM_THREADS'] == '3', os.environ['OMP_NUM_THREADS']"

EXPOSE 22
VOLUME ["/workspace"]
WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/visconf-entrypoint"]

ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/venv/bin:${PATH}"
# Keep pip cache on the large workspace volume, never on the container overlay FS
ENV PIP_CACHE_DIR=/workspace/.pip-cache

# Install OS packages (including tesseract for OCR)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip \
    build-essential cmake git vim sudo \
    libgl1 libopencv-dev \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Create venv and upgrade pip
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --upgrade pip

# Install project dependencies at build time.
# PyTorch nightly (cu128) must come first via its own index URL so that
# the CUDA 12.8 wheels are picked (RTX 5090 / sm_120 requires nightly).
COPY requirements.txt /tmp/requirements.txt
RUN /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt

# Create render group for GPU access
RUN groupadd -f render

# Configure ubuntu user (already exists in base image)
RUN usermod -aG render ubuntu && \
    echo "ubuntu ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/ubuntu && \
    chmod 0440 /etc/sudoers.d/ubuntu

WORKDIR /workspace
RUN chown -R ubuntu:ubuntu /opt/venv

USER ubuntu
EXPOSE 8888 7860 7861
CMD ["/bin/bash"]

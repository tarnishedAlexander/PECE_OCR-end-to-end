FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/venv/bin:${PATH}"

# Install OS packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip \
    build-essential cmake git vim sudo \
    libgl1 libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Create venv and install dependencies
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --upgrade pip

# Install project dependencies at build time for plug-and-play usage.
COPY requirements.txt /tmp/requirements.txt
RUN /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt

# Create render group for GPU access
RUN groupadd -f render

# Configure ubuntu user (already exists in base image)
RUN usermod -aG render ubuntu && \
    echo "ubuntu ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/ubuntu && \
    chmod 0440 /etc/sudoers.d/ubuntu

WORKDIR /workspace
RUN chown -R ubuntu:ubuntu /workspace /opt/venv

USER ubuntu
EXPOSE 8888 7860 7861
CMD ["/bin/bash"]

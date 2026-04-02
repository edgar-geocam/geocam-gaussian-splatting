FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

RUN apt-get update && apt-get install -y \
    git build-essential cmake ninja-build \
    python3 python3-pip python3-dev \
    libgl1-mesa-glx libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install PyTorch
RUN pip3 install torch==2.4.0 torchvision --index-url https://download.pytorch.org/whl/cu121

# Install gsplat
RUN pip3 install gsplat==1.4.0

# Install nerfstudio
RUN pip3 install nerfstudio

CMD ["bash"]

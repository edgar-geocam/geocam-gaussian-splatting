FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX"

RUN apt-get update && apt-get install -y \
    git build-essential cmake ninja-build \
    libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev \
    libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev \
    libeigen3-dev libxxf86vm-dev libembree-dev \
    colmap imagemagick \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Clone 2DGS
RUN git clone https://github.com/hbb1/2d-gaussian-splatting.git --recursive

# Pin numpy <2.0 (PyTorch 2.0.1 is incompatible with numpy 2.x)
RUN pip install "numpy<2.0"

# Build custom CUDA modules
RUN pip install /workspace/2d-gaussian-splatting/submodules/diff-surfel-rasterization
RUN pip install /workspace/2d-gaussian-splatting/submodules/simple-knn

# Install remaining Python dependencies
RUN pip install plyfile tqdm open3d trimesh lpips scikit-image opencv-python-headless "numpy<2.0"

CMD ["bash"]

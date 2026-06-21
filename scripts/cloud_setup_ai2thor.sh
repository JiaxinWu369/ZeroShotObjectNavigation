#!/usr/bin/env bash
set -euo pipefail

APT_PREFIX=()
if [ "$(id -u)" -ne 0 ]; then
    APT_PREFIX=(sudo)
fi

"${APT_PREFIX[@]}" apt update
"${APT_PREFIX[@]}" apt install -y \
    xvfb \
    x11-utils \
    mesa-utils \
    libgl1 \
    libglu1-mesa \
    libglib2.0-0 \
    libxrender1 \
    libxext6 \
    libsm6 \
    libxi6 \
    libxrandr2 \
    libxcursor1 \
    libxinerama1 \
    libegl1 \
    libosmesa6 \
    libnss3 \
    libgconf-2-4 \
    libasound2

python -m pip install \
    numpy \
    pandas \
    pyyaml \
    tqdm \
    matplotlib \
    networkx \
    pytest \
    ai2thor

python --version
pip --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
nvidia-smi
python -c "import ai2thor; print(ai2thor.__version__)"

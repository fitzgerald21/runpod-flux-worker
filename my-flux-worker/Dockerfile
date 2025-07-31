# Use a currently available and stable NVIDIA PyTorch image as a base
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

# Set the working directory
WORKDIR /

# --- Step 1: Install System Dependencies ---
# We need to install sudo and other tools as this is a more minimal base image.
RUN echo "Starting Step 1: Installing system dependencies with apt-get" && \
    apt-get update && \
    apt-get install -y git wget libgl1-mesa-glx libglib2.0-0 sudo && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "Finished Step 1."

# --- Step 2: Clone ComfyUI Repository ---
RUN echo "Starting Step 2: Cloning ComfyUI repository" && \
    git clone https://github.com/comfyanonymous/ComfyUI.git && \
    echo "Finished Step 2."

# Set the working directory to ComfyUI
WORKDIR /ComfyUI

# --- Step 3: Install Python Dependencies ---
RUN echo "Starting Step 3: Installing Python requirements for ComfyUI" && \
    pip install --no-cache-dir -r requirements.txt && \
    echo "Finished Step 3."

# --- Step 4: Download Models ---
RUN echo "Starting Step 4: Downloading models" && \
    wget -O ./models/unet/flux1-dev-kontext_fp8_scaled.safetensors https://huggingface.co/black-forest-labs/FLUX.1-kontext-fp8/resolve/main/flux1-dev-kontext_fp8_scaled.safetensors && \
    wget -O ./models/clip/t5xxl_fp16.safetensors https://huggingface.co/black-forest-labs/FLUX.1-kontext-fp8/resolve/main/t5xxl_fp16.safetensors && \
    wget -O ./models/clip/clip_l.safetensors https://huggingface.co/black-forest-labs/FLUX.1-kontext-fp8/resolve/main/clip_l.safetensors && \
    wget -O ./models/vae/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-kontext-fp8/resolve/main/ae.safetensors && \
    echo "Finished Step 4."

# Switch back to the root directory for the worker code
WORKDIR /

# --- Step 5: Copy Worker Code ---
RUN echo "Starting Step 5: Copying worker source code"
COPY ./src /src
COPY ./workflow_api.json /workflow_api.json
RUN echo "Finished Step 5."

# Set the command to run the Runpod handler
CMD ["python", "-u", "/src/rp_handler.py"]

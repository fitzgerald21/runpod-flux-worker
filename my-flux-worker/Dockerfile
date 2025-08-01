# Start from a base image provided by the runpod-workers-comfyui project.
# This image contains a clean ComfyUI installation and the necessary runpod handler setup.
FROM runpod/worker-comfyui:latest-base

# Switch to root user to install packages
USER root

# Install wget, a tool for downloading files from the web
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

# Switch back to the default user
USER 1000

# Set the working directory to the ComfyUI installation
WORKDIR /app/ComfyUI

# Create the necessary model directories if they don't exist
RUN mkdir -p models/diffusion_models models/clip models/t5 models/vae

# Download the FLUX Kontext FP8 UNet model
RUN wget -O models/diffusion_models/flux1-kontext-dev-fp8-e4m3fn.safetensors https://huggingface.co/6chan/flux1-kontext-dev-fp8/resolve/main/flux1-kontext-dev-fp8-e4m3fn.safetensors

# Download the CLIP-L text encoder
RUN wget -O models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors

# Download the T5-XXL text encoder (FP8 version)
RUN wget -O models/t5/t5xxl_fp8_e4m3fn_scaled.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn_scaled.safetensors

# Download the VAE model
RUN wget -O models/vae/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev/resolve/main/ae.safetensors

# The base image has an entrypoint that will automatically start the RunPod handler script.

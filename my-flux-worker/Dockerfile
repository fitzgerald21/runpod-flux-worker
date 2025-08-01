# Use an official RunPod PyTorch base image for compatibility and speed
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set environment variables to prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Update package lists and install essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the Python dependencies list into the container
COPY requirements.txt.

# Install the Python packages
# --no-cache-dir reduces image size by not storing the pip cache
# --prefer-binary can speed up installation for packages with pre-compiled wheels
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy the rest of the application code into the container
COPY..

# Command to run the worker script when the container starts
# The -u flag ensures that Python output is unbuffered and streams to logs immediately
CMD ["python", "-u", "handler.py"]

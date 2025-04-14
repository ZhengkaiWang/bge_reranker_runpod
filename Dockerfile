FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /app/src/
COPY test_input.json /app/

# Set the entrypoint
CMD ["python", "-u", "src/rp_handler.py"]

# Use Python slim image
FROM python:3.11-slim

# Install system dependencies for OpenCV and ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/downloads data/output

# Expose port (Render sets $PORT)
EXPOSE 8000

# Run the application
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}

FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory (opsional)
WORKDIR /

# Copy main.py ke root container
COPY main.py /main.py

# Copy folder app dan file requirements
COPY app/ /app/
COPY requirements.txt /requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /requirements.txt

# Expose port
EXPOSE 8080

# Jalankan app dari root
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

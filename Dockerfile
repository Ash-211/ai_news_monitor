FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

WORKDIR /app

# Install system dependencies for PostgreSQL driver and newspaper3k (lxml) compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libxml2-dev \
    libxslt-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Step 1: Pre-install CPU-only PyTorch directly to ensure it does not fetch the GPU version
RUN pip install --no-cache-dir torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu

# Step 2: Install remaining requirements (pip will skip torch since it's already installed)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model during build so it is baked into the container
RUN python -m spacy download en_core_web_sm

# Copy models and source files
COPY models/ ./models/
COPY src/ ./src/

# Set up user permissions for Hugging Face (which runs container as UID 1000)
RUN useradd -m -u 1000 user && \
    chown -R user:user /app
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

EXPOSE 7860

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]

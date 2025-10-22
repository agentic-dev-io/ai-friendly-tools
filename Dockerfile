FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Copy package files
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

# Install the package using pip
RUN pip install -e .

# Create directory for logs and config
RUN mkdir -p /root/aift/logs

# Set default command
ENTRYPOINT ["aift"]
CMD ["--help"]

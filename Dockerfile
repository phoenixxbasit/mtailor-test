# Base image with Python 3.10
FROM python:3.10-bookworm

# Use dumb-init to handle signals properly
RUN apt-get update && apt-get install -y dumb-init curl && \
    update-ca-certificates

# Set working directory
WORKDIR /app

# Copy all project files
COPY . .

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose the port used by FastAPI
EXPOSE 8192

# Healthcheck for Cerebrium monitoring
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8192/health || exit 1

# Run the server with dumb-init to forward signals
CMD ["dumb-init", "--", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8192"]

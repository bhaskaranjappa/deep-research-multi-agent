# Use an explicit, lightweight Python base image
FROM python:3.11-slim

# Set system environment optimization flags
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Establish isolated working directory
WORKDIR /app

# Install system dependencies needed for network handling and scraping tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies cleanly without caching overhead
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source elements
COPY main.py agent.py tools.py app.py.

# Expose production port
EXPOSE 8501

# Run via optimized Uvicorn production worker layout
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
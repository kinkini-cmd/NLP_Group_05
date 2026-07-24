# Base Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies (important for NLP packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project into container
COPY . .

# Download NLTK data (important for your preprocessing)
RUN python -m nltk.downloader stopwords wordnet omw-1.4

# Expose Flask/FastAPI port
EXPOSE 5000

# Run the application
CMD ["python", "run.py"]
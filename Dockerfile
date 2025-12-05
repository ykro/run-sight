FROM python:3.12-slim

WORKDIR /app

# Install system dependencies if needed (none strictly for these python packages, maybe git)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy individual files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY utils.py .
COPY app.py .
COPY analysis.md .
COPY .streamlit .streamlit


# Expose Streamlit port
EXPOSE 8080

# Environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run
CMD ["streamlit", "run", "app.py"]

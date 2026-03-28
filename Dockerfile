# Use lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files into container
COPY . .

# Install dependencies
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

# Streamlit config (HF mapped internal port)
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# API Configuration
ENV API_PORT=8000
ENV API_ADDRESS=127.0.0.1

# Ensure execution capability on the launcher
RUN chmod +x start.sh

# Expose Space port (FastAPI internal bind doesn't need expose out of host)
EXPOSE 7860

# Run Orchestrator
CMD ["./start.sh"]

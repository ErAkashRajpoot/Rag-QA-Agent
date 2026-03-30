# Use lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files into container
COPY . .

# Install dependencies
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

# Set PYTHONPATH so 'app' package is resolvable
ENV PYTHONPATH=/app:$PYTHONPATH

# Direct mode: Streamlit imports the RAG engine directly (no separate FastAPI process)
ENV DEPLOYMENT_MODE=direct

# Streamlit settings
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Expose HF Spaces port
EXPOSE 7860

# Run Streamlit directly
CMD ["streamlit", "run", "app/frontend/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]

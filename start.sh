#!/bin/bash
# Start FastAPI backend in the background
echo "Starting FastAPI Backend..."
uvicorn app.main:api_handler --host 0.0.0.0 --port 8000 &

# Start Streamlit frontend in the foreground
echo "Starting Streamlit Frontend..."
streamlit run app/frontend/streamlit_app.py --server.port 7860 --server.address 0.0.0.0

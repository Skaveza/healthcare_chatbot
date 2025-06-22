# base image
FROM python:3.9-slim

WORKDIR /app

# Copy files and install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Set HOME env var to /app 
ENV HOME=/app

# Optional: Also set Streamlit config directory explicitly
ENV STREAMLIT_CONFIG_DIR=/app/.streamlit

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

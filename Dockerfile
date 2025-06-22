# base image
FROM python:3.9-slim

WORKDIR /app

# Copy files and install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Create .streamlit config directory in /app and set permissions
RUN mkdir -p /app/.streamlit && chmod -R 755 /app/.streamlit

# Set environment variables
ENV HOME=/app
ENV STREAMLIT_CONFIG_DIR=/app/.streamlit
ENV XDG_CONFIG_HOME=/app/.streamlit  # Add this line

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

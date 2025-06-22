# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Create .streamlit config directory and add a config file
RUN mkdir -p /app/.streamlit && \
    echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = 8501\n\
" > /app/.streamlit/config.toml

# Set environment variables
ENV HOME=/app
ENV STREAMLIT_CONFIG_DIR=/app/.streamlit
ENV XDG_CONFIG_HOME=/app/.streamlit

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables early so Streamlit uses the right config path
ENV HOME=/app
ENV STREAMLIT_CONFIG_DIR=/app/.streamlit
ENV XDG_CONFIG_HOME=/app/.streamlit

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Create .streamlit config directory, set permissions, and create config.toml
RUN mkdir -p /app/.streamlit && \
    chmod -R 755 /app/.streamlit && \
    echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = 8501\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
" > /app/.streamlit/config.toml

# Expose the Streamlit port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

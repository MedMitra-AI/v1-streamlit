# Use a lightweight Python base image
FROM python:3.12.7-slim

# Set a working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . /app

# Expose the Streamlit default port
EXPOSE 8501

# Run Streamlit with your actual script name
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

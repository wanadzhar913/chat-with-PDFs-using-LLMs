# Use the official Python image as the base
FROM python:3.11-slim

# Set environment variables
ENV OPENAI_API_KEY=""

# Create and set the working directory in the container
WORKDIR /app

# Copy only the requirements file initially to leverage Docker's caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY app.py .
COPY .streamlit/config.toml .

# Setup an app user so the container doesn't run as the root user
RUN useradd app
USER app

# Expose the port on which Streamlit runs
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

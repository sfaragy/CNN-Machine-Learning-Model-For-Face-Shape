# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgtk2.0-dev \
    libopencv-dev \
    wget \
    git

# Install Python dependencies
RUN pip install tensorflow keras numpy opencv-python Flask

# Copy the current directory contents into the container at /app
COPY . /app

# Set the working directory to /app
WORKDIR /app

# Expose the application port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]

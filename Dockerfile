# Use an official Python runtime as a parent image
FROM python:3.7-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Make port 8000 available to the world outside this container
EXPOSE 7000

# Define environment variable
ENV NAME hyh

# Run app.py when the container launches
CMD ["python", "app.py"]


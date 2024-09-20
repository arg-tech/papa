# "slim" means it's lightweight, so we avoid bloating the container unnecessarily.
FROM python:3.9-slim

# Setting up the work directory inside the container to /app.
# This is where all your code will live when inside the container.
WORKDIR /app

# Copy everything from the current directory (your project) 
# into the /app directory inside the container. 
# This includes your Python scripts and other project files.
COPY . /app

# Time to install dependencies! This is telling pip to install any libraries 
# listed in requirements.txt
# The --no-cache-dir option keeps the image smaller by avoiding the caching of package installation files.
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available so that your app can be accessed from outside the container.
EXPOSE 80

# Just little tweaks to make the container run a bit smoother.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Finally, the big moment—when the container starts.
# This will need to be changed!!!!
CMD ["python", "src/papa.py"]

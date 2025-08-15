# Custom Langflow Dockerfile
FROM langflowai/langflow:1.4.2

# Set working directory
WORKDIR /app

# Copy requirements file first (for better Docker layer caching)
COPY requirements.txt /app/requirements.txt

# Install custom Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Create custom components directory
RUN mkdir -p /app/custom_components

# Copy custom components
COPY app/custom_components/ /app/custom_components/

# Set permissions
#RUN chmod -R 755 /app/custom_components

# Set environment variable for custom components path
ENV LANGFLOW_COMPONENTS_PATH=/app/custom_components

# Expose Langflow port
EXPOSE 7860

# Default command - no custom entrypoint needed
CMD ["langflow", "run", "--host", "0.0.0.0", "--port", "7860"]
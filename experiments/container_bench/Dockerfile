FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-container.txt .
RUN pip install --no-cache-dir -r requirements-container.txt
COPY . .
CMD ["python", "container_bench.py"]

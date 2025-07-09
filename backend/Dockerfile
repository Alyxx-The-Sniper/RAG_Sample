FROM python:3.11-slim 

# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements file early
COPY requirements.txt .

# Upgrade pip/setuptools/wheel + install torch explicitly first
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code (see build context in docker compose)
COPY  . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

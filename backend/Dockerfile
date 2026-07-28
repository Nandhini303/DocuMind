# DocuMind AI — single-container full-stack image
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code + static frontend
COPY main.py .
COPY static ./static

# AWS App Runner / most PaaS defaults expect the app to listen on 8080
ENV PORT=8080
EXPOSE 8080

# Never bake secrets into the image — GEMINI_API_KEY is injected at
# runtime via the platform's environment variable settings.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]

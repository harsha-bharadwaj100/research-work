FROM node:20 AS frontend-builder

WORKDIR /app/frontend

COPY frontend/package*.json ./

RUN npm install

COPY frontend/ .

RUN npm run build


FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Copy backend
COPY backend/ ./backend/

# Copy frontend dist from builder
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Expose port
EXPOSE 8000

# Start backend
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}

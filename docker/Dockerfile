FROM python:3.11-slim
# TODO: add install, copy, expose, entrypoints
FROM python:3.11-slim

# Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Crear directorio de trabajo
WORKDIR /app

# Copiar requirements e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY src/ src/
COPY data/ data/
COPY tests/ tests/
COPY artifacts/ artifacts/

# Exponer puerto (para FastAPI)
EXPOSE 8000

# Entrypoint por defecto: entrenamiento
CMD ["python", "-m", "src.train", "--data", "data/customer_churn_synth.csv", "--outdir", "artifacts/"]

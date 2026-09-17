FROM python:3.10-slim

WORKDIR /app

# Instalar utilitarios del sistema y compiladores necesarios
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
COPY quant_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la plataforma cuantitativa
COPY quant_app/ /app/quant_app/

WORKDIR /app/quant_app

# Railway inyecta dinámicamente la variable de entorno PORT
ENV PORT=8501
EXPOSE 8501

# Ejecutar Streamlit escuchando en $PORT y en 0.0.0.0
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false"]

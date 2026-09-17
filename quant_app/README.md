# 📈 AI Quantitative Investment & Portfolio Platform (Dockerized)

Plataforma Web interactiva basada en **Streamlit**, **Docker**, **Yahoo Finance (yfinance)** y modelos avanzados de **Machine Learning, Deep Learning y Finanzas Cuantitativas**.

---

## 🏛️ Características y Arquitectura del Sistema

1. **Datos de Mercado Gratuitos en Tiempo Real (Yahoo Finance)**:
   - Ingesta automática mediante `yfinance` para cualquier ticker (`AAPL`, `MSFT`, `NVDA`, `GOOGL`, `TLT`, etc.).
   - Modelado de microestructura: estimación de spreads bid-ask y volatilidad local.
   - Generador sintético GBM calibrado de contingencia autónoma si la API de Yahoo experimenta rate-limiting o interrupciones.

2. **Aversión al Riesgo de Arrow-Pratt ($\gamma \in [1.0, 10.0]$)**:
   - Cuestionario interactivo psicométrico y financiero de 5 dimensiones (horizonte temporal, tolerancia a drawdowns, objetivos, estabilidad de ingresos, experiencia de mercado).
   - Calibración continua del coeficiente relativo de aversión al riesgo $\gamma$.

3. **Optimización Cuantitativa de Utilidad & Restricción Presupuestaria**:
   - Maximización de Utilidad Esperada cuadrática:
     $$\max_{w} \quad w^T \mu - \frac{\gamma}{2} w^T \Sigma_{\text{LW}} w - \sum_{i=1}^N \left( c_{\text{broker}} |w_i - w_{0,i}| + \frac{\text{Spread}_i}{2} |w_i| \right)$$
   - **Regularización de Ledoit-Wolf**: Contracción óptima de covarianza que mitiga el sobreajuste muestral y evita carteras espurias hiperconcentradas.
   - **Asignación Discreta**: Conversión exacta de ponderaciones continuas a títulos enteros de acciones dado un presupuesto en \$ USD, calculando el remanente en efectivo (*cash buffer*).
   - Soporte opcional para venta en corto (*Short Selling*).

4. **Dinámica Analítica de Renta Fija (Bonds)**:
   - Cálculo analítico de Duración de Macaulay, Duración Modificada ($D^*$), Convexidad ($C$) y DV01 ($ por bp).
   - Simulación interactiva de Shocks de Rendimiento ($\Delta y \in [-200, +200]$ bps) comparando el precio exacto contra la aproximación de Taylor de primer orden (lineal) y segundo orden (convexidad).

5. **Señales Direccionales ML & Maduración de Posición**:
   - **Diferenciación Fraccionaria ($d^*=0.40$)**: Conserva memoria de soporte/resistencia mientras elimina la no-estacionariedad.
   - **Machine Learning**: Clasificador de bosque aleatorio (*Random Forest*) que proyecta señales direccionales (*LONG*, *SHORT*, *NEUTRAL*) con probabilidades calibradas.
   - **Semivida de Ornstein-Uhlenbeck (Half-Life)**: Determina matemáticamente cuántos días de mercado esperar para que madure el trade o revierta al equilibrio.
   - **Niveles Dinámicos de Salida**: Take-profit ($\pm 3\sigma$) y Stop-loss dinámico ($\pm 2\sigma$).

6. **Laboratorio Didáctico de Modelos**:
   - Fórmulas matemáticas en KaTeX y explicaciones pedagógicas de nivel posgrado de cada modelo implementado.

---

## 🐳 Despliegue con Docker y Docker Compose

### Opción 1: Con Docker Compose (Recomendado)

Desde la carpeta `quant_app/`:

```bash
docker-compose up --build
```

La aplicación compilará la imagen optimizada `python:3.10-slim`, instalará las dependencias y se levantará en:
👉 **`http://localhost:8501`**

Para detener el contenedor:
```bash
docker-compose down
```

### Opción 2: Con Docker CLI directo

```bash
# Construir la imagen
docker build -t quant-app .

# Ejecutar el contenedor en segundo plano en el puerto 8501
docker run -d -p 8501:8501 --name quant-app-container quant-app
```

---

## 💻 Ejecución Local (Sin Docker)

Si prefiere ejecutarlo directamente en su entorno local con Python:

```bash
# 1. Asegurar la instalación de dependencias
pip install -r requirements.txt

# 2. Iniciar la aplicación Streamlit
streamlit run app.py
```

En sistemas Windows con múltiples versiones de Python instaladas:
```powershell
& "C:\Users\ThinkPad\AppData\Local\Python\pythoncore-3.10-64\python.exe" -m streamlit run app.py
```

---

## 📁 Estructura del Proyecto

```
quant_app/
├── Dockerfile                  # Contenedor Docker basado en python:3.10-slim
├── docker-compose.yml          # Orquestación de servicio multicontenedor (puerto 8501)
├── requirements.txt            # Dependencias fijadas (Streamlit, Plotly, yfinance, PyTorch, etc.)
├── app.py                      # Interfaz web principal de Streamlit con tabs temáticos
├── README.md                   # Documentación técnica de ejecución y modelos
└── modules/
    ├── __init__.py
    ├── data_loader.py          # Extractor de cotizaciones Yahoo Finance y spreads
    ├── risk_profiler.py        # Estimador psicométrico de aversión al riesgo gamma
    ├── portfolio_optimizer.py  # Optimizador con Ledoit-Wolf, presupuesto y comisiones
    ├── fixed_income.py         # Análisis de duración, convexidad y shocks de tasas
    ├── portfolio_monte_carlo.py# Simulación Monte Carlo multivariada de todo el portafolio (Cholesky)
    ├── deep_learning_model.py  # Red Neuronal BiLSTM con Atención Temporal (PyTorch)
    └── timing_signals.py       # Señales de ML fraccionario y maduración Ornstein-Uhlenbeck
```

---

## 🏛️ Afiliación Institucional & Licencia de Propiedad Intelectual

* **Institución Académica**: Desarrollado para el **Taller de Finanzas Cuantitativas** en la **Pontificia Universidad Javeriana de Cali - Colombia**.
* **Autoría & Propiedad Intelectual**: Todos los derechos reservados &copy; 2026 **Rodney Menezes**.
* **Aviso Legal**: El diseño conceptual, arquitectura cuantitativa, algoritmos econométricos, modelos de Deep Learning e implementaciones de código de esta plataforma son propiedad intelectual de Rodney Menezes. Su uso está destinado a propósitos educativos y de investigación para la Pontificia Universidad Javeriana de Cali. Queda prohibida su copia, distribución o comercialización sin autorización expresa por escrito.


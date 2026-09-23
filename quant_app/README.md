# 📈 AI Quantitative Investment & Portfolio Platform (Dockerized)

Plataforma Web interactiva basada en **Streamlit**, **Docker**, **Yahoo Finance (yfinance)** y modelos avanzados de **Machine Learning, Deep Learning y Finanzas Cuantitativas**.

---

## 🏛️ Características y Arquitectura del Sistema

1. **Datos de Mercado Gratuitos en Tiempo Real (Yahoo Finance)**:
   - Ingesta automática mediante `yfinance` para cualquier ticker (`AAPL`, `NVDA`, `IWM`, `TLT`, `BND`, etc.).
   - Modelado de microestructura: estimación de spreads bid-ask y volatilidad local.
   - Generador sintético GBM calibrado de contingencia autónoma si la API de Yahoo experimenta rate-limiting o interrupciones.

2. **Aversión al Riesgo de Arrow-Pratt ($\gamma \in [1.0, 10.0]$)**:
   - Cuestionario interactivo psicométrico y financiero de 5 dimensiones.
   - Calibración continua del coeficiente relativo de aversión al riesgo $\gamma$.

3. **Optimización Multi-Paradigma de Portafolios**:
   - **Markowitz con Ledoit-Wolf**: Maximización de utilidad cuadrática con contracción analítica de covarianza y asignación discreta exacta en títulos enteros de acciones (\$ USD).
   - **Hierarchical Risk Parity (HRP)**: Machine Learning no supervisado para asignación óptima de capital. Agrupa activos mediante clustering jerárquico (*single linkage*), cuasi-diagonalización y bisección recursiva con dendrograma interactivo.
   - **Black-Litterman con Vistas de Deep Learning (IA)**: Fusión bayesiana entre el equilibrio de mercado CAPM ($\Pi$) y vistas direccionales probabilísticas extraídas de la Red Neuronal BiLSTM ($P, Q, \Omega$).
   - **Simulación Monte Carlo Multivariada**: Proyección estocástica del valor de todo el portafolio mediante descomposición de Cholesky $\Sigma_{\text{LW}} = L L^T$.

4. **Backtesting Histórico Walk-Forward & Underwater Plot**:
   - Simulación histórica de la estrategia frente al Benchmark (`SPY` o 60/40).
   - Métricas cuantitativas avanzadas: CAGR, Volatilidad anualizada, **Ratio de Sharpe**, **Ratio de Sortino** (penalización asimétrica de pérdidas) y **Ratio de Calmar** (velocidad de recuperación de drawdown).
   - Gráfico interactivo **Underwater Drawdown** para evaluar la severidad de caídas históricas.

5. **Stress-Testing de Crisis Macroeconómicas & Regímenes de Mercado (GMM)**:
   - Simulación de impacto patrimonial (\$ USD) ante 4 eventos de cola sistémicos: *Subprime 2008*, *COVID-19 2020*, *FED Rate Hikes 2022* y *AI Tech Rally 2023–2024*.
   - Auditoría desglosada activo por activo frente a una cartera 60/40.
   - **Detección No Supervisada de Regímenes (Gaussian Mixture Models)**: Clasificación bayesiana en tiempo real de estados de mercado (*Bull*, *Lateral*, *Bear*) y recomendaciones dinámicas de ajuste de riesgo.

6. **Dinámica Analítica de Renta Fija (Bonds)**:
   - Incorporación directa de ETFs de renta fija cotizados en Yahoo Finance (`BND`, `TLT`, `SHY`, `AGG`, etc.).
   - Duración Modificada Efectiva ($D^*$), Convexidad ($C$) y DV01.
   - Simulación interactiva de Shocks de Rendimiento ($\Delta y \in [-400, +400]$ bps).

7. **Señales Deep Learning & Maduración de Posición**:
   - Red Neuronal Recurrente BiLSTM con Auto-Atención Temporal (PyTorch) entrenada sobre precios, retornos y memoria fraccionaria ($d^*=0.40$).
   - Señales *LONG*, *SHORT*, *NEUTRAL*, Stop-Loss/Take-Profit dinámicos y semivida de Ornstein-Uhlenbeck.

8. **Laboratorio Didáctico de Modelos**:
   - 5 experimentos interactivos con fórmulas KaTeX y manipuladores en vivo (Ledoit-Wolf, Frontera Eficiente, Convexidad de Bonos, Memoria Fraccionaria, Procesos Monte Carlo).

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
    ├── portfolio_optimizer.py  # Optimizador Markowitz con Ledoit-Wolf, presupuesto y comisiones
    ├── hrp_optimizer.py        # Hierarchical Risk Parity (HRP) & Dendrograma Plotly
    ├── backtester.py           # Backtesting walk-forward, Sortino, Calmar & Underwater plot
    ├── black_litterman.py      # Black-Litterman impulsado por vistas de IA (BiLSTM)
    ├── stress_testing.py       # Stress-testing de 4 crisis sistémicas y desglose USD
    ├── market_regimes.py       # Regímenes de mercado GMM no supervisados (Bull/Lateral/Bear)
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


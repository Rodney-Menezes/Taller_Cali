# 🏛️ Plataforma Institucional de Finanzas Cuantitativas & Gestión de Inversiones

Plataforma analítica e interactiva basada en **Streamlit**, **Docker**, **Yahoo Finance (yfinance)** y modelos avanzados de **Machine Learning, Deep Learning y Finanzas Cuantitativas**.

Desarrollada para el **Taller de Finanzas Cuantitativas** en la **Pontificia Universidad Javeriana de Cali - Colombia**.  
**Autoría & Propiedad Intelectual**: Rodney Menezes © 2026. Todos los derechos reservados.

---

## 🏛️ Características y Arquitectura del Sistema

La plataforma integra ocho (8) paradigmas y modelos analíticos de frontera financiera:

1. **Datos de Mercado Gratuitos en Tiempo Real (Yahoo Finance)**:
   - Ingesta automática mediante `yfinance` para cualquier ticker (`AAPL`, `NVDA`, `IWM`, `TLT`, `BND`, etc.).
   - Modelado de microestructura: estimación de spreads bid-ask y volatilidad local.
   - Generador sintético GBM calibrado de contingencia autónoma si la API experimenta limitaciones de tasa.

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
   - Gráfico interactivo **Underwater Drawdown** para evaluar la severidad y duración de caídas históricas.

5. **Stress-Testing de Crisis Macroeconómicas & Regímenes de Mercado (GMM)**:
   - Simulación de impacto patrimonial (\$ USD) ante 4 eventos de cola sistémicos: *Subprime 2008*, *COVID-19 2020*, *FED Rate Hikes 2022* y *AI Tech Rally 2023–2024*.
   - Auditoría desglosada activo por activo frente a una cartera 60/40.
   - **Detección No Supervisada de Regímenes (Gaussian Mixture Models)**: Clasificación bayesiana en tiempo real de estados de mercado (*Bull*, *Lateral*, *Bear*) y recomendaciones dinámicas de ajuste de riesgo.

6. **Dinámica Analítica de Renta Fija (Bonds)**:
   - Incorporación directa de ETFs de renta fija cotizados en Yahoo Finance (`BND`, `TLT`, `SHY`, `AGG`, etc.).
   - Duración Modificada Efectiva ($D^*$), Convexidad ($C$) y DV01.
   - Simulación interactiva de Shocks de Rendimiento ($\Delta y \in [-400, +400]$ bps) con comparación entre re-pricing exacto y aproximaciones de Taylor de primer y segundo orden.

7. **Señales Deep Learning & Maduración de Posición**:
   - Red Neuronal Recurrente BiLSTM con Auto-Atención Temporal (PyTorch) entrenada sobre precios, retornos y memoria fraccionaria ($d^*=0.40$).
   - Señales direccionales (*LONG*, *SHORT*, *NEUTRAL*), Stop-Loss/Take-Profit dinámicos y semivida de reversión a la media basada en el proceso estocástico de Ornstein-Uhlenbeck.

8. **Laboratorio Didáctico de Modelos Cuantitativos**:
   - Experimentos interactivos con fórmulas matemáticas y simuladores dinámicos en vivo (Ledoit-Wolf, Frontera Eficiente, Convexidad de Bonos, Memoria Fraccionaria, Procesos Monte Carlo).

---

## 📁 Estructura del Repositorio

```
Taller_Cali/
├── Dockerfile                  # Contenedor raíz optimizado para despliegue en Railway / Docker
├── railway.json                # Configuración de despliegue continuo en Railway
├── LICENSE                     # Licencia de uso
├── README.md                   # Documentación principal del taller
├── graficos/                   # Directorio de recursos visuales
└── quant_app/                  # Aplicación cuantitativa completa
    ├── Dockerfile              # Dockerfile local de la app
    ├── docker-compose.yml      # Orquestación de servicio (puerto 8501)
    ├── requirements.txt        # Dependencias fijadas (Streamlit, Plotly, PyTorch, SciPy, etc.)
    ├── app.py                  # Interfaz web principal de Streamlit
    ├── README.md               # Documentación interna de quant_app
    └── modules/
        ├── __init__.py
        ├── data_loader.py          # Extractor de cotizaciones y microestructura
        ├── risk_profiler.py        # Estimador psicométrico de aversión al riesgo gamma
        ├── portfolio_optimizer.py  # Optimizador Markowitz con Ledoit-Wolf y fricciones
        ├── hrp_optimizer.py        # Hierarchical Risk Parity (HRP) y Dendrograma
        ├── backtester.py           # Backtesting walk-forward, Sortino, Calmar y Underwater
        ├── black_litterman.py      # Black-Litterman impulsado por vistas de IA (BiLSTM)
        ├── stress_testing.py       # Stress-testing de 4 crisis sistémicas y desglose USD
        ├── market_regimes.py       # Regímenes de mercado GMM no supervisados
        ├── fixed_income.py         # Análisis analítico de duración, convexidad y shocks
        ├── portfolio_monte_carlo.py# Simulación Monte Carlo multivariada (Cholesky)
        ├── deep_learning_model.py  # Red Neuronal BiLSTM con Atención Temporal (PyTorch)
        └── timing_signals.py       # Señales de ML fraccionario y maduración Ornstein-Uhlenbeck
```

---

## 🐳 Despliegue con Docker y Docker Compose

### Opción 1: Desde la carpeta `Taller_Cali` (Despliegue General)

```bash
docker build -t taller-cali-app .
docker run -p 8501:8501 taller-cali-app
```

### Opción 2: Con Docker Compose (Recomendado para Desarrollo)

Desde la subcarpeta `quant_app/`:

```bash
cd quant_app
docker compose up --build
```

La aplicación se compilará con `python:3.10-slim` y estará disponible en:
👉 **`http://localhost:8501`**

Para detener el servicio:
```bash
docker compose down
```

---

## 🚀 Despliegue en la Nube (Railway)

El repositorio incluye la configuración de producción `railway.json` que orquesta la compilación del `Dockerfile` raíz y enlaza automáticamente la variable de entorno `$PORT` para servir la plataforma en la nube sin configuración manual adicional.

---

## 💻 Ejecución Local (Sin Contenedores)

Si prefiere ejecutarlo directamente en su entorno local con Python:

```bash
# 1. Acceder al directorio de la aplicación
cd quant_app

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Iniciar Streamlit
streamlit run app.py
```

En entornos Windows con Python 3.10:
```powershell
& "C:\Users\ThinkPad\AppData\Local\Python\pythoncore-3.10-64\python.exe" -m streamlit run app.py
```

---

## 🏛️ Afiliación Institucional & Licencia de Propiedad Intelectual

* **Institución Académica**: Desarrollado para el **Taller de Finanzas Cuantitativas** en la **Pontificia Universidad Javeriana de Cali - Colombia**.
* **Autoría & Propiedad Intelectual**: Todos los derechos reservados &copy; 2026 **Rodney Menezes**.
* **Aviso Legal**: El diseño conceptual, arquitectura cuantitativa, algoritmos econométricos, modelos de Deep Learning e implementaciones de código de esta plataforma son propiedad intelectual de Rodney Menezes. Su uso está destinado a propósitos educativos y de investigación para la Pontificia Universidad Javeriana de Cali. Queda prohibida su copia, distribución o comercialización sin autorización expresa por escrito.

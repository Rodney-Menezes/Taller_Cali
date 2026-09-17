# Masterclass Enciclopédica: Inteligencia Artificial en Finanzas Cuantitativas (120 Minutos)
## De la Econometría Estocástica y Machine Learning Clásico a Deep Learning y Agentes Autónomos (DRL)

Guía docente y pedagógica completa para la impartición de la Masterclass de 2 horas. Cubre la taxonomía integral de la IA financiera, derivaciones matemáticas rigurosas, benchmarking multi-modelo y agentes autónomos de trading y validación institucional.

---

## 🧭 Taxonomía Cuantitativa: ¿Qué papel juega cada enfoque?

1. **Econometría y Procesos Estocásticos:** Modela dinámicas analíticas continuas mediante Ecuaciones Diferenciales Estocásticas (SDEs) e integrales de Itô (Black-Scholes, Heston SDE, Merton Jump Diffusion).
2. **Machine Learning Clásico:** Algoritmos no paramétricos que aprenden fronteras no lineales y relaciones estadísticas complejas (Random Forest, Bagging, Ridge Regression, Ledoit-Wolf Shrinkage, Diferenciación Fraccionaria de López de Prado).
3. **Deep Learning Supervisado y Auto-Supervisado:** Extracción jerárquica de representaciones latentes mediante tensores y descenso de gradiente (1D Convolutional Denoising Autoencoder, Causal Gated LSTM, Causal Temporal Transformer con Time2Vec).
4. **Modelos Generativos Profundos:** Generación de escenarios de estrés leptocúrticos y colas pesadas mediante aproximación de medidas continuas (Variational Autoencoders VAE, Wasserstein GAN con Gradient Penalty WGAN-GP).
5. **Agentes Autónomos de Reinforcement Learning (DRL):** Control estocástico y toma de decisiones autónomas en entornos Markovianos de mercado (Procesos de Decisión de Markov MDP, Ecuación de Bellman, Deep Q-Networks DQN con Target Network y Replay Buffer).
6. **Validación Institucional:** Control del sesgo de múltiples pruebas (*multiple testing bias*) y fuga de datos temporal (Purged & Embargoed K-Fold Cross Validation, Deflated Sharpe Ratio DSR, Probabilistic Sharpe Ratio PSR).

---

## ⏱️ Cronograma Minuto a Minuto y Mapeo de Modelos (120 min)

### 📊 Módulo 1: Microestructura, Machine Learning Clásico y Deep Learning (00:00 - 00:20)
* **Fundamento Matemático:** Operador binomial $(1-B)^d = \sum_{k=0}^{\infty} (-1)^k \binom{d}{k} B^k$.
* **Modelos Evaluados:**
  * **1A:** Retornos Clásicos ($d=1.0$).
  * **1B:** Fast Fractional Differentiation ($d^*=0.40$).
  * **1C (Machine Learning Clásico):** Ensamble Random Forest / Bagging para señales predictivas sobre features fraccionarias.
  * **1D (Deep Learning):** 1D Convolutional Denoising Autoencoder para filtrado de micro-ruido sin desfasar la serie temporal.

### 📈 Módulo 2: Modelado Dinámico de Volatilidad: De Heston a Transformers (00:20 - 00:45)
* **Fundamento Matemático:** SDEs de Heston con efecto apalancamiento ($\rho = -0.75$), condición de Feller $2\kappa\theta > \xi^2$, y embeddings sinusoidales Time2Vec con atención causal estricta.
* **Modelos Evaluados:**
  * **2A:** Proceso Estocástico de Heston.
  * **2B (ML Clásico):** Ridge Regression sobre momentos rodantes (media, desvío, energía).
  * **2C (Deep Learning Recurrente):** Causal Gated LSTM con compuertas y activación `Softplus`.
  * **2D (Deep Learning Transformer):** Causal Temporal Transformer con Time2Vec continuo.

### 🌪️ Módulo 3: Modelos Generativos y Stress Testing Multiactivo (00:45 - 01:05)
* **Fundamento Matemático:** ELBO de VAEs ($\mathbb{E}[\ln p_\theta] - \mathcal{D}_{KL}$) y dualidad de Kantorovich-Rubinstein para WGAN-GP 1-Lipschitz.
* **Modelos Evaluados:**
  * **3A:** Monte Carlo Multivariado Normal.
  * **3B:** Variational Autoencoder (VAE) con Reparameterization Trick.
  * **3C:** Wasserstein GAN con Gradient Penalty (WGAN-GP).

### 🛡️ Módulo 4: Deep Hedging bajo Saltos de Merton y Costos de Almgren-Chriss (01:05 - 01:30)
* **Fundamento Matemático:** Mercado incompleto bajo Merton Jump Diffusion ($dN_t \sim \text{Poisson}$) y fricciones no lineales de Almgren-Chriss $c = \kappa_1 |\Delta \delta| S + \kappa_2 (\Delta \delta)^2 S$.
* **Modelos Evaluados:**
  * **4A:** Delta Analítico de Black-Scholes.
  * **4B:** Feed-Forward Deep Hedger (MLP).
  * **4C:** Recurrent Deep Hedger (LSTM) minimizando la Medida de Riesgo Entrópica y revelando la **banda de inacción óptima**.

### 🤖 Módulo 5: Agentes Autónomos de Reinforcement Learning (DRL) en Trading (01:30 - 01:50)
* **Fundamento Matemático:** Entorno de Mercado como Proceso de Decisión de Markov (MDP), Ecuación de Optimalidad de Bellman $Q^*(s, a) = R(s, a) + \gamma \max_{a'} Q^*(s', a')$.
* **Agentes Implementados:**
  * **Entorno de Mercado Markoviano:** `FinancialMarketEnv` con retornos, volatilidad, estado de posición previa y comisiones.
  * **Agente Autónomo DQN:** Red profunda $Q(s, a; \theta)$ con Target Network $\theta^-$ y Replay Buffer $\mathcal{D}$ que toma decisiones de posición autónomas (+1 Long, 0 Cash, -1 Short) maximizando el P&L neto.

### 🔍 Módulo 6: Framework Institucional de Validación y Prevención de Overfitting (01:50 - 02:00)
* **Fundamento Matemático:** Purged and Embargoed K-Fold Cross Validation (López de Prado) y Deflated Sharpe Ratio (DSR, Bailey & López de Prado 2014):
  $$PSR(SR_0) = \Phi\left( \frac{(\widehat{SR} - SR_0) \sqrt{T - 1}}{\sqrt{1 - \gamma_3 \widehat{SR} + \frac{\gamma_4 - 1}{4} \widehat{SR}^2}} \right), \quad DSR = PSR(SR^*)$$
* **Código Implementado:** Clase `PurgedKFold` visualizada con scatter temporal y cálculo analítico de PSR y DSR sobre las estrategias entrenadas.

---

## 🚀 Cómo Ejecutar el Notebook
El archivo ya está completamente ejecutado y guardado en disco:
```bash
jupyter notebook taller/Taller_IA_Finanzas_Cuantitativas.ipynb
```

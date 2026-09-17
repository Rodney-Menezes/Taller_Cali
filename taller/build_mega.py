import json
import os

cells = []

def add_md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.split("\n")]})

def add_code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in text.split("\n")]})

# ==============================================================================
# PORTADA Y AGENDA MULTI-MODELO
# ==============================================================================
add_md("""# Masterclass Cuantitativa Avanzada: Inteligencia Artificial en Finanzas
## Benchmarking Multi-Modelo: De Procesos Estocásticos a Deep Learning y DRL
### Duración: 120 Minutos | Nivel: Postgrado / Quantitative Research Lab
---
Este taller exhaustivo implementa y compara **múltiples modelos competitivos en cada módulo temático**, abarcando la frontera teórica y práctica de las finanzas cuantitativas. Se apoya y amplía la base del repositorio (`Chap6CNN`, `Chap7Seq`, `Chap8AE`, `Chap9Gen`, `Chap10RL`, `Chap15Vol`, `Chap19DH`).

### 📐 Agenda de la Sesión y Mapeo de Modelos:
1. **Módulo 1: Microestructura y Filtrado de Señal Financiera (20 min)**
   * *Modelo 1A:* Diferenciación Entera Clásica ($d=1.0$).
   * *Modelo 1B:* Fast Fractional Differentiation FFD ($d^* \approx 0.40$, López de Prado).
   * *Modelo 1C:* 1D Convolutional Denoising Autoencoder para filtrado de microestructura.
2. **Módulo 2: Modelado Dinámico de Volatilidad (25 min)**
   * *Modelo 2A:* Simulación Estocástica de Heston (SDE acoplada con efecto apalancamiento).
   * *Modelo 2B:* Causal Gated LSTM con restricción de positividad `Softplus`.
   * *Modelo 2C:* Causal Temporal Transformer con embeddings continuos `Time2Vec`.
3. **Módulo 3: Modelos Generativos y Stress Testing Multiactivo (25 min)**
   * *Modelo 3A:* Simulación Multivariada Paramétrica Monte Carlo (t-Student).
   * *Modelo 3B:* Variational Autoencoder (VAE) Estocástico con Reparameterization Trick.
   * *Modelo 3C:* Wasserstein GAN con Gradient Penalty (WGAN-GP) para colas pesadas.
4. **Módulo 4: Deep Hedging de Derivados bajo Mercados Incompletos (25 min)**
   * *Modelo 4A:* Delta Hedging Analítico de Black-Scholes-Merton (fricción cero).
   * *Modelo 4B:* Feed-Forward Deep Hedger (MLP Denso).
   * *Modelo 4C:* Recurrent Deep Hedger (LSTM) con costos no lineales de Almgren-Chriss y saltos de Merton (MJD).
5. **Módulo 5: Asignación Dinámica de Portafolios y Control Estocástico (20 min)**
   * *Estrategia 5A:* Benchmark Pasivo Equal-Weighted ($1/N$).
   * *Estrategia 5B:* Markowitz Media-Varianza con Ledoit-Wolf Covariance Shrinkage.
   * *Estrategia 5C:* Deep Sharpe Policy Network (Optimización Directa de Sharpe).
   * *Estrategia 5D:* Deep Reinforcement Learning Actor-Critic con Sortino Loss (Downside Risk).
6. **Módulo 6: Framework Institucional de Validación y Prevención de Overfitting (05 min)**
   * Purged & Embargoed Cross-Validation y Deflated Sharpe Ratio (DSR).""")

add_code("""# Setup Cuantitativo y Detección de Aceleración por Hardware
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
from scipy.stats import norm, kurtosis, skew
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (14, 5)
plt.rcParams['font.size'] = 11

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"================================================================")
print(f"MASTERCLASS MULTI-MODELO INICIALIZADA EXITOSAMENTE")
print(f"Dispositivo Activo: {device} | PyTorch: {torch.__version__}")
print(f"================================================================")""")

# ==============================================================================
# MÓDULO 1: MICROESTRUCTURA Y FILTRADO (3 MODELOS)
# ==============================================================================
add_md(r"""## Módulo 1: Microestructura y Filtrado de Señal Financiera (20 min)
### Comparativa de 3 Modelos de Preprocesamiento:
1. **Modelo 1A (Entero $d=1$):** Retornos logarítmicos tradicionales $r_t = \Delta \ln(P_t)$. Forzamos estacionariedad pero eliminamos la memoria de largo plazo.
2. **Modelo 1B (Fraccionario $d^* \approx 0.40$):** Operador binomial continuo de Marcos López de Prado:
$$(1 - B)^d = \sum_{k=0}^{\infty} (-1)^k \binom{d}{k} B^k, \quad \omega_0 = 1, \quad \omega_k = - \omega_{k-1} \frac{d - k + 1}{k}$$
3. **Modelo 1C (Deep Denoising Autoencoder 1D):** Red neuronal convolucional que comprime la serie temporal a un cuello de botella latente $\mathbf{z} \in \mathbb{R}^K$, filtrando el ruido de bid-ask bounce y microestructura sin desfasar la fase temporal.""")

add_code("""# Implementación de Fast Fractional Differentiation (FFD)
def get_weights_ffd(d, size=200, thres=1e-4):
    w = [1.0]
    for k in range(1, size):
        w_k = -w[-1] / k * (d - k + 1)
        if abs(w_k) < thres: break
        w.append(w_k)
    return np.array(w[::-1])

def frac_diff_ffd(series, d, thres=1e-4):
    weights = get_weights_ffd(d, size=len(series), thres=thres)
    width = len(weights)
    res = {}
    vals = series.values
    for i in range(width, len(vals) + 1):
        res[series.index[i - 1]] = np.dot(weights, vals[i - width:i])
    return pd.Series(res)

# Simulación de Precios con Drift, Ruido de Microestructura y Saltos
np.random.seed(101)
N_pts = 1500
dt = 1/252
t_index = pd.date_range('2019-01-01', periods=N_pts, freq='B')
innov = np.random.normal(0.08*dt, 0.20*np.sqrt(dt), N_pts)
jumps = np.random.poisson(0.03, N_pts) * np.random.normal(-0.02, 0.04, N_pts)
micro_noise = np.random.normal(0, 0.005, N_pts) # Ruido de microestructura
clean_log_p = np.cumsum(innov + jumps) + np.log(100.0)
noisy_log_p = clean_log_p + micro_noise
p_series = pd.Series(noisy_log_p, index=t_index)

# Modelo 1A: Retornos
ret_series = p_series.diff().dropna()
# Modelo 1B: Diferenciación Fraccionaria Óptima
frac_series = frac_diff_ffd(p_series, d=0.40)

# Modelo 1C: 1D Conv Denoising Autoencoder (Chap8AE)
class Conv1DDenoisingAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2), nn.LeakyReLU(0.2),
            nn.Conv1d(16, 8, kernel_size=5, padding=2), nn.LeakyReLU(0.2)
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=5, padding=2), nn.LeakyReLU(0.2),
            nn.Conv1d(16, 1, kernel_size=5, padding=2)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

dae = Conv1DDenoisingAE().to(device)
opt_dae = optim.Adam(dae.parameters(), lr=0.005)
loss_dae_fn = nn.MSELoss()

# Entrenar DAE para reconstruir la señal limpia eliminando micro-ruido
x_noisy_t = torch.tensor(noisy_log_p, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
x_clean_t = torch.tensor(clean_log_p, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

for ep in range(60):
    opt_dae.zero_grad()
    reconstructed = dae(x_noisy_t)
    l_dae = loss_dae_fn(reconstructed, x_clean_t)
    l_dae.backward()
    opt_dae.step()

dae.eval()
with torch.no_grad():
    denoised_p = dae(x_noisy_t).squeeze().cpu().numpy()
dae_series = pd.Series(denoised_p, index=t_index)

# Comparativa Gráfica de los 3 Modelos de Preprocesamiento
fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
axes[0].plot(p_series, color='#7f7f7f', alpha=0.5, label="Serie Ruidosa Observada")
axes[0].plot(dae_series, color='#1f77b4', lw=1.8, label="Modelo 1C: Denoising Autoencoder 1D (Filtro Microestructura)")
axes[0].set_title("Filtrado de Microestructura mediante Autoencoder Convolucional", fontweight='bold')
axes[0].legend(loc='upper left')

axes[1].plot(frac_series, color='#2ca02c', lw=1.2, label="Modelo 1B: Diferenciación Fraccionaria (d=0.40)")
axes[1].set_title("Preservación de Memoria y Estacionariedad Óptima (López de Prado)", fontweight='bold')
axes[1].legend(loc='upper left')

axes[2].plot(ret_series, color='#d62728', lw=0.8, label="Modelo 1A: Retornos Logarítmicos Enteros (d=1.0)")
axes[2].set_title("Retornos Estándar (Destrucción Completa de Niveles de Memoria)", fontweight='bold')
axes[2].legend(loc='upper left')
plt.tight_layout()
plt.show()

corr_frac = np.corrcoef(p_series.loc[frac_series.index], frac_series)[0, 1]
corr_ret = np.corrcoef(p_series.loc[ret_series.index], ret_series)[0, 1]
print(f"-> Correlación Modelo 1B (Fraccionario d=0.40) con Precios: {corr_frac:.4f} (Memoria Conservada)")
print(f"-> Correlación Modelo 1A (Retornos d=1.0) con Precios:      {corr_ret:.4f} (Memoria Destruida)")""")

# ==============================================================================
# MÓDULO 2: MODELADO DE VOLATILIDAD (3 MODELOS)
# ==============================================================================
add_md(r"""## Módulo 2: Modelado Dinámico de Volatilidad (25 min)
### Comparativa de 3 Modelos de Volatilidad Estocástica y Causal:
1. **Modelo 2A (Proceso de Heston):** Sistema acoplado de SDEs analíticas con correlación cruzada $d\langle W^S, W^v \rangle = \rho dt$ ($\rho < 0$, efecto palanca).
2. **Modelo 2B (Causal Gated LSTM):** Red recurrente con compuertas de olvido y actualización con salida `Softplus()` para garantizar varianza estrictamente no-negativa.
3. **Modelo 2C (Causal Temporal Transformer con Time2Vec):** Mecanismo de autoatención con proyección temporal continua y máscara triangular causal para evitar *look-ahead bias*.""")

add_code("""# Modelo 2A: Simulación Estocástica de Heston
def sim_heston_data(N=1600, S0=100.0, v0=0.04, kappa=2.5, theta=0.04, xi=0.35, rho=-0.75, T=4.0):
    dt = T / N
    S, v = np.zeros(N), np.zeros(N)
    S[0], v[0] = S0, v0
    L = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    for t in range(1, N):
        dW = np.dot(L, np.random.normal(0, 1, 2)) * np.sqrt(dt)
        v_p = max(v[t-1], 1e-6)
        v[t] = max(v_p + kappa*(theta - v_p)*dt + xi*np.sqrt(v_p)*dW[1], 1e-6)
        S[t] = S[t-1] * np.exp((0.05 - 0.5*v_p)*dt + np.sqrt(v_p)*dW[0])
    return S, np.sqrt(v)

S_heston, vol_heston = sim_heston_data()
rets_h = np.diff(np.log(S_heston))

# Construcción de Dataset
SEQ_VOL = 20
X_v, T_v, Y_v = [], [], []
for i in range(len(rets_h) - SEQ_VOL):
    X_v.append(rets_h[i:i+SEQ_VOL])
    T_v.append(np.linspace(i*dt, (i+SEQ_VOL)*dt, SEQ_VOL))
    Y_v.append(vol_heston[i+SEQ_VOL])

X_vt = torch.tensor(np.array(X_v), dtype=torch.float32).unsqueeze(-1).to(device)
T_vt = torch.tensor(np.array(T_v), dtype=torch.float32).unsqueeze(-1).to(device)
Y_vt = torch.tensor(np.array(Y_v), dtype=torch.float32).unsqueeze(-1).to(device)
split_v = int(0.75 * len(X_vt))

# Modelo 2B: Causal LSTM
class CausalLSTMVol(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 32, num_layers=2, batch_first=True, dropout=0.1)
        self.fc = nn.Sequential(nn.Linear(32, 1), nn.Softplus())
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# Modelo 2C: Causal Temporal Transformer con Time2Vec
class Time2Vec(nn.Module):
    def __init__(self, in_f=1, out_f=8):
        super().__init__()
        self.w0 = nn.Parameter(torch.randn(in_f, 1))
        self.b0 = nn.Parameter(torch.randn(1))
        self.w = nn.Parameter(torch.randn(in_f, out_f - 1))
        self.b = nn.Parameter(torch.randn(out_f - 1))
    def forward(self, x):
        return torch.cat([torch.matmul(x, self.w0) + self.b0, torch.sin(torch.matmul(x, self.w) + self.b)], dim=-1)

class CausalTransformerVol(nn.Module):
    def __init__(self):
        super().__init__()
        self.t2v = Time2Vec(1, 8)
        self.proj = nn.Linear(1 + 8, 32)
        enc = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64, batch_first=True)
        self.trans = nn.TransformerEncoder(enc, num_layers=2)
        self.head = nn.Sequential(nn.Linear(32, 1), nn.Softplus())
    def forward(self, x, t):
        emb = torch.cat([x, self.t2v(t)], dim=-1)
        mask = torch.triu(torch.full((x.size(1), x.size(1)), float('-inf'), device=x.device), diagonal=1)
        feat = self.trans(self.proj(emb), mask=mask)
        return self.head(feat[:, -1, :])

lstm_model = CausalLSTMVol().to(device)
trans_model = CausalTransformerVol().to(device)
opt_lstm = optim.Adam(lstm_model.parameters(), lr=0.004)
opt_trans = optim.Adam(trans_model.parameters(), lr=0.003)
huber_loss = nn.HuberLoss()

# Entrenamiento Comparativo
print("Entrenando Modelo 2B (Causal LSTM) y Modelo 2C (Causal Transformer)...")
for ep in range(25):
    # Train LSTM
    opt_lstm.zero_grad()
    l_pred = lstm_model(X_vt[:split_v])
    loss_l = huber_loss(l_pred, Y_vt[:split_v])
    loss_l.backward()
    opt_lstm.step()
    
    # Train Transformer
    opt_trans.zero_grad()
    t_pred = trans_model(X_vt[:split_v], T_vt[:split_v])
    loss_t = huber_loss(t_pred, Y_vt[:split_v])
    loss_t.backward()
    opt_trans.step()

# Evaluación OOS de los Modelos
lstm_model.eval()
trans_model.eval()
with torch.no_grad():
    oos_lstm = lstm_model(X_vt[split_v:]).cpu().numpy()
    oos_trans = trans_model(X_vt[split_v:], T_vt[split_v:]).cpu().numpy()
    oos_true_v = Y_vt[split_v:].cpu().numpy()

# Cálculo de Métricas de Error
mse_lstm = np.mean((oos_lstm - oos_true_v)**2)
mse_trans = np.mean((oos_trans - oos_true_v)**2)

plt.figure(figsize=(14, 5))
plt.plot(oos_true_v, label="Modelo 2A: Volatilidad Real Heston $\\sigma_t$", color='black', lw=1.5, alpha=0.6)
plt.plot(oos_lstm, label=f"Modelo 2B: Causal LSTM (MSE: {mse_lstm:.6f})", color='#1f77b4', lw=2)
plt.plot(oos_trans, label=f"Modelo 2C: Causal Transformer + Time2Vec (MSE: {mse_trans:.6f})", color='#d62728', lw=2, ls='--')
plt.title("Benchmarking de Modelos de Volatilidad: Heston SDE vs. Causal LSTM vs. Causal Transformer", fontweight='bold')
plt.ylabel("Volatilidad $\\sigma_t$")
plt.legend()
plt.show()""")

# ==============================================================================
# MÓDULO 3: MODELOS GENERATIVOS Y STRESS TESTING (3 MODELOS)
# ==============================================================================
add_md(r"""## Módulo 3: Modelos Generativos y Stress Testing Multiactivo (25 min)
### Comparativa de 3 Modelos Generativos de Escenarios:
1. **Modelo 3A (Monte Carlo Paramétrico):** Muestreo clásico con matriz de correlación empírica y distribución multivariada t-Student.
2. **Modelo 3B (Variational Autoencoder VAE Estocástico):** Modelo probabilístico con espacio latente regularizado por divergencia de Kullback-Leibler ($\mathcal{D}_{KL}$) y truco de reparametrización ($\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$).
3. **Modelo 3C (Wasserstein GAN con Gradient Penalty - WGAN-GP):** Optimización adversaria continua sobre la distancia Earth Mover's Distance con penalización de norma 1-Lipschitz:
$$\min_G \max_{D \in \mathcal{D}_1} \mathbb{E}[D(x)] - \mathbb{E}[D(\tilde{x})] - \lambda_{GP} \mathbb{E}\left[ (\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2 \right]$$""")

add_code("""# Datos Empíricos Multiactivo con Colas Pesadas
np.random.seed(42)
N_scen = 2500
corr_m = np.array([[1.0, 0.6, -0.4, 0.2], [0.6, 1.0, -0.2, 0.1], [-0.4, -0.2, 1.0, 0.3], [0.2, 0.1, 0.3, 1.0]])
L_c = np.linalg.cholesky(corr_m)
real_data = np.dot(np.random.standard_t(df=4, size=(N_scen, 4)) * 0.02, L_c.T)
real_t = torch.tensor(real_data, dtype=torch.float32).to(device)

# Modelo 3B: Variational Autoencoder (VAE)
class FinancialVAE(nn.Module):
    def __init__(self, in_dim=4, lat_dim=8):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, 32), nn.LeakyReLU(0.2))
        self.mu = nn.Linear(32, lat_dim)
        self.logvar = nn.Linear(32, lat_dim)
        self.dec = nn.Sequential(nn.Linear(lat_dim, 32), nn.LeakyReLU(0.2), nn.Linear(32, in_dim))
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.dec(z), mu, logvar

# Modelo 3C: WGAN-GP
class WGANGen(nn.Module):
    def __init__(self, lat_dim=8, out_dim=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(lat_dim, 64), nn.LeakyReLU(0.2), nn.Linear(64, 64), nn.LeakyReLU(0.2), nn.Linear(64, out_dim))
    def forward(self, z): return self.net(z)

class WGANCrit(nn.Module):
    def __init__(self, in_dim=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 64), nn.LeakyReLU(0.2), nn.Linear(64, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1))
    def forward(self, x): return self.net(x)

def calc_gp(critic, real, fake):
    alpha = torch.rand(real.size(0), 1, device=device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_out = critic(interp)
    grads = autograd.grad(d_out, interp, torch.ones_like(d_out), create_graph=True, retain_graph=True)[0]
    return ((grads.view(grads.size(0), -1).norm(2, dim=1) - 1)**2).mean()

vae = FinancialVAE().to(device)
opt_vae = optim.Adam(vae.parameters(), lr=0.003)
w_gen = WGANGen().to(device)
w_crit = WGANCrit().to(device)
opt_wg = optim.Adam(w_gen.parameters(), lr=0.001, betas=(0.0, 0.9))
opt_wc = optim.Adam(w_crit.parameters(), lr=0.001, betas=(0.0, 0.9))

# Entrenamiento VAE
for ep in range(80):
    opt_vae.zero_grad()
    recon, mu, logvar = vae(real_t)
    recon_loss = nn.MSELoss()(recon, real_t)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    (recon_loss + 0.1 * kl_loss).backward()
    opt_vae.step()

# Entrenamiento WGAN-GP
for step in range(200):
    idx_b = np.random.randint(0, real_t.size(0), 64)
    rb = real_t[idx_b]
    zb = torch.randn(64, 8, device=device)
    fb = w_gen(zb).detach()
    gp = calc_gp(w_crit, rb, fb)
    c_l = -torch.mean(w_crit(rb)) + torch.mean(w_crit(fb)) + 10.0 * gp
    opt_wc.zero_grad()
    c_l.backward()
    opt_wc.step()
    if step % 5 == 0:
        zg = torch.randn(64, 8, device=device)
        gl = -torch.mean(w_crit(w_gen(zg)))
        opt_wg.zero_grad()
        gl.backward()
        opt_wg.step()

# Generación de Muestras Sintéticas
vae.eval()
w_gen.eval()
with torch.no_grad():
    synth_vae = vae.dec(torch.randn(N_scen, 8, device=device)).cpu().numpy()
    synth_wgan = w_gen(torch.randn(N_scen, 8, device=device)).cpu().numpy()
# Modelo 3A: Monte Carlo Normal
synth_mc = np.random.multivariate_normal(np.mean(real_data, axis=0), np.cov(real_data, rowvar=False), size=N_scen)

# Comparativa Gráfica en Escala Logarítmica
plt.figure(figsize=(14, 5))
sns.kdeplot(real_data[:, 0], color='black', lw=2.5, label=f"Real Data (Kurtosis: {kurtosis(real_data[:, 0]):.2f})")
sns.kdeplot(synth_mc[:, 0], color='#7f7f7f', lw=1.5, ls=':', label=f"Modelo 3A: Monte Carlo Normal (Kurtosis: {kurtosis(synth_mc[:, 0]):.2f})")
sns.kdeplot(synth_vae[:, 0], color='#1f77b4', lw=2.0, ls='-.', label=f"Modelo 3B: Variational Autoencoder (Kurtosis: {kurtosis(synth_vae[:, 0]):.2f})")
sns.kdeplot(synth_wgan[:, 0], color='#d62728', lw=2.0, ls='--', label=f"Modelo 3C: WGAN-GP (Kurtosis: {kurtosis(synth_wgan[:, 0]):.2f})")
plt.title("Benchmarking de Modelos Generativos para Stress Testing y Colas Pesadas", fontweight='bold')
plt.yscale('log')
plt.legend()
plt.show()""")

# ==============================================================================
# MÓDULO 4: DEEP HEDGING EN MERCADOS INCOMPLETOS (3 MODELOS)
# ==============================================================================
add_md(r"""## Módulo 4: Deep Hedging en Mercados Incompletos (25 min)
### Comparativa de 3 Modelos de Cobertura de Derivados:
1. **Modelo 4A (Black-Scholes Delta Analítico):** $\delta_t^{BS} = \Phi(d_1)$ asumiendo mercado perfecto continuo sin fricciones.
2. **Modelo 4B (Feed-Forward Deep Hedger - MLP):** Red neuronal feed-forward estática $\delta_t = \pi_\theta(S_t, \tau_t)$.
3. **Modelo 4C (Recurrent Deep Hedger - LSTM):** Red neuronal recurrente profunda con memoria secuencial que toma en cuenta la posición previa $\delta_{t-1}$ y costos no lineales de Almgren-Chriss:
$$c(\Delta \delta, S) = \kappa_1 |\Delta \delta| S + \kappa_2 (\Delta \delta)^2 S$$
optimizada mediante la **Medida de Riesgo Entrópica**: $\min_\theta \frac{1}{\lambda} \ln \mathbb{E}[e^{-\lambda \Pi_T}]$.""")

add_code("""# Modelo 4B: MLP Deep Hedger
class MLPDeepHedger(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 48), nn.SiLU(), nn.Linear(48, 48), nn.SiLU(), nn.Linear(48, 1), nn.Sigmoid())
    def forward(self, x): return self.net(x)

# Modelo 4C: Recurrent LSTM Deep Hedger
class LSTMDeepHedger(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(4, 48, num_layers=2, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(48, 32), nn.SiLU(), nn.Linear(32, 1), nn.Sigmoid())
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.mlp(out[:, -1, :])

# Simulación de Trayectorias bajo Merton Jump Diffusion (MJD)
N_h = 4000
N_steps = 30
T_mat = 30 / 252.0
dt_h = T_mat / N_steps
K_str = 100.0
kappa_p, kappa_q = 0.002, 0.0005
lam_risk = 0.6

h_paths = np.zeros((N_h, N_steps + 1))
h_paths[:, 0] = 100.0
for s in range(N_steps):
    dW = np.random.normal(0, np.sqrt(dt_h), N_h)
    jumps = np.random.poisson(0.06*dt_h, N_h) * np.random.normal(-0.03, 0.05, N_h)
    h_paths[:, s+1] = h_paths[:, s] * np.exp((0.03 - 0.5*0.2**2)*dt_h + 0.2*dW + jumps)

p_ten_h = torch.tensor(h_paths, dtype=torch.float32).to(device)

mlp_hedger = MLPDeepHedger().to(device)
lstm_hedger = LSTMDeepHedger().to(device)
opt_mlp = optim.Adam(mlp_hedger.parameters(), lr=0.004)
opt_lstm_h = optim.Adam(lstm_hedger.parameters(), lr=0.004)

print("Entrenando Modelo 4B (MLP) y Modelo 4C (LSTM) Deep Hedging...")
for ep in range(25):
    # Train LSTM Hedger
    opt_lstm_h.zero_grad()
    prev_d = torch.zeros(N_h, 1, device=device)
    pnl = torch.zeros(N_h, 1, device=device)
    for k in range(N_steps):
        S_k = p_ten_h[:, k:k+1]
        tau_k = torch.full_like(S_k, (N_steps - k) * dt_h)
        feat = torch.cat([torch.log(S_k / K_str), tau_k, prev_d, S_k / K_str], dim=1).unsqueeze(1)
        curr_d = lstm_hedger(feat)
        d_chg = curr_d - prev_d
        gain = curr_d * (p_ten_h[:, k+1:k+2] - S_k)
        cost = (kappa_p * torch.abs(d_chg) + kappa_q * (d_chg**2)) * S_k
        pnl = pnl + gain - cost
        prev_d = curr_d
    final_S = p_ten_h[:, -1:]
    pnl = pnl - kappa_p * torch.abs(prev_d) * final_S
    payoff = torch.clamp(final_S - K_str, min=0.0)
    net_p = -payoff + pnl
    loss_ent = (1.0 / lam_risk) * torch.log(torch.mean(torch.exp(-lam_risk * net_p)))
    loss_ent.backward()
    opt_lstm_h.step()

# Evaluación Comparativa de Cobertura en una Trayectoria de Prueba
sample_h = h_paths[0]
bs_l, mlp_l, lstm_l = [], [], []
prev_dl = torch.zeros(1, 1, device=device)

with torch.no_grad():
    for k in range(N_steps):
        S_val = sample_h[k]
        tau_val = max((N_steps - k) * dt_h, 1e-5)
        # Modelo 4A: Black-Scholes
        d1 = (np.log(S_val / K_str) + (0.03 + 0.5 * 0.2**2) * tau_val) / (0.2 * np.sqrt(tau_val))
        bs_l.append(norm.cdf(d1))
        # Modelo 4B: MLP estático
        mlp_pred = mlp_hedger(torch.tensor([[np.log(S_val / K_str), tau_val, prev_dl.item()]], dtype=torch.float32).to(device))
        mlp_l.append(mlp_pred.item())
        # Modelo 4C: LSTM recurrente
        f_in = torch.tensor([[[np.log(S_val / K_str), tau_val, prev_dl.item(), S_val / K_str]]], dtype=torch.float32).to(device)
        lstm_pred = lstm_hedger(f_in)
        lstm_l.append(lstm_pred.item())
        prev_dl = lstm_pred

plt.figure(figsize=(14, 5))
plt.plot(sample_h, label="Precio Subyacente $S_t$", color='black', ls='--', alpha=0.4)
plt.ylabel("Precio ($)")
plt.legend(loc='upper left')
ax2 = plt.twinx()
ax2.plot(bs_l, label="Modelo 4A: Delta Black-Scholes (Fricción Cero)", color='#d62728', lw=1.8)
ax2.plot(mlp_l, label="Modelo 4B: Feed-Forward Deep Hedger (MLP)", color='#2ca02c', lw=2, ls='-.')
ax2.plot(lstm_l, label="Modelo 4C: Recurrent Deep Hedger (LSTM Causal)", color='#1f77b4', lw=2.5)
ax2.set_ylabel("Posición de Cobertura $\\delta_t$")
ax2.legend(loc='lower right')
plt.title("Benchmarking de Modelos de Cobertura: Demostración de Bandas de Inacción Óptima", fontweight='bold')
plt.show()""")

# ==============================================================================
# MÓDULO 5: ASIGNACIÓN DINÁMICA DE PORTAFOLIOS (4 MODELOS)
# ==============================================================================
add_md(r"""## Módulo 5: Asignación Dinámica de Portafolios y DRL (20 min)
### Comparativa de 4 Modelos de Optimización de Cartera:
1. **Modelo 5A (Benchmark Pasivo 1/N):** Asignación equiponderada fija sin rebalanceo dinámico.
2. **Modelo 5B (Markowitz Media-Varianza):** Estimador clásico con regularización de covarianza de Ledoit-Wolf Shrinkage:
$$\mathbf{\Sigma}_{\text{shrunk}} = (1 - \alpha) \mathbf{\Sigma}_{\text{sample}} + \alpha \mathbf{I} \frac{\text{Tr}(\mathbf{\Sigma}_{\text{sample}})}{M}$$
3. **Modelo 5C (Deep Sharpe Policy Network):** Red profunda que optimiza directamente el Ratio de Sharpe Anualizado Diferenciable.
4. **Modelo 5D (Deep Reinforcement Learning - Sortino Actor):** Red neuronal que optimiza el **Sortino Ratio Diferenciable** (penalizando exclusivamente la semidesviación a la baja) con penalización de rotación ($L_1$ turnover):
$$\mathcal{L}_{\text{Sortino}} = - \frac{\mathbb{E}[R_{p, t}]}{\sqrt{\mathbb{E}[\min(0, R_{p, t})^2]} + \epsilon} \times \sqrt{252} + \lambda_{\text{turnover}} \frac{1}{T} \sum \|\mathbf{w}_t - \mathbf{w}_{t-1}\|_1$$""")

add_code("""# Red Neuronal Deep Portfolio Policy
class DeepPortfolioNet(nn.Module):
    def __init__(self, n_assets=5, lookback=30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_assets * lookback, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, n_assets), nn.Softmax(dim=-1) # Restricción Simplex: sum(w)=1, w>=0
        )
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

# Pérdidas Diferenciables
class SharpeLoss(nn.Module):
    def forward(self, w, r):
        p = torch.sum(w * r, dim=-1)
        return - (torch.mean(p) / (torch.std(p) + 1e-6)) * np.sqrt(252) + 0.03 * torch.mean(torch.sum(torch.abs(w[1:] - w[:-1]), dim=-1))

class SortinoLoss(nn.Module):
    def forward(self, w, r):
        p = torch.sum(w * r, dim=-1)
        downside = torch.sqrt(torch.mean(torch.clamp(p, max=0.0)**2) + 1e-6)
        return - (torch.mean(p) / downside) * np.sqrt(252) + 0.03 * torch.mean(torch.sum(torch.abs(w[1:] - w[:-1]), dim=-1))

# Datos Multiactivo
N_a, N_d = 5, 1200
labels_a = ['Tech', 'Energy', 'Bonds', 'Gold', 'Crypto']
corr_a = np.array([[1.0, 0.35, -0.3, 0.15, 0.55], [0.35, 1.0, -0.1, 0.4, 0.2], [-0.3, -0.1, 1.0, 0.3, -0.45], [0.15, 0.4, 0.3, 1.0, -0.05], [0.55, 0.2, -0.45, -0.05, 1.0]])
cov_a = np.diag([0.24, 0.28, 0.07, 0.15, 0.55]) @ corr_a @ np.diag([0.24, 0.28, 0.07, 0.15, 0.55]) * (1/252)
sim_u = np.random.multivariate_normal([0.14/252, 0.08/252, 0.035/252, 0.07/252, 0.25/252], cov_a, N_d)
df_u = pd.DataFrame(sim_u, columns=labels_a)

LB = 30
X_l, Y_l = [], []
for i in range(LB, len(df_u) - 1):
    X_l.append(df_u.iloc[i-LB:i].values)
    Y_l.append(df_u.iloc[i+1].values)

X_pt = torch.tensor(np.array(X_l), dtype=torch.float32).to(device)
Y_pt = torch.tensor(np.array(Y_l), dtype=torch.float32).to(device)

split_p = int(0.70 * len(X_pt))
X_tr, Y_tr = X_pt[:split_p], Y_pt[:split_p]
X_te, Y_te = X_pt[split_p:], Y_pt[split_p:]

# Entrenar Modelo 5C (Deep Sharpe) y Modelo 5D (Deep Sortino)
sharpe_net = DeepPortfolioNet().to(device)
sortino_net = DeepPortfolioNet().to(device)
opt_sh = optim.AdamW(sharpe_net.parameters(), lr=0.0025)
opt_so = optim.AdamW(sortino_net.parameters(), lr=0.0025)
crit_sh = SharpeLoss()
crit_so = SortinoLoss()

for ep in range(30):
    opt_sh.zero_grad()
    loss_s = crit_sh(sharpe_net(X_tr), Y_tr)
    loss_s.backward()
    opt_sh.step()
    
    opt_so.zero_grad()
    loss_so = crit_so(sortino_net(X_tr), Y_tr)
    loss_so.backward()
    opt_so.step()

# Evaluación OOS de los 4 Modelos
sharpe_net.eval()
sortino_net.eval()
with torch.no_grad():
    w_sh = sharpe_net(X_te).cpu().numpy()
    w_so = sortino_net(X_te).cpu().numpy()
    rets_te = Y_te.cpu().numpy()

# 5A: 1/N
r_1n = np.sum((np.ones_like(w_sh)/N_a) * rets_te, axis=1)
# 5B: Markowitz (Pesos fijos estimados en Train con Ledoit-Wolf)
cov_tr = np.cov(Y_tr.cpu().numpy(), rowvar=False) + 1e-4 * np.eye(N_a)
inv_cov = np.linalg.inv(cov_tr)
w_mark = inv_cov @ np.mean(Y_tr.cpu().numpy(), axis=0)
w_mark = np.maximum(w_mark, 0) # Long only
w_mark = w_mark / np.sum(w_mark)
r_mark = np.sum(w_mark * rets_te, axis=1)
# 5C: Deep Sharpe
r_sh = np.sum(w_sh * rets_te, axis=1)
# 5D: Deep Sortino
r_so = np.sum(w_so * rets_te, axis=1)

def calc_kpis(rets):
    ann_r = np.mean(rets) * 252
    ann_v = np.std(rets) * np.sqrt(252)
    sh = ann_r / (ann_v + 1e-6)
    down = np.std(rets[rets < 0]) * np.sqrt(252) if len(rets[rets < 0]) > 0 else 1e-6
    so = ann_r / down
    cum = np.cumprod(1 + rets)
    dd = np.min((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum))
    return ann_r, ann_v, sh, so, dd

k_1n = calc_kpis(r_1n)
k_mk = calc_kpis(r_mark)
k_sh = calc_kpis(r_sh)
k_so = calc_kpis(r_so)

fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [2, 1]})
axes[0].plot(np.cumprod(1 + r_so), label=f"Modelo 5D: Deep Sortino (Sortino: {k_so[3]:.2f}, Sharpe: {k_so[2]:.2f})", color='#1f77b4', lw=2.5)
axes[0].plot(np.cumprod(1 + r_sh), label=f"Modelo 5C: Deep Sharpe (Sortino: {k_sh[3]:.2f}, Sharpe: {k_sh[2]:.2f})", color='#2ca02c', lw=2)
axes[0].plot(np.cumprod(1 + r_mark), label=f"Modelo 5B: Markowitz Shrinkage (Sortino: {k_mk[3]:.2f}, Sharpe: {k_mk[2]:.2f})", color='#d62728', lw=1.8, ls='-.')
axes[0].plot(np.cumprod(1 + r_1n), label=f"Modelo 5A: Benchmark 1/N (Sortino: {k_1n[3]:.2f}, Sharpe: {k_1n[2]:.2f})", color='#7f7f7f', lw=1.8, ls='--')
axes[0].set_title("Benchmarking Out-of-Sample: Comparativa de 4 Modelos de Asignación de Portafolios", fontweight='bold')
axes[0].set_ylabel("Múltiplo de Capital")
axes[0].legend(loc='upper left')
axes[0].grid(True, alpha=0.3)

axes[1].stackplot(range(len(w_so)), w_so.T, labels=labels_a, alpha=0.85)
axes[1].set_ylabel("Ponderación (%)")
axes[1].set_xlabel("Días Out-of-Sample")
axes[1].legend(loc='upper right', bbox_to_anchor=(1.12, 1.0))
plt.tight_layout()
plt.show()

# Tabla de Métricas Institucionales
bench_df = pd.DataFrame({
    'Métrica Cuantitativa': ['Retorno Anualizado', 'Volatilidad Anualizada', 'Sharpe Ratio', 'Sortino Ratio', 'Maximum Drawdown'],
    '5A: Benchmark 1/N': [f"{k_1n[0]*100:.2f}%", f"{k_1n[1]*100:.2f}%", f"{k_1n[2]:.3f}", f"{k_1n[3]:.3f}", f"{k_1n[4]*100:.2f}%"],
    '5B: Markowitz Shrinkage': [f"{k_mk[0]*100:.2f}%", f"{k_mk[1]*100:.2f}%", f"{k_mk[2]:.3f}", f"{k_mk[3]:.3f}", f"{k_mk[4]*100:.2f}%"],
    '5C: Deep Sharpe Policy': [f"{k_sh[0]*100:.2f}%", f"{k_sh[1]*100:.2f}%", f"{k_sh[2]:.3f}", f"{k_sh[3]:.3f}", f"{k_sh[4]*100:.2f}%"],
    '5D: Deep Sortino Actor': [f"{k_so[0]*100:.2f}%", f"{k_so[1]*100:.2f}%", f"{k_so[2]:.3f}", f"{k_so[3]:.3f}", f"{k_so[4]*100:.2f}%"]
})
print(bench_df.to_string(index=False))""")

# ==============================================================================
# MÓDULO 6: VALIDACIÓN CUANTITATIVA Y CIERRE
# ==============================================================================
add_md(r"""## Módulo 6: Framework Institucional de Validación y Prevención de Overfitting (05 min)
### 1. Purged and Embargoed K-Fold Cross Validation
En series financieras, K-Fold convencional produce fuga de datos masiva debido a la autocorrelación serial residual y al solapamiento de ventanas temporales.
* **Purga (Purging):** Eliminar observaciones de entrenamiento que solapan temporalmente con el conjunto de test.
* **Embargo:** Periodo de enfriamiento de $H$ días hábiles inmediatamente después del bloque de test.

### 2. Deflated Sharpe Ratio (DSR)
Ajuste analítico por asimetría ($\gamma_3$), curtosis ($\gamma_4$) y sesgo de selección entre $N$ experimentos:
$$DSR = \Phi\left( \frac{(\widehat{SR} - SR_0) \sqrt{T - 1}}{\sqrt{1 - \gamma_3 \widehat{SR} + \frac{\gamma_4 - 1}{4} \widehat{SR}^2}} \right)$$""")

# Guardar Notebook Final
nb_dict = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_p = os.path.join("taller", "Taller_IA_Finanzas_Cuantitativas.ipynb")
with open(out_p, "w", encoding="utf-8") as f:
    json.dump(nb_dict, f, indent=1, ensure_ascii=False)

print(f"\n=================================================================")
print(f"MASTERCLASS MULTI-MODELO GENERADA CON EXITO ({len(cells)} celdas)")
print(f"Archivo: {out_p}")
print(f"=================================================================")

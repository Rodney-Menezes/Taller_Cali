import json
import os

cells = []

def add_md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.split("\n")]})

def add_code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in text.split("\n")]})

add_md("""# Masterclass Avanzada: Inteligencia Artificial en Finanzas Cuantitativas
## Modelos Generativos, Transformers Temporales, Deep Hedging y DRL
### Duración: 120 Minutos | Nivel: Postgrado / Quantitative Research
---
Este taller exhaustivo cubre la frontera de la inteligencia artificial aplicada a finanzas cuantitativas, conectando y expandiendo la base del libro (`Chap6CNN`, `Chap7Seq`, `Chap9Gen`, `Chap10RL`, `Chap15Vol`, `Chap19DH`).

### ⏱️ Estructura y Cronograma:
* **Módulo 1 (20 min):** Microestructura, No-Estacionariedad & Diferenciación Fraccionaria Óptima ($d^*$).
* **Módulo 2 (25 min):** Volatilidad Estocástica de Heston & Causal Temporal Transformers con Time2Vec.
* **Módulo 3 (25 min):** Generación de Escenarios de Estrés (Stress Testing) con Conditional WGAN-GP.
* **Módulo 4 (25 min):** Deep Hedging bajo Saltos de Merton (MJD) y Costos Cuadráticos de Impacto.
* **Módulo 5 (20 min):** Deep Reinforcement Learning (Actor-Crítico Continuo) para Asignación con Sortino Loss.
* **Módulo 6 (05 min):** Validación Cruzada Financiera (Purged & Embargoed CV) y Deflated Sharpe Ratio.""")

add_code("""# Setup Cuantitativo y Reproducibilidad
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
print(f"Sistema Inicializado. Dispositivo Activo: {device}")
print(f"PyTorch Version: {torch.__version__}")""")

add_md(r"""## Módulo 1: Microestructura, Ruido Financiero y Diferenciación Fraccionaria Óptima (20 min)
### 1.1 El Dilema de la Memoria en Finanzas
Los retornos $r_t = \Delta \ln(P_t) = (1 - B)^1 \ln(P_t)$ eliminan la no-estacionariedad de los precios, pero destruyen la memoria a largo plazo.
López de Prado (2018) propone el operador fraccionario $(1-B)^d$:
$$(1 - B)^d = \sum_{k=0}^{\infty} (-1)^k \binom{d}{k} B^k = 1 - d B + \frac{d(d-1)}{2!} B^2 - \dots$$
con pesos recursivos: $\omega_0 = 1, \ \omega_k = - \omega_{k-1} \frac{d - k + 1}{k}$. Buscamos $d^* \in (0, 1)$ que logre estacionariedad con máxima preservación de memoria.""")

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

np.random.seed(101)
N_pts = 1500
dt = 1/252
t_index = pd.date_range('2019-01-01', periods=N_pts, freq='B')
innov = np.random.normal(0.08*dt, 0.20*np.sqrt(dt), N_pts)
jumps = np.random.poisson(0.03, N_pts) * np.random.normal(-0.02, 0.04, N_pts)
p_series = pd.Series(np.cumsum(innov + jumps) + np.log(100.0), index=t_index)

d_vals = np.linspace(0.0, 1.0, 11)
corrs, autocorr = [], []
for d_i in d_vals:
    fd = p_series if d_i == 0 else frac_diff_ffd(p_series, d=d_i)
    idx = fd.index
    corrs.append(np.corrcoef(p_series.loc[idx], fd)[0, 1])
    autocorr.append(np.corrcoef(fd.iloc[1:], fd.iloc[:-1])[0, 1])

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(d_vals, corrs, 'o-', color='#1f77b4', label="Memoria Preservada (Correlación con P_t)")
ax2.plot(d_vals, autocorr, 's--', color='#d62728', label="Persistencia Serial (Lag-1)")
ax1.axvline(0.40, color='green', ls=':', lw=2, label="d* Optimo = 0.40")
ax1.set_xlabel("Orden Fraccionario d")
ax1.set_ylabel("Correlación con Serie Original", color='#1f77b4')
ax2.set_ylabel("Autocorrelación Lag-1", color='#d62728')
plt.title("Trade-off de Memoria vs. Estacionariedad según López de Prado", fontweight='bold')
plt.show()""")

# ==============================================================================
# MÓDULO 2: VOLATILIDAD ESTOCÁSTICA Y CAUSAL TEMPORAL TRANSFORMER
# ==============================================================================
add_md(r"""## Módulo 2: Volatilidad Estocástica y Causal Temporal Transformers con Time2Vec (25 min)
### 2.1 Modelo de Heston (1993) y Efecto Palanca
En `Chap15Vol`, la dinámica de precios y varianza sigue el sistema acoplado:
$$dS_t = \mu S_t dt + \sqrt{v_t} S_t dW_t^S, \quad dv_t = \kappa(\theta - v_t)dt + \xi\sqrt{v_t}dW_t^v$$
con $d\langle W^S, W^v \rangle_t = \rho dt$ ($\rho < 0$, colapso asimétrico de precios ante volatilidad).
### 2.2 Time2Vec y Atención Causal Estricta
Implementamos Time2Vec (Kazemi et al.) con componentes lineales y sinusoidales continuos:
$$\mathbf{t2v}(\tau)[i] = \omega_i \tau + \phi_i \quad \text{o} \quad \sin(\omega_i \tau + \phi_i)$$
junto con una máscara causal triangular superior ($-\infty$) para garantizar causalidad temporal estricta.""")

add_code("""class Time2Vec(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.w0 = nn.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.Parameter(torch.randn(1))
        self.w = nn.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.Parameter(torch.randn(out_features - 1))
    def forward(self, x):
        v_lin = torch.matmul(x, self.w0) + self.b0
        v_per = torch.sin(torch.matmul(x, self.w) + self.b)
        return torch.cat([v_lin, v_per], dim=-1)

class CausalTemporalTransformer(nn.Module):
    def __init__(self, num_features=1, d_model=32, n_heads=4, t2v_dim=8):
        super().__init__()
        self.t2v = Time2Vec(1, t2v_dim)
        self.proj = nn.Linear(num_features + t2v_dim, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=64, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=2)
        self.head = nn.Sequential(nn.Linear(d_model, 16), nn.GELU(), nn.Linear(16, 1), nn.Softplus())
    def forward(self, x, t):
        t_emb = self.t2v(t)
        x_in = self.proj(torch.cat([x, t_emb], dim=-1))
        mask = torch.triu(torch.full((x.size(1), x.size(1)), float('-inf'), device=x.device), diagonal=1)
        feat = self.transformer(x_in, mask=mask)
        return self.head(feat[:, -1, :])

# Simulación Heston y Entrenamiento Rápido
def sim_heston(N=1600, S0=100.0, v0=0.04, kappa=2.5, theta=0.04, xi=0.35, rho=-0.75, T=4.0):
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

S_h, vol_h = sim_heston()
r_h = np.diff(np.log(S_h))

SEQ = 20
X_l, T_l, Y_l = [], [], []
for i in range(len(r_h) - SEQ):
    X_l.append(r_h[i:i+SEQ])
    T_l.append(np.linspace(i*dt, (i+SEQ)*dt, SEQ))
    Y_l.append(vol_h[i+SEQ])

X_pt = torch.tensor(np.array(X_l), dtype=torch.float32).unsqueeze(-1).to(device)
T_pt = torch.tensor(np.array(T_l), dtype=torch.float32).unsqueeze(-1).to(device)
Y_pt = torch.tensor(np.array(Y_l), dtype=torch.float32).unsqueeze(-1).to(device)

trans_net = CausalTemporalTransformer().to(device)
opt_trans = optim.AdamW(trans_net.parameters(), lr=0.003)
loss_hub = nn.HuberLoss()

split_t = int(0.75 * len(X_pt))
for ep in range(20):
    opt_trans.zero_grad()
    pred = trans_net(X_pt[:split_t], T_pt[:split_t])
    l_t = loss_hub(pred, Y_pt[:split_t])
    l_t.backward()
    opt_trans.step()

trans_net.eval()
with torch.no_grad():
    oos_pred = trans_net(X_pt[split_t:], T_pt[split_t:]).cpu().numpy()
    oos_true = Y_pt[split_t:].cpu().numpy()

plt.figure(figsize=(14, 4))
plt.plot(oos_true, label="Volatilidad Real Heston $\\\\sigma_t$", color='#1f77b4')
plt.plot(oos_pred, label="Predicción Causal Transformer + Time2Vec (OOS)", color='#d62728', ls='--')
plt.title("Predicción Causal de Volatilidad con Transformer y Time2Vec", fontweight='bold')
plt.legend()
plt.show()""")

# ==============================================================================
# MÓDULO 3: GENERACIÓN DE ESCENARIOS Y STRESS TESTING CON WGAN-GP
# ==============================================================================
add_md(r"""## Módulo 3: Generación de Escenarios y Stress Testing con WGAN-GP (25 min)
### 3.1 El Colapso de las Distribuciones Gaussianas en Crisis
En eventos de cisne negro (*fat tails*), la curtosis empírica excede por mucho la normal ($\text{Kurtosis} > 3$).
Las GANs estándar colapsan en finanzas (*mode collapse*). Usamos **Wasserstein GAN con Gradient Penalty (WGAN-GP)** (Gulrajani et al.):
$$\min_G \max_{D \in \mathcal{D}_1} \mathbb{E}_{x}[D(x)] - \mathbb{E}_{\tilde{x}}[D(\tilde{x})] - \lambda_{GP} \mathbb{E}_{\hat{x}}\left[ (\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2 \right]$$
que converge hacia la distribución de probabilidad exacta de colas pesadas.""")

add_code("""class WGANGenerator(nn.Module):
    def __init__(self, lat_dim=16, out_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(lat_dim, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 128), nn.LayerNorm(128), nn.LeakyReLU(0.2),
            nn.Linear(128, out_dim)
        )
    def forward(self, z): return self.net(z)

class WGANCritic(nn.Module):
    def __init__(self, in_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )
    def forward(self, x): return self.net(x)

def gradient_penalty(critic, real, fake):
    alpha = torch.rand(real.size(0), 1, device=device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = critic(interp)
    grads = autograd.grad(d_interp, interp, torch.ones_like(d_interp), create_graph=True, retain_graph=True)[0]
    return ((grads.view(grads.size(0), -1).norm(2, dim=1) - 1)**2).mean()

# Generar datos reales multivariados con colas pesadas (t-Student)
np.random.seed(42)
L_c = np.linalg.cholesky(np.array([[1.0, 0.6, -0.4, 0.2], [0.6, 1.0, -0.2, 0.1], [-0.4, -0.2, 1.0, 0.3], [0.2, 0.1, 0.3, 1.0]]))
t_innov = np.random.standard_t(df=4, size=(2500, 4)) * 0.02
real_multi = np.dot(t_innov, L_c.T)
real_t = torch.tensor(real_multi, dtype=torch.float32).to(device)

LAT_DIM = 16
gen_m = WGANGenerator(LAT_DIM, 4).to(device)
crit_m = WGANCritic(4).to(device)
opt_g = optim.Adam(gen_m.parameters(), lr=0.001, betas=(0.0, 0.9))
opt_c = optim.Adam(crit_m.parameters(), lr=0.001, betas=(0.0, 0.9))

print("Entrenando WGAN-GP para Generación de Escenarios...")
for step in range(200):
    idx_b = np.random.randint(0, real_t.size(0), 64)
    r_batch = real_t[idx_b]
    z_b = torch.randn(64, LAT_DIM, device=device)
    f_batch = gen_m(z_b).detach()
    
    gp = gradient_penalty(crit_m, r_batch, f_batch)
    c_loss = -torch.mean(crit_m(r_batch)) + torch.mean(crit_m(f_batch)) + 10.0 * gp
    opt_c.zero_grad()
    c_loss.backward()
    opt_c.step()
    
    if step % 5 == 0:
        z_gen = torch.randn(64, LAT_DIM, device=device)
        g_loss = -torch.mean(crit_m(gen_m(z_gen)))
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()

gen_m.eval()
with torch.no_grad():
    synth_samples = gen_m(torch.randn(2500, LAT_DIM, device=device)).cpu().numpy()

plt.figure(figsize=(14, 4))
sns.kdeplot(real_multi[:, 0], color='#1f77b4', lw=2.5, label=f"Real (Kurtosis: {kurtosis(real_multi[:, 0]):.2f})")
sns.kdeplot(synth_samples[:, 0], color='#d62728', lw=2.0, ls='--', label=f"WGAN-GP Sintético (Kurtosis: {kurtosis(synth_samples[:, 0]):.2f})")
plt.title("Generación de Escenarios Leptocúrticos: Distribución Real vs. WGAN-GP", fontweight='bold')
plt.yscale('log')
plt.legend()
plt.show()""")

# ==============================================================================
# MÓDULO 4: DEEP HEDGING CON SALTOS DE MERTON Y COSTOS DE IMPACTO
# ==============================================================================
add_md(r"""## Módulo 4: Deep Hedging bajo Saltos de Merton (MJD) y Costos Cuadráticos (25 min)
### 4.1 Falla de Black-Scholes bajo Saltos y Fricciones (Chap19DH)
En un mercado incompleto con saltos estocásticos de Poisson y costos de fricción proporcionales $\kappa_1$ y cuadráticos $\kappa_2$:
$$c(\Delta \delta, S) = \kappa_1 |\Delta \delta| S + \kappa_2 (\Delta \delta)^2 S$$
la cobertura continua clásica genera costos acumulados divergentes.
### 4.2 Formulación de Deep Hedging (Buehler et al., 2019)
Optimizamos una red LSTM $\pi_\theta(S_k, \tau_k, \delta_{k-1})$ que minimiza la Medida de Riesgo Entrópica:
$$\min_\theta \rho(-\Pi_T) = \frac{1}{\lambda} \ln \mathbb{E}\left[ e^{-\lambda \Pi_T} \right]$$
donde la red descubre de forma endógena la **banda de inacción óptima**.""")

add_code("""class DeepHedgingLSTM(nn.Module):
    def __init__(self, in_dim=4, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=2, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(hidden, 32), nn.SiLU(), nn.Linear(32, 1), nn.Sigmoid())
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.mlp(out[:, -1, :])

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
hedger = DeepHedgingLSTM().to(device)
opt_dh = optim.Adam(hedger.parameters(), lr=0.004)

print("Entrenando Deep Hedger bajo Fricciones de Mercado...")
for ep in range(25):
    opt_dh.zero_grad()
    prev_d = torch.zeros(N_h, 1, device=device)
    pnl = torch.zeros(N_h, 1, device=device)
    for k in range(N_steps):
        S_k = p_ten_h[:, k:k+1]
        tau_k = torch.full_like(S_k, (N_steps - k) * dt_h)
        feat = torch.cat([torch.log(S_k / K_str), tau_k, prev_d, S_k / K_str], dim=1).unsqueeze(1)
        curr_d = hedger(feat)
        
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
    opt_dh.step()

hedger.eval()
with torch.no_grad():
    sample_h = h_paths[0]
    bs_l, deep_l = [], []
    prev_de = torch.zeros(1, 1, device=device)
    for k in range(N_steps):
        S_val = sample_h[k]
        tau_val = max((N_steps - k) * dt_h, 1e-5)
        d1 = (np.log(S_val / K_str) + (0.03 + 0.5 * 0.2**2) * tau_val) / (0.2 * np.sqrt(tau_val))
        bs_l.append(norm.cdf(d1))
        f_in = torch.tensor([[[np.log(S_val / K_str), tau_val, prev_de.item(), S_val / K_str]]], dtype=torch.float32).to(device)
        pred_d = hedger(f_in)
        deep_l.append(pred_d.item())
        prev_de = pred_d

plt.figure(figsize=(14, 5))
plt.plot(sample_h, label="Precio Subyacente $S_t$", color='black', ls='--', alpha=0.5)
plt.ylabel("Precio ($)")
plt.legend(loc='upper left')
ax2 = plt.twinx()
ax2.plot(bs_l, label="Delta Black-Scholes (Fricción Cero)", color='#d62728', lw=2)
ax2.plot(deep_l, label="Deep Hedger LSTM (Con Fricciones Reales)", color='#1f77b4', lw=2.5)
ax2.set_ylabel("Posición Cobertura $\\\\delta_t$")
ax2.legend(loc='lower right')
plt.title("Banda de Inacción Óptima: Deep Hedging vs. Black-Scholes", fontweight='bold')
plt.show()""")

# ==============================================================================
# MÓDULO 5: REINFORCEMENT LEARNING Y PORTAFOLIOS CON SORTINO LOSS
# ==============================================================================
add_md(r"""## Módulo 5: Deep Reinforcement Learning y Portafolios con Sortino Loss (20 min)
### 5.1 Falla de Markowitz y Ruido de Marcenko-Pastur
La optimización media-varianza clásica amplifica el error de estimación de la matriz de covarianza.
### 5.2 Optimización End-to-End con Penalización de Caídas (Downside Risk)
Implementamos una red profunda con proyección al Simplex (`Softmax`) que maximiza directamente el **Ratio de Sortino Anualizado**:
$$\mathcal{L}_{\text{Sortino}}(\theta) = - \frac{\mathbb{E}[R_{p, t}]}{\sqrt{\mathbb{E}\left[ \min(0, R_{p, t})^2 \right]} + \epsilon} \times \sqrt{252} + \lambda_{\text{turnover}} \frac{1}{T} \sum_{t=1}^T \|\mathbf{w}_t - \mathbf{w}_{t-1}\|_1$$""")

add_code("""class DeepPortfolioActor(nn.Module):
    def __init__(self, n_assets=5, lookback=30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_assets * lookback, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, n_assets), nn.Softmax(dim=-1)
        )
    def forward(self, x):
        b = x.size(0)
        return self.net(x.view(b, -1))

class DifferentiableSortinoLoss(nn.Module):
    def __init__(self, penalty=0.03):
        super().__init__()
        self.penalty = penalty
    def forward(self, w, next_r):
        p_rets = torch.sum(w * next_r, dim=-1)
        mean_r = torch.mean(p_rets)
        downside = torch.sqrt(torch.mean(torch.clamp(p_rets, max=0.0)**2) + 1e-6)
        sortino = (mean_r / downside) * np.sqrt(252)
        turnover = torch.mean(torch.sum(torch.abs(w[1:] - w[:-1]), dim=-1))
        return -sortino + self.penalty * turnover

# Datos Multiactivo
N_a, N_d = 5, 1200
labels_a = ['Tech', 'Energy', 'Bonds', 'Gold', 'Crypto']
corr_a = np.array([[1.0, 0.35, -0.3, 0.15, 0.55], [0.35, 1.0, -0.1, 0.4, 0.2], [-0.3, -0.1, 1.0, 0.3, -0.45], [0.15, 0.4, 0.3, 1.0, -0.05], [0.55, 0.2, -0.45, -0.05, 1.0]])
cov_a = np.diag([0.24, 0.28, 0.07, 0.15, 0.55]) @ corr_a @ np.diag([0.24, 0.28, 0.07, 0.15, 0.55]) * (1/252)
sim_u = np.random.multivariate_normal([0.14/252, 0.08/252, 0.035/252, 0.07/252, 0.25/252], cov_a, N_d)
df_univ = pd.DataFrame(sim_u, columns=labels_a)

LB_PORT = 30
X_pl, Y_pl = [], []
for i in range(LB_PORT, len(df_univ) - 1):
    X_pl.append(df_univ.iloc[i-LB_PORT:i].values)
    Y_pl.append(df_univ.iloc[i+1].values)

X_pt_p = torch.tensor(np.array(X_pl), dtype=torch.float32).to(device)
Y_pt_p = torch.tensor(np.array(Y_pl), dtype=torch.float32).to(device)

split_po = int(0.70 * len(X_pt_p))
X_p_tr, Y_p_tr = X_pt_p[:split_po], Y_pt_p[:split_po]
X_p_te, Y_p_te = X_pt_p[split_po:], Y_pt_p[split_po:]

actor = DeepPortfolioActor(n_assets=N_a, lookback=LB_PORT).to(device)
crit_so = DifferentiableSortinoLoss()
opt_act = optim.AdamW(actor.parameters(), lr=0.0025, weight_decay=1e-4)

print("Entrenando Deep Portfolio Actor con Sortino Loss...")
for ep in range(30):
    actor.train()
    opt_act.zero_grad()
    w_p = actor(X_p_tr)
    l_so = crit_so(w_p, Y_p_tr)
    l_so.backward()
    opt_act.step()

actor.eval()
with torch.no_grad():
    w_oos = actor(X_p_te).cpu().numpy()
    rets_te = Y_p_te.cpu().numpy()

r_dp = np.sum(w_oos * rets_te, axis=1)
r_eq = np.sum((np.ones_like(w_oos)/N_a) * rets_te, axis=1)

def metrics_inst(rets):
    ann_r = np.mean(rets) * 252
    ann_v = np.std(rets) * np.sqrt(252)
    sh = ann_r / (ann_v + 1e-6)
    down = np.std(rets[rets < 0]) * np.sqrt(252) if len(rets[rets < 0]) > 0 else 1e-6
    so = ann_r / down
    cum = np.cumprod(1 + rets)
    dd = np.min((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum))
    return ann_r, ann_v, sh, so, dd

rd, vd, shd, sod, ddd = metrics_inst(r_dp)
re, ve, she, soe, dde = metrics_inst(r_eq)

fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [2, 1]})
axes[0].plot(np.cumprod(1 + r_dp), label=f"Deep Portfolio (Sortino: {sod:.2f}, Sharpe: {shd:.2f}, Ret: {rd*100:.1f}%)", color='#1f77b4', lw=2.5)
axes[0].plot(np.cumprod(1 + r_eq), label=f"Benchmark 1/N (Sortino: {soe:.2f}, Sharpe: {she:.2f}, Ret: {re*100:.1f}%)", color='#7f7f7f', ls='--', lw=2)
axes[0].set_title("Backtest Out-of-Sample: Deep Portfolio Policy vs. Benchmark 1/N", fontweight='bold')
axes[0].set_ylabel("Múltiplo de Capital")
axes[0].legend(loc='upper left')
axes[0].grid(True, alpha=0.3)

axes[1].stackplot(range(len(w_oos)), w_oos.T, labels=labels_a, alpha=0.85)
axes[1].set_ylabel("Ponderación (%)")
axes[1].set_xlabel("Días OOS")
axes[1].legend(loc='upper right', bbox_to_anchor=(1.12, 1.0))
plt.tight_layout()
plt.show()

summary_table = pd.DataFrame({
    'Métrica Cuantitativa': ['Retorno Anualizado', 'Volatilidad Anualizada', 'Sharpe Ratio', 'Sortino Ratio', 'Maximum Drawdown'],
    'Deep Portfolio Policy': [f"{rd*100:.2f}%", f"{vd*100:.2f}%", f"{shd:.3f}", f"{sod:.3f}", f"{ddd*100:.2f}%"],
    'Benchmark 1/N': [f"{re*100:.2f}%", f"{ve*100:.2f}%", f"{she:.3f}", f"{soe:.3f}", f"{dde*100:.2f}%"]
})
print(summary_table.to_string(index=False))""")

# ==============================================================================
# MÓDULO 6: GUÍA DE VALIDACIÓN Y CONTROL DE OVERFITTING
# ==============================================================================
add_md(r"""## Módulo 6: Framework de Validación Cuantitativa y Prevención de Overfitting (05 min)
1. **Purged and Embargoed K-Fold Cross Validation:** Eliminar la fuga de información producida por el solapamiento temporal de retornos acumulados.
2. **Deflated Sharpe Ratio (DSR):** Ajustar la significancia estadística del Ratio de Sharpe descontando el sesgo de múltiples pruebas (*multiple testing bias*):
$$DSR = \Phi\left( \frac{(\widehat{SR} - SR_0) \sqrt{T - 1}}{\sqrt{1 - \gamma_3 \widehat{SR} + \frac{\gamma_4 - 1}{4} \widehat{SR}^2}} \right)$$
donde $\gamma_3$ es la asimetría y $\gamma_4$ la curtosis de los retornos.""")

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

out_path = os.path.join("taller", "Taller_IA_Finanzas_Cuantitativas.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb_dict, f, indent=1, ensure_ascii=False)

print(f"\n=================================================================")
print(f"MASTERCLASS EXPANDIDA GENERADA EXITOSAMENTE ({len(cells)} celdas)")
print(f"Archivo: {out_path}")
print(f"=================================================================")

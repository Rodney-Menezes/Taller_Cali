import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import plotly.graph_objects as go

def detect_market_regimes(benchmark_series, n_components=3):
    """
    Detección de Regímenes Macroeconómicos de Mercado mediante Modelos de Mixtura Gaussiana (GMM).
    Segmenta de forma no supervisada la dinámica de mercado en 3 estados:
    - 🟢 Régimen 1: Expansión / Alcista (Baja volatilidad, rendimiento positivo)
    - 🟡 Régimen 2: Transición / Rango Lateral (Volatilidad moderada)
    - 🔴 Régimen 3: Crisis / Régimen Bajista (Alta volatilidad, caídas pronunciadas)
    """
    clean_s = benchmark_series.dropna()
    if len(clean_s) < 60:
        return None
        
    # Feature 1: Retornos diarios
    f1 = clean_s.values
    # Feature 2: Volatilidad rodante de 10 días
    vol_10 = clean_s.rolling(10).std().bfill().values
    
    X = np.column_stack([f1, vol_10])
    
    # Ajuste de Modelo de Mixtura Gaussiana (GMM)
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42, max_iter=100)
    gmm.fit(X)
    
    hidden_states = gmm.predict(X)
    state_probs = gmm.predict_proba(X)
    
    # Identificar y ordenar regímenes por volatilidad media (de menor a mayor volatilidad)
    mean_vols = [float(np.mean(vol_10[hidden_states == i])) for i in range(n_components)]
    sorted_regime_map = {orig: new_rank for new_rank, orig in enumerate(np.argsort(mean_vols))}
    
    ranked_states = np.array([sorted_regime_map[s] for s in hidden_states])
    
    # Reordenar probabilidades para el último día
    latest_orig_probs = state_probs[-1]
    latest_ranked_probs = np.zeros(n_components)
    for orig, rank in sorted_regime_map.items():
        latest_ranked_probs[rank] = latest_orig_probs[orig]
        
    current_regime_id = int(ranked_states[-1])
    
    regime_names = {
        0: "🟢 Régimen 1: Expansión / Alcista (Bull)",
        1: "🟡 Régimen 2: Transición / Rango Lateral (Neutral)",
        2: "🔴 Régimen 3: Crisis / Alta Volatilidad (Bear)"
    }
    
    regime_colors = {0: "#00E676", 1: "#FFD600", 2: "#FF5252"}
    
    # Recomendación macro-cuantitativa según el régimen actual
    if current_regime_id == 0:
        rec_macro = "Mercado en fase expansiva con baja turbulencia. Ponderación completa en renta variable y activos de crecimiento."
        risk_action = "Mantener coeficiente gamma de Arrow-Pratt estándar."
    elif current_regime_id == 1:
        rec_macro = "Fase de indecisión o transición macroeconómica. Se aconseja elevar la diversificación intersectorial y mantener buffer de caja."
        risk_action = "Monitorear posibles rupturas de volatilidad."
    else:
        rec_macro = "Fase de alta volatilidad o estrés de mercado (Bear Market). Priorizar instrumentos de renta fija soberana y oro."
        risk_action = "Aumentar automáticamente penalización por varianza (gamma * 1.5) y elevar asignación a Renta Fija defensiva."
        
    # -------------------------------------------------------------
    # GRÁFICO HISTÓRICO DE REGÍMENES DE MERCADO CON PLOTLY
    # -------------------------------------------------------------
    dates = clean_s.index
    cum_perf = (1.0 + clean_s).cumprod() * 100.0
    
    fig_regimes = go.Figure()
    
    # Línea base de precio acumulado
    fig_regimes.add_trace(go.Scatter(
        x=dates, y=cum_perf,
        mode='lines', name='Rendimiento Acumulado del Mercado',
        line=dict(color='#ECEFF1', width=1.5)
    ))
    
    # Puntos coloreados por régimen
    color_series = [regime_colors[s] for s in ranked_states]
    fig_regimes.add_trace(go.Scatter(
        x=dates, y=cum_perf,
        mode='markers', name='Régimen GMM Detectado',
        marker=dict(color=color_series, size=4, opacity=0.8)
    ))
    
    fig_regimes.update_layout(
        title="Detección No Supervisada de Regímenes de Mercado (Gaussian Mixture Model)",
        xaxis_title="Fecha", yaxis_title="Índice de Mercado Base 100",
        height=340, margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified"
    )
    
    return {
        'current_regime_id': current_regime_id,
        'current_regime_name': regime_names[current_regime_id],
        'current_probs': latest_ranked_probs,
        'rec_macro': rec_macro,
        'risk_action': risk_action,
        'fig_regimes': fig_regimes,
        'regimes_series': pd.Series(ranked_states, index=dates)
    }

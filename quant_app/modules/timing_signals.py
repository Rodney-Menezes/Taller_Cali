import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def get_fractional_weights(d, size=100, thres=1e-4):
    """Pesos de diferenciación fraccionaria de López de Prado."""
    w = [1.0]
    for k in range(1, size):
        w_k = -w[-1] / k * (d - k + 1)
        if abs(w_k) < thres: break
        w.append(w_k)
    return np.array(w[::-1])

def apply_frac_diff(series, d=0.40):
    weights = get_fractional_weights(d, size=len(series))
    width = len(weights)
    res = {}
    vals = series.values
    for i in range(width, len(vals) + 1):
        res[series.index[i - 1]] = np.dot(weights, vals[i - width:i])
    return pd.Series(res)

def estimate_ou_half_life(series):
    """
    Estima la semivida (half-life) de reversión a la media mediante el proceso de Ornstein-Uhlenbeck:
    Delta y_t = theta * (mu - y_{t-1}) Delta t + epsilon_t
    Half-life = ln(2) / lambda
    """
    y = series.values
    delta_y = np.diff(y)
    y_lag = y[:-1]
    
    # Regresión lineal OLS: delta_y = alpha + beta * y_lag
    X = np.vstack([np.ones(len(y_lag)), y_lag]).T
    beta = np.linalg.lstsq(X, delta_y, rcond=None)[0][1]
    
    if beta < 0:
        half_life = -np.log(2.0) / beta
        half_life = float(np.clip(half_life, 2.0, 60.0))
    else:
        half_life = 20.0 # Valor por defecto si la serie tiene tendencia explosiva
        
    return half_life

def generate_timing_recommendations(prices_df, returns_df):
    """
    Genera recomendaciones de dirección (Long/Short) y tiempo óptimo de maduración (días)
    utilizando Machine Learning (Random Forest sobre memoria fraccionaria).
    """
    recommendations = []
    
    for ticker in prices_df.columns:
        p_series = prices_df[ticker].dropna()
        r_series = returns_df[ticker].dropna()
        current_price = p_series.iloc[-1]
        
        # 1. Feature Engineering Fraccionario
        try:
            fd_series = apply_frac_diff(np.log(p_series), d=0.40)
        except Exception:
            fd_series = r_series
            
        common_idx = p_series.index.intersection(fd_series.index)
        fd_aligned = fd_series.loc[common_idx]
        
        # Construcción de matriz de características X
        X = []
        y = []
        vals = fd_aligned.values
        p_vals = p_series.loc[common_idx].values
        
        # Lookback de 10 días para predecir si el retorno a 5 días será positivo o negativo
        lookback = 10
        horizon = 5
        for i in range(lookback, len(vals) - horizon):
            feat = vals[i-lookback:i]
            # Dirección del precio en el horizonte futuro
            target = 1 if p_vals[i + horizon] > p_vals[i] else 0
            X.append(feat)
            y.append(target)
            
        if len(X) > 50:
            X = np.array(X)
            y = np.array(y)
            rf = RandomForestClassifier(n_estimators=30, max_depth=3, random_state=42)
            rf.fit(X[:-20], y[:-20]) # Entrenar con el pasado
            
            # Predicción para el momento actual
            latest_feat = vals[-lookback:].reshape(1, -1)
            prob_up = rf.predict_proba(latest_feat)[0][1]
        else:
            prob_up = 0.50
            
        # 2. Señal Direccional Normalizada en [-1.0, +1.0]
        directional_score = (prob_up - 0.50) * 2.0 # Score de -1 (Short) a +1 (Long)
        
        # 3. Estimación de Maduración (Tiempo de espera) con Ornstein-Uhlenbeck
        half_life_days = estimate_ou_half_life(fd_aligned)
        
        # 4. Volatilidad y Niveles Dinámicos de Salida (Stop-Loss y Take-Profit)
        daily_vol = r_series.iloc[-20:].std()
        
        if directional_score >= 0.25:
            action = "Comprar en Largo (Long)"
            badge = "LONG"
            stop_loss = current_price * (1.0 - 2.0 * daily_vol)
            take_profit = current_price * (1.0 + 3.0 * daily_vol)
            holding_days = int(np.round(half_life_days * 0.8))
        elif directional_score <= -0.25:
            action = "Vender en Corto (Short)"
            badge = "SHORT"
            stop_loss = current_price * (1.0 + 2.0 * daily_vol)
            take_profit = current_price * (1.0 - 3.0 * daily_vol)
            holding_days = int(np.round(half_life_days * 0.8))
        else:
            action = "Neutral / Mantener (Cash)"
            badge = "NEUTRAL"
            stop_loss = current_price * (1.0 - daily_vol)
            take_profit = current_price * (1.0 + daily_vol)
            holding_days = int(np.round(half_life_days))
            
        holding_days = max(3, min(45, holding_days))
        
        recommendations.append({
            'Activo': ticker,
            'Precio Actual ($)': round(current_price, 2),
            'Señal ML': badge,
            'Recomendación': action,
            'Confianza Probabilística': f"{int(max(prob_up, 1.0 - prob_up) * 100)}%",
            'Score Direccional': round(directional_score, 3),
            'Maduración Sugerida (Días)': holding_days,
            'Take-Profit Sugerido ($)': round(take_profit, 2),
            'Stop-Loss Dinámico ($)': round(stop_loss, 2),
            'Volatilidad Diaria': f"{round(daily_vol * 100, 2)}%"
        })
        
    return pd.DataFrame(recommendations)

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

def ledoit_wolf_covariance(returns):
    """
    Calcula el estimador de contracción de covarianza de Ledoit-Wolf.
    Retorna la covarianza encogida, la intensidad de contracción delta y la covarianza muestral.
    """
    lw = LedoitWolf()
    lw.fit(returns)
    shrunk_cov = lw.covariance_ * 252 # Anualizado
    shrinkage_intensity = lw.shrinkage_
    sample_cov = np.cov(returns, rowvar=False) * 252 # Anualizado
    
    return {
        'shrunk_cov': shrunk_cov,
        'shrinkage_intensity': shrinkage_intensity,
        'sample_cov': sample_cov,
        'feature_names': list(returns.columns)
    }

def optimize_portfolio_utility(returns, gamma, budget, latest_prices, spreads=None, 
                               broker_fee_bps=10.0, current_weights=None, allow_short=False,
                               max_weight_per_asset=0.30, min_weight_per_asset=0.0):
    """
    Optimiza la cartera maximizando la Utilidad Esperada cuadrática:
    Max w^T mu - (gamma / 2) * w^T Sigma_LW w - sum [ c_broker * |w - w_0| + (spread / 2) * |w| ] + lambda_div * sum(ln(w))
    Sujeto a: sum(w) <= 1, min_w <= w_i <= max_w.
    """
    n_assets = returns.shape[1]
    asset_names = list(returns.columns)
    
    # 1. Parámetros de Retorno y Covarianza Regularizada
    ann_returns = returns.mean().values * 252
    lw_result = ledoit_wolf_covariance(returns)
    cov_lw = lw_result['shrunk_cov']
    cov_sample = lw_result['sample_cov']
    shrinkage = lw_result['shrinkage_intensity']
    
    # 2. Costos de Fricción (Comisión de corretaje y bid-ask spread)
    c_broker = broker_fee_bps / 10000.0 # ej. 10 bps = 0.0010
    if spreads is None:
        spreads_vec = np.full(n_assets, 0.0010)
    else:
        spreads_vec = np.array([spreads.get(a, 0.0010) for a in asset_names])
        
    if current_weights is None:
        w0 = np.zeros(n_assets)
    else:
        w0 = np.array([current_weights.get(a, 0.0) for a in asset_names])
        
    # 3. Función Objetivo: Utilidad Negativa (para minimizar)
    def objective_utility(w, cov_matrix):
        # Retorno esperado
        port_ret = np.dot(w, ann_returns)
        # Penalización por varianza ponderada por aversión al riesgo gamma
        port_var = np.dot(w.T, np.dot(cov_matrix, w))
        # Costo de fricción por rebalanceo y spread
        turnover = np.sum(np.abs(w - w0))
        costs = c_broker * turnover + 0.5 * np.sum(spreads_vec * np.abs(w))
        
        # Penalización de entropía suave para evitar monopolios de 1 solo activo (fomenta diversificación)
        entropy_bonus = 0.0015 * np.sum(np.log(np.maximum(w, 1e-5))) if not allow_short else 0.0
        
        utility = port_ret - 0.5 * gamma * port_var - costs + entropy_bonus
        return -utility # Minimizar negativa

    # Restricciones
    # Inversión total en activos <= 1.0 (el remanente queda en efectivo)
    constraints = [
        {'type': 'ineq', 'fun': lambda w: 1.0 - np.sum(w)}
    ]
    
    # Límites por activo (Tope máximo de concentración)
    upper_b = float(max_weight_per_asset) if max_weight_per_asset is not None else 1.0
    upper_b = max(upper_b, 1.05 / n_assets) # Garantizar viabilidad numérica de sum(w) <= 1
    
    if allow_short:
        bounds = [(-0.20, upper_b) for _ in range(n_assets)]
    else:
        bounds = [(min_weight_per_asset, upper_b) for _ in range(n_assets)]
        
    w_initial = np.ones(n_assets) / n_assets
    
    # Optimización 1: Ledoit-Wolf Regularizado
    res_lw = minimize(objective_utility, w_initial, args=(cov_lw,),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    w_opt_lw = res_lw.x
    w_opt_lw[np.abs(w_opt_lw) < 1e-4] = 0.0 # Limpiar pesos residuales
    
    # Optimización 2: Covarianza Muestral Clásica (para comparar sobreajuste)
    res_sample = minimize(objective_utility, w_initial, args=(cov_sample,),
                          method='SLSQP', bounds=bounds, constraints=constraints)
    w_opt_sample = res_sample.x
    w_opt_sample[np.abs(w_opt_sample) < 1e-4] = 0.0

    # 4. Asignación Presupuestaria Discreta (Enteros de Títulos)
    shares = {}
    invested_cash = {}
    total_invested = 0.0
    
    for i, a in enumerate(asset_names):
        p = latest_prices.get(a, 100.0)
        target_dollar = w_opt_lw[i] * budget
        if target_dollar >= p:
            n_shares = int(np.floor(target_dollar / p))
        else:
            n_shares = 0
        shares[a] = n_shares
        actual_dollar = n_shares * p
        invested_cash[a] = actual_dollar
        total_invested += actual_dollar
        
    cash_remaining = budget - total_invested
    
    # Métricas del portafolio óptimo
    port_ret_lw = float(np.dot(w_opt_lw, ann_returns))
    port_vol_lw = float(np.sqrt(np.dot(w_opt_lw.T, np.dot(cov_lw, w_opt_lw))))
    sharpe_lw = (port_ret_lw - 0.035) / (port_vol_lw + 1e-6) # Tasa libre de riesgo 3.5%
    
    port_ret_samp = float(np.dot(w_opt_sample, ann_returns))
    port_vol_samp = float(np.sqrt(np.dot(w_opt_sample.T, np.dot(cov_sample, w_opt_sample))))
    sharpe_samp = (port_ret_samp - 0.035) / (port_vol_samp + 1e-6)
    
    total_costs_paid = float(c_broker * np.sum(np.abs(w_opt_lw - w0)) + 0.5 * np.sum(spreads_vec * np.abs(w_opt_lw))) * budget

    return {
        'asset_names': asset_names,
        'weights_lw': pd.Series(w_opt_lw, index=asset_names),
        'weights_sample': pd.Series(w_opt_sample, index=asset_names),
        'shares': pd.Series(shares),
        'invested_cash': pd.Series(invested_cash),
        'total_invested': total_invested,
        'cash_remaining': cash_remaining,
        'port_ret_lw': port_ret_lw,
        'port_vol_lw': port_vol_lw,
        'sharpe_lw': sharpe_lw,
        'port_ret_samp': port_ret_samp,
        'port_vol_samp': port_vol_samp,
        'sharpe_samp': sharpe_samp,
        'shrinkage_intensity': shrinkage,
        'total_costs_paid': total_costs_paid,
        'cov_lw': cov_lw,
        'cov_sample': cov_sample
    }

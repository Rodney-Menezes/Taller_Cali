import numpy as np
import pandas as pd

def simulate_multivariate_portfolio_mc(
    shares_dict,
    latest_prices,
    annual_returns,
    cov_matrix,
    initial_budget,
    cash_buffer,
    asset_names,
    time_horizon_days=252,
    n_simulations=500,
    risk_free_rate=0.035,
    random_seed=42
):
    """
    Simulación Monte Carlo Multivariada de TODO el Portafolio:
    Aplica descomposición de Cholesky sobre la covarianza regularizada de Ledoit-Wolf
    para simular los precios correlacionados de todos los activos conjuntamente,
    y computa la evolución del patrimonio neto (acciones + caja) en $ USD.
    """
    np.random.seed(random_seed)
    n_assets = len(asset_names)
    
    # 1. Parámetros diarios y Descomposición de Cholesky
    dt = 1.0 / 252.0
    daily_cov = cov_matrix * dt
    # Estabilización numérica con jitter
    jitter = 1e-7 * np.eye(n_assets)
    try:
        L = np.linalg.cholesky(daily_cov + jitter)
    except np.linalg.LinAlgError:
        # Fallback de autovalores si hay problemas de definición positiva
        eigvals, eigvecs = np.linalg.eigh(daily_cov)
        eigvals = np.maximum(eigvals, 1e-7)
        reconstructed = eigvecs @ np.diag(eigvals) @ eigvecs.T
        L = np.linalg.cholesky(reconstructed + jitter)

    # Derivas ajustadas por convexidad de Ito: mu - 0.5 * sigma^2
    variances = np.diag(daily_cov)
    drift = (annual_returns * dt) - 0.5 * variances
    
    # Precios iniciales y títulos
    s0 = np.array([latest_prices[a] for a in asset_names])
    shares = np.array([shares_dict.get(a, 0) for a in asset_names])
    
    # Matriz para almacenar la riqueza del portafolio: [dias + 1, simulaciones]
    wealth_paths = np.zeros((time_horizon_days + 1, n_simulations))
    # Inicialización en t = 0
    init_portfolio_val = np.sum(shares * s0) + cash_buffer
    wealth_paths[0, :] = init_portfolio_val
    
    # Precios actuales por simulación: [n_assets, n_simulations]
    current_prices = np.tile(s0[:, np.newaxis], (1, n_simulations))
    
    # Tasa libre de riesgo diaria para el buffer de caja
    daily_rf = (1.0 + risk_free_rate) ** dt - 1.0
    current_cash = np.full(n_simulations, cash_buffer)
    
    for t in range(1, time_horizon_days + 1):
        # 2. Generar choques normales estándar e inducir correlación con Cholesky
        Z = np.random.normal(0, 1, size=(n_assets, n_simulations))
        correlated_shocks = L @ Z # [n_assets, n_simulations]
        
        # 3. Evolución geométrica browniana correlacionada
        returns_step = np.exp(drift[:, np.newaxis] + correlated_shocks)
        current_prices = current_prices * returns_step
        
        # 4. Acumulación de caja
        current_cash = current_cash * (1.0 + daily_rf)
        
        # 5. Valor total de la cartera: sum(titulos * precios) + caja
        portfolio_equity = np.sum(shares[:, np.newaxis] * current_prices, axis=0)
        wealth_paths[t, :] = portfolio_equity + current_cash

    # -------------------------------------------------------------
    # CÁLCULO DE MÉTRICAS INSTITUCIONALES DE RIESGO Y RETORNO
    # -------------------------------------------------------------
    final_wealth = wealth_paths[-1, :] # Distribución en T
    final_returns_pct = (final_wealth - initial_budget) / initial_budget
    
    expected_wealth = float(np.mean(final_wealth))
    median_wealth = float(np.median(final_wealth))
    std_wealth = float(np.std(final_wealth))
    
    # Percentiles
    p01 = float(np.percentile(final_wealth, 1))
    p05 = float(np.percentile(final_wealth, 5))
    p25 = float(np.percentile(final_wealth, 25))
    p75 = float(np.percentile(final_wealth, 75))
    p95 = float(np.percentile(final_wealth, 95))
    p99 = float(np.percentile(final_wealth, 99))
    
    # Value at Risk (VaR)
    var_95_dollar = max(0.0, float(initial_budget - p05))
    var_95_pct = (var_95_dollar / initial_budget) * 100.0
    
    var_99_dollar = max(0.0, float(initial_budget - p01))
    var_99_pct = (var_99_dollar / initial_budget) * 100.0
    
    # Conditional Value at Risk (CVaR / Expected Shortfall al 95%)
    worst_5_pct = final_wealth[final_wealth <= p05]
    if len(worst_5_pct) > 0:
        cvar_95_dollar = max(0.0, float(initial_budget - np.mean(worst_5_pct)))
    else:
        cvar_95_dollar = var_95_dollar
    cvar_95_pct = (cvar_95_dollar / initial_budget) * 100.0
    
    # Probabilidad de pérdida de capital (W_T < W_0)
    prob_loss = float(np.mean(final_wealth < initial_budget) * 100.0)
    
    # Drawdown máximo promedio a lo largo de las trayectorias
    peaks = np.maximum.accumulate(wealth_paths, axis=0)
    drawdowns = (wealth_paths - peaks) / peaks
    max_drawdown_avg = float(np.abs(np.mean(np.min(drawdowns, axis=0))) * 100.0)

    # Cono de riqueza temporal para gráfico de abanico
    time_index = np.arange(time_horizon_days + 1)
    cone_df = pd.DataFrame({
        'Dia': time_index,
        'P05': np.percentile(wealth_paths, 5, axis=1),
        'P25': np.percentile(wealth_paths, 25, axis=1),
        'Mediana': np.median(wealth_paths, axis=1),
        'P75': np.percentile(wealth_paths, 75, axis=1),
        'P95': np.percentile(wealth_paths, 95, axis=1),
        'Media': np.mean(wealth_paths, axis=1)
    })

    return {
        'wealth_paths': wealth_paths,
        'final_wealth': final_wealth,
        'cone_df': cone_df,
        'initial_budget': initial_budget,
        'expected_wealth': expected_wealth,
        'median_wealth': median_wealth,
        'std_wealth': std_wealth,
        'p05': p05,
        'p95': p95,
        'var_95_dollar': var_95_dollar,
        'var_95_pct': var_95_pct,
        'var_99_dollar': var_99_dollar,
        'var_99_pct': var_99_pct,
        'cvar_95_dollar': cvar_95_dollar,
        'cvar_95_pct': cvar_95_pct,
        'prob_loss': prob_loss,
        'max_drawdown_avg': max_drawdown_avg,
        'time_horizon_days': time_horizon_days,
        'n_simulations': n_simulations
    }

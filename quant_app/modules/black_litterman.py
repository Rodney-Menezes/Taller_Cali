import numpy as np
import pandas as pd
import plotly.graph_objects as go

def calculate_black_litterman(cov_matrix, asset_names, dl_signals_dict, latest_prices=None, budget=10000.0, tau=0.05, gamma=3.0, market_weights=None):
    """
    Implementa el modelo de Black-Litterman impulsado por Inteligencia Artificial (AI-Driven Black-Litterman).
    Combina el equilibrio implícito de mercado (CAPM Pi) con las vistas cuantitativas generadas por la
    Red Neuronal BiLSTM con Auto-Atención Temporal (PyTorch).
    """
    n = len(asset_names)
    cov = np.array(cov_matrix, dtype=np.float64)
    
    if n <= 1:
        w_dict = {asset_names[0]: 1.0} if n == 1 else {}
        return {
            'weights': w_dict,
            'mu_bl': {asset_names[0]: 0.08} if n == 1 else {},
            'implied_pi': {asset_names[0]: 0.08} if n == 1 else {},
            'fig_comparison': None,
            'shares': {},
            'cash_buffer': budget
        }
        
    # 1. Pesos de equilibrio de mercado (Default: ponderación proporcional o inversa a volatilidad)
    if market_weights is None:
        vols = np.sqrt(np.diag(cov))
        inv_vols = 1.0 / np.clip(vols, 1e-4, None)
        w_mkt = inv_vols / np.sum(inv_vols)
    else:
        w_mkt = np.array([market_weights.get(a, 1.0 / n) for a in asset_names])
        w_mkt /= np.sum(w_mkt)
        
    # 2. Retornos de equilibrio implícitos del mercado (Prior CAPM)
    # Pi = gamma * Sigma * w_mkt
    pi = gamma * np.dot(cov, w_mkt)
    
    # 3. Construcción de Vistas Cuantitativas de la IA (P, Q, Omega)
    P_list = []
    Q_list = []
    omega_diag = []
    views_info = []
    
    for i, a in enumerate(asset_names):
        if a in dl_signals_dict:
            res = dl_signals_dict[a]
            probs = res.get('probabilities', [0.33, 0.33, 0.34])
            conf = res.get('confidence', 50.0) / 100.0
            
            # Convicción neta direccional (-1 a +1)
            p_short = probs[0]
            p_long = probs[2]
            conviction = float(p_long - p_short)
            
            # Solo incorporar vistas donde la red tenga una postura definida
            if abs(conviction) > 0.04:
                p_row = np.zeros(n)
                p_row[i] = 1.0
                
                # Retorno esperado subjetivo modelado: proporcional a la convicción de la red
                # y ajustado a la volatilidad histórica del activo
                asset_vol = np.sqrt(cov[i, i])
                expected_view_return = float(pi[i] + conviction * asset_vol * 1.5)
                
                # Incertidumbre de la vista (Omega): menor confianza = mayor varianza/incertidumbre
                view_var = float(tau * cov[i, i] * (1.0 / max(conf, 0.20)))
                
                P_list.append(p_row)
                Q_list.append(expected_view_return)
                omega_diag.append(view_var)
                
                views_info.append({
                    'Activo': a,
                    'Señal AI': res.get('signal', 'NEUTRAL'),
                    'Convicción': f"{conviction*100:+.1f}%",
                    'Retorno Esperado Vista (Q)': f"{expected_view_return*100:.2f}%",
                    'Confianza': f"{conf*100:.1f}%"
                })
                
    # 4. Cálculo de la Distribución Posterior de Black-Litterman
    if len(P_list) > 0:
        P = np.array(P_list)
        Q = np.array(Q_list)
        Omega = np.diag(omega_diag)
        
        # Inversas regulares con estabilización de pseudoinversa
        inv_tau_cov = np.linalg.pinv(tau * cov)
        inv_omega = np.linalg.pinv(Omega)
        
        # Matriz M = [ (tau*Sigma)^-1 + P^T * Omega^-1 * P ]^-1
        M = np.linalg.pinv(inv_tau_cov + np.dot(P.T, np.dot(inv_omega, P)))
        
        # Vector posterior mu_BL = M * [ (tau*Sigma)^-1 * Pi + P^T * Omega^-1 * Q ]
        term1 = np.dot(inv_tau_cov, pi)
        term2 = np.dot(P.T, np.dot(inv_omega, Q))
        mu_bl = np.dot(M, term1 + term2)
        
        # Covarianza posterior Sigma_BL = Sigma + M
        sigma_bl = cov + M
    else:
        mu_bl = pi
        sigma_bl = cov * (1.0 + tau)
        
    # 5. Ponderaciones óptimas resultantes de Black-Litterman
    # w_BL = (1 / gamma) * Sigma_BL^-1 * mu_BL
    inv_sigma_bl = np.linalg.pinv(sigma_bl)
    w_raw = (1.0 / gamma) * np.dot(inv_sigma_bl, mu_bl)
    
    # Restricción de no venta en corto y normalización a 100%
    w_clean = np.maximum(w_raw, 0.0)
    if np.sum(w_clean) > 0:
        w_clean /= np.sum(w_clean)
    else:
        w_clean = np.ones(n) / n
        
    # Limitar concentración máxima (35% por defecto)
    w_clean = np.minimum(w_clean, 0.35)
    w_clean /= np.sum(w_clean)
    
    weights_dict = {asset_names[i]: float(w_clean[i]) for i in range(n)}
    pi_dict = {asset_names[i]: float(pi[i]) for i in range(n)}
    mu_bl_dict = {asset_names[i]: float(mu_bl[i]) for i in range(n)}
    
    # 6. Gráfico Comparativo: Retornos Previos CAPM vs Posteriores Black-Litterman
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        x=asset_names, y=[pi_dict[a] * 100.0 for a in asset_names],
        name='Retorno Implícito de Equilibrio (CAPM Pi)',
        marker_color='#90A4AE'
    ))
    fig_comp.add_trace(go.Bar(
        x=asset_names, y=[mu_bl_dict[a] * 100.0 for a in asset_names],
        name='Retorno Posterior Black-Litterman (AI-Driven)',
        marker_color='#00E676'
    ))
    fig_comp.update_layout(
        title="Retornos Anualizados: Equilibrio Implícito de Mercado vs. Black-Litterman con IA",
        xaxis_title="Activo", yaxis_title="Rendimiento Esperado Anual (%)",
        barmode='group', height=320, margin=dict(l=20, r=20, t=40, b=20)
    )
    
    # 7. Asignación Discreta en USD
    shares_dict = {}
    invested_total = 0.0
    if latest_prices is not None:
        for a in asset_names:
            p = latest_prices.get(a, 0.0)
            target_alloc = budget * weights_dict[a]
            sh = int(np.floor(target_alloc / p)) if p > 0 else 0
            shares_dict[a] = sh
            invested_total += sh * p
            
    cash_rem = max(0.0, budget - invested_total)
    
    return {
        'weights': weights_dict,
        'mu_bl': mu_bl_dict,
        'implied_pi': pi_dict,
        'views_table': pd.DataFrame(views_info) if len(views_info) > 0 else pd.DataFrame(),
        'fig_comparison': fig_comp,
        'shares': shares_dict,
        'invested_capital': invested_total,
        'cash_buffer': cash_rem
    }

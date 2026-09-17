import numpy as np
import pandas as pd

def calculate_bond_metrics(face_value=1000.0, coupon_rate=0.05, ytm=0.045, maturity_years=10, freq=2):
    """
    Calcula la Dinámica Analítica de Renta Fija:
    - Precio del bono
    - Duración de Macaulay
    - Duración Modificada (D*)
    - Convexidad (C)
    - DV01 (Dollar Value of 01 basis point)
    """
    m = freq # Pagos por año (ej. 2 = semestral)
    n_periods = int(maturity_years * m)
    coupon_pmt = (coupon_rate * face_value) / m
    r_period = ytm / m
    
    times = np.arange(1, n_periods + 1) / m
    cash_flows = np.full(n_periods, coupon_pmt)
    cash_flows[-1] += face_value
    
    # Factores de descuento
    discount_factors = 1.0 / (1.0 + r_period) ** (times * m)
    pv_cash_flows = cash_flows * discount_factors
    bond_price = float(np.sum(pv_cash_flows))
    
    # 1. Duración de Macaulay (en años)
    mac_duration = float(np.sum(times * pv_cash_flows) / bond_price)
    
    # 2. Duración Modificada (D*)
    mod_duration = float(mac_duration / (1.0 + r_period))
    
    # 3. DV01 (Valor en dólares de 1 punto básico de variación en tasa)
    dv01 = float(mod_duration * bond_price * 0.0001)
    
    # 4. Convexidad (C)
    t_conv = times * (times + 1.0 / m)
    convexity = float(np.sum(t_conv * pv_cash_flows) / (bond_price * (1.0 + r_period) ** 2))
    
    return {
        'bond_price': bond_price,
        'mac_duration': mac_duration,
        'mod_duration': mod_duration,
        'convexity': convexity,
        'dv01': dv01,
        'face_value': face_value,
        'coupon_rate': coupon_rate,
        'ytm': ytm,
        'maturity_years': maturity_years
    }

def simulate_yield_shocks(bond_metrics, shock_bps_range=200, n_points=50):
    """
    Simula el impacto de un shock en las tasas de interés (Yield Shocks Delta y).
    Compara:
    1. Precio Exacto (Full Re-pricing)
    2. Aproximación Lineal (Sólo Duración: -D* * Delta y)
    3. Aproximación Cuadrática (Duración + Convexidad: -D* * Delta y + 0.5 * C * (Delta y)^2)
    """
    shocks_bps = np.linspace(-shock_bps_range, shock_bps_range, n_points)
    shocks_decimal = shocks_bps / 10000.0 # ej. 100 bps = 0.01
    
    base_price = bond_metrics['bond_price']
    d_mod = bond_metrics['mod_duration']
    c_conv = bond_metrics['convexity']
    fv = bond_metrics['face_value']
    c_rate = bond_metrics['coupon_rate']
    mat = bond_metrics['maturity_years']
    base_ytm = bond_metrics['ytm']
    
    exact_prices = []
    duration_approx = []
    convexity_approx = []
    
    for dy in shocks_decimal:
        # 1. Full Re-pricing Exacto
        new_ytm = base_ytm + dy
        new_metrics = calculate_bond_metrics(face_value=fv, coupon_rate=c_rate, ytm=new_ytm, maturity_years=mat)
        exact_prices.append(new_metrics['bond_price'])
        
        # 2. Aproximación de Duración (Lineal)
        pct_change_dur = -d_mod * dy
        duration_approx.append(base_price * (1.0 + pct_change_dur))
        
        # 3. Aproximación de Duración + Convexidad (Segundo orden)
        pct_change_conv = -d_mod * dy + 0.5 * c_conv * (dy ** 2)
        convexity_approx.append(base_price * (1.0 + pct_change_conv))
        
    df_shocks = pd.DataFrame({
        'Shock_bps': shocks_bps,
        'Shock_decimal': shocks_decimal,
        'Precio_Exacto': exact_prices,
        'Aprox_Duracion_Lineal': duration_approx,
        'Aprox_Duracion_Convexidad': convexity_approx
    })
    
    return df_shocks

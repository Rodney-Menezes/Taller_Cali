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

def get_fixed_income_profile(ticker, current_price=None):
    """
    Retorna el perfil institucional y dinámico de los principales instrumentos de Renta Fija
    disponibles en Yahoo Finance: TLT, IEF, SHY, BND, LQD, HYG, TIP, AGG
    """
    profiles = {
        'TLT': {
            'name': 'iShares 20+ Year Treasury Bond ETF',
            'category': 'Tesoro EE.UU. a Largo Plazo (20+ Años)',
            'face_value': 1000.0,
            'coupon_rate': 0.0425,
            'ytm': 0.0455,
            'maturity_years': 25,
            'effective_duration': 16.8,
            'convexity': 3.95,
            'credit_rating': 'AAA (Soberano EE.UU.)',
            'description': 'Máxima sensibilidad a las tasas de interés de largo plazo. Actúa como activo refugio durante caídas severas de renta variable.'
        },
        'IEF': {
            'name': 'iShares 7-10 Year Treasury Bond ETF',
            'category': 'Tesoro EE.UU. a Mediano Plazo (7-10 Años)',
            'face_value': 1000.0,
            'coupon_rate': 0.0380,
            'ytm': 0.0420,
            'maturity_years': 8,
            'effective_duration': 7.6,
            'convexity': 0.85,
            'credit_rating': 'AAA (Soberano EE.UU.)',
            'description': 'Sensibilidad intermedia (benchmark del bono a 10 años). Equilibrio entre protección contra desaceleración y moderado riesgo de tasa.'
        },
        'SHY': {
            'name': 'iShares 1-3 Year Treasury Bond ETF',
            'category': 'Tesoro EE.UU. a Corto Plazo (1-3 Años)',
            'face_value': 1000.0,
            'coupon_rate': 0.0450,
            'ytm': 0.0465,
            'maturity_years': 2,
            'effective_duration': 1.9,
            'convexity': 0.06,
            'credit_rating': 'AAA (Soberano EE.UU.)',
            'description': 'Mínimo riesgo de tasa de interés (cuasi-caja). Preserva capital con rendimiento estable cuando la curva de rendimientos está plana.'
        },
        'BND': {
            'name': 'Vanguard Total Bond Market ETF',
            'category': 'Renta Fija Agregada Grado Inversión (EE.UU.)',
            'face_value': 1000.0,
            'coupon_rate': 0.0410,
            'ytm': 0.0470,
            'maturity_years': 8,
            'effective_duration': 6.4,
            'convexity': 0.62,
            'credit_rating': 'AA+ (65% Gobierno, 35% Corporativo)',
            'description': 'Exposición diversificada al mercado completo de deuda grado de inversión de EE.UU. (gobierno y corporaciones de alta calidad).'
        },
        'LQD': {
            'name': 'iShares iBoxx $ Investment Grade Corporate Bond ETF',
            'category': 'Deuda Corporativa Grado de Inversión',
            'face_value': 1000.0,
            'coupon_rate': 0.0520,
            'ytm': 0.0535,
            'maturity_years': 12,
            'effective_duration': 8.3,
            'convexity': 1.15,
            'credit_rating': 'BBB a AAA (Corporativo Institucional)',
            'description': 'Mayor cupón corriente que los bonos soberanos a cambio de asumir riesgo de crédito corporativo de alta calidad.'
        },
        'HYG': {
            'name': 'iShares iBoxx $ High Yield Corporate Bond ETF',
            'category': 'Deuda Corporativa High Yield / Alto Rendimiento',
            'face_value': 1000.0,
            'coupon_rate': 0.0680,
            'ytm': 0.0715,
            'maturity_years': 4,
            'effective_duration': 3.7,
            'convexity': 0.22,
            'credit_rating': 'BB a CCC (Alto Rendimiento)',
            'description': 'Elevado flujo de caja corriente. Mayor correlación positiva con renta variable debido al ciclo económico y menor duración.'
        },
        'TIP': {
            'name': 'iShares TIPS Bond ETF',
            'category': 'Bonos Protegidos contra la Inflación (TIPS)',
            'face_value': 1000.0,
            'coupon_rate': 0.0220,
            'ytm': 0.0210,
            'maturity_years': 7,
            'effective_duration': 6.8,
            'convexity': 0.70,
            'credit_rating': 'AAA (Soberano EE.UU.)',
            'description': 'El capital principal ajusta semestralmente con la inflación de EE.UU. Ofrece cobertura ante sorpresas del IPC.'
        }
    }
    
    prof = profiles.get(ticker.upper(), {
        'name': f'Instrumento de Renta Fija ({ticker})',
        'category': 'Renta Fija General',
        'face_value': 1000.0,
        'coupon_rate': 0.0450,
        'ytm': 0.0450,
        'maturity_years': 10,
        'effective_duration': 7.5,
        'convexity': 0.80,
        'credit_rating': 'Grado Inversión',
        'description': f'Instrumento de renta fija o ETF ({ticker}) cotizado en Yahoo Finance.'
    })
    
    metrics = calculate_bond_metrics(
        face_value=prof['face_value'],
        coupon_rate=prof['coupon_rate'],
        ytm=prof['ytm'],
        maturity_years=prof['maturity_years']
    )
    
    if current_price is not None:
        prof['market_price'] = float(current_price)
        prof['dv01_share'] = float(prof['effective_duration'] * current_price * 0.0001)
    else:
        prof['market_price'] = metrics['bond_price']
        prof['dv01_share'] = metrics['dv01']
        
    prof['metrics'] = metrics
    return prof

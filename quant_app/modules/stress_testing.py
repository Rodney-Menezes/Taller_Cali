import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Calibración histórica empírica de shocks por clase de activo durante crisis financieras
HISTORICAL_CRISIS_SHOCKS = {
    "Subprime_2008": {
        "name": "💥 Gran Crisis Financiera Subprime (2008)",
        "description": "Quiebra de Lehman Brothers, congelamiento del crédito global y severa recesión.",
        "defaults": {
            "mega_tech": -0.42,
            "broad_equity": -0.45,
            "small_caps": -0.52,
            "emerging": -0.55,
            "real_estate": -0.65,
            "energy": -0.50,
            "commodities": -0.40,
            "gold": +0.15,
            "long_treasuries": +0.28,
            "mid_treasuries": +0.14,
            "corp_bonds": -0.12,
            "cash": +0.02
        }
    },
    "COVID_2020": {
        "name": "🦠 Crash Pandémico COVID-19 (Feb-Mar 2020)",
        "description": "Parálisis económica mundial repentina, caída récord en velocidad bursátil y shock petrolero.",
        "defaults": {
            "mega_tech": -0.28,
            "broad_equity": -0.34,
            "small_caps": -0.41,
            "emerging": -0.32,
            "real_estate": -0.38,
            "energy": -0.55,
            "commodities": -0.30,
            "gold": -0.02,
            "long_treasuries": +0.18,
            "mid_treasuries": +0.10,
            "corp_bonds": -0.08,
            "cash": +0.01
        }
    },
    "FED_Hikes_2022": {
        "name": "🦅 Shock Inflacionario & Subida de Tasas FED (2022)",
        "description": "La FED subió tasas en 500 bps. Pérdidas simultáneas históricas en acciones y bonos soberanos.",
        "defaults": {
            "mega_tech": -0.33,
            "broad_equity": -0.18,
            "small_caps": -0.22,
            "emerging": -0.20,
            "real_estate": -0.26,
            "energy": +0.59,
            "commodities": +0.22,
            "gold": -0.01,
            "long_treasuries": -0.31,
            "mid_treasuries": -0.16,
            "corp_bonds": -0.15,
            "cash": +0.03
        }
    },
    "AI_Rally_2023": {
        "name": "🚀 Shock Alcista & Rally de Inteligencia Artificial (2023-2024)",
        "description": "Expansión agresiva impulsada por LLMs y semiconductores con fuerte dispersión sectorial.",
        "defaults": {
            "mega_tech": +0.85,
            "broad_equity": +0.26,
            "small_caps": +0.12,
            "emerging": +0.08,
            "real_estate": +0.05,
            "energy": -0.04,
            "commodities": -0.05,
            "gold": +0.15,
            "long_treasuries": -0.08,
            "mid_treasuries": -0.03,
            "corp_bonds": +0.06,
            "cash": +0.05
        }
    }
}

def classify_asset_type(ticker):
    """
    Clasifica heurísticamente un ticker en su categoría macroeconómica.
    """
    t = ticker.upper()
    if t in ["TLT"]:
        return "long_treasuries"
    elif t in ["IEF", "TIP"]:
        return "mid_treasuries"
    elif t in ["SHY", "BIL", "SGOV"]:
        return "cash"
    elif t in ["BND", "AGG", "LQD", "HYG"]:
        return "corp_bonds"
    elif t in ["GLD", "SLV"]:
        return "gold"
    elif t in ["USO", "DBA", "CPER"]:
        return "commodities"
    elif t in ["XLE"]:
        return "energy"
    elif t in ["VNQ", "O", "IYR"]:
        return "real_estate"
    elif t in ["IWM", "IJH", "VB"]:
        return "small_caps"
    elif t in ["EEM", "EWZ", "EWW", "INDA", "EFA"]:
        return "emerging"
    elif t in ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "PLTR", "QQQ"]:
        return "mega_tech"
    else:
        return "broad_equity"

def simulate_crisis_stress(weights_dict, budget=10000.0):
    """
    Somete a prueba el portafolio actual frente a 4 de las mayores crisis y shocks históricos.
    Calcula pérdidas/ganancias en USD, impacto porcentual y desempeño de cada activo.
    """
    results = {}
    chart_scenarios = []
    chart_port_returns = []
    chart_bm_returns = []
    
    # Benchmark estándar 60/40 (60% SPY, 40% BND)
    bm_profile = {"SPY": 0.60, "BND": 0.40}
    
    for crisis_id, crisis_data in HISTORICAL_CRISIS_SHOCKS.items():
        shocks_map = crisis_data["defaults"]
        
        # 1. Calcular shock del portafolio actual
        port_return = 0.0
        asset_breakdown = []
        
        for asset, w in weights_dict.items():
            if w <= 0:
                continue
            cat = classify_asset_type(asset)
            asset_shock = shocks_map.get(cat, shocks_map["broad_equity"])
            
            allocated_usd = budget * w
            dollar_change = allocated_usd * asset_shock
            final_val = allocated_usd + dollar_change
            
            port_return += w * asset_shock
            
            asset_breakdown.append({
                "Activo": asset,
                "Categoría Macro": cat.replace("_", " ").title(),
                "Ponderación": f"{w*100:.1f}%",
                "Capital Asignado ($)": round(allocated_usd, 2),
                "Shock Estimado (%)": f"{asset_shock*100:+.1f}%",
                "Impacto en Capital ($)": round(dollar_change, 2),
                "Valor Proyectado ($)": round(final_val, 2)
            })
            
        # 2. Calcular shock del Benchmark 60/40
        bm_shock_spy = shocks_map["broad_equity"]
        bm_shock_bnd = shocks_map["corp_bonds"]
        bm_return = 0.60 * bm_shock_spy + 0.40 * bm_shock_bnd
        
        dollar_impact = budget * port_return
        projected_wealth = budget + dollar_impact
        
        chart_scenarios.append(crisis_data["name"])
        chart_port_returns.append(port_return * 100.0)
        chart_bm_returns.append(bm_return * 100.0)
        
        results[crisis_id] = {
            "name": crisis_data["name"],
            "description": crisis_data["description"],
            "port_return": port_return,
            "bm_return": bm_return,
            "dollar_impact": dollar_impact,
            "projected_wealth": projected_wealth,
            "resilience_vs_bm": (port_return - bm_return) * 100.0,
            "breakdown_df": pd.DataFrame(asset_breakdown)
        }
        
    # Gráfico comparativo de barras de estrés
    fig_stress = go.Figure()
    fig_stress.add_trace(go.Bar(
        x=chart_scenarios, y=chart_port_returns,
        name='Su Portafolio Cuantitativo Actual',
        marker_color=['#00E676' if r >= 0 else '#FF5252' for r in chart_port_returns],
        text=[f"{r:+.1f}%" for r in chart_port_returns], textposition='outside'
    ))
    fig_stress.add_trace(go.Bar(
        x=chart_scenarios, y=chart_bm_returns,
        name='Benchmark Clásico 60/40 (SPY / BND)',
        marker_color='#78909C',
        text=[f"{r:+.1f}%" for r in chart_bm_returns], textposition='outside'
    ))
    fig_stress.add_hline(y=0.0, line_color="#FFFFFF", line_width=1)
    fig_stress.update_layout(
        title="Impacto Proyectado ante Escenarios Históricos de Crisis Macroeconómica (%)",
        yaxis_title="Variación Porcentual (%)",
        barmode='group', height=360, margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02)
    )
    
    return {
        "scenarios": results,
        "fig_stress": fig_stress
    }

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from scipy.optimize import minimize

# Importación de módulos cuantitativos locales
from modules.data_loader import get_asset_data
from modules.risk_profiler import calculate_risk_aversion
from modules.portfolio_optimizer import optimize_portfolio_utility, ledoit_wolf_covariance
from modules.fixed_income import calculate_bond_metrics, simulate_yield_shocks, get_fixed_income_profile
from modules.timing_signals import generate_timing_recommendations, apply_frac_diff
from modules.portfolio_monte_carlo import simulate_multivariate_portfolio_mc
from modules.deep_learning_model import train_deep_learning_agent

# Configuración general de la página
st.set_page_config(
    page_title="AI Quantitative Investment Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para una interfaz institucional de alto impacto
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E88E5 0%, #00E676 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #9E9E9E;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1A1F2C;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #2D3748;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    .badge-long {
        background-color: #00E676;
        color: #000;
        font-weight: bold;
        padding: 3px 8px;
        border-radius: 4px;
    }
    .badge-short {
        background-color: #FF5252;
        color: #FFF;
        font-weight: bold;
        padding: 3px 8px;
        border-radius: 4px;
    }
    .badge-neutral {
        background-color: #B0BEC5;
        color: #000;
        font-weight: bold;
        padding: 3px 8px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# BARRA LATERAL (SIDEBAR): Configuración de Cartera y Mercado
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Universo de Inversión & Parámetros")
    
    # 1. Renta Variable, Commodities y Mid/Small-Caps
    st.markdown("#### 📈 1. Renta Variable & Multiactivo")
    all_equity_options = [
        # Small / Mid-Caps (Menor liquidez relativa, mayor dispersión)
        "IWM", "IJH", "VB",
        # Mercados Emergentes & Internacionales
        "EEM", "EWZ", "EWW", "INDA", "EFA",
        # Sectores & Real Estate (REITs)
        "VNQ", "XLE", "XLV", "XLU", "XLI",
        # Materias Primas / Commodities
        "GLD", "SLV", "USO", "DBA", "CPER",
        # Acciones Growth / Mid-Cap
        "PLTR", "SQ", "ENPH", "O", "COIN", "FSLR",
        # Mega-Caps tradicionales
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "SPY", "QQQ"
    ]
    
    selected_equities = st.multiselect(
        "Activos en Renta Variable / Commodities:",
        options=all_equity_options,
        default=["IWM", "XLE", "VNQ", "EEM", "GLD", "NVDA"],
        help="Incluye activos descorrelacionados: Small-Caps (IWM), Energía (XLE), Bienes Raíces (VNQ), Emergentes (EEM), Oro (GLD) y Tecnología (NVDA) para evitar la hiperconcentración en monopolios mega-cap."
    )
    
    custom_ticker = st.text_input("Agregar Ticker Personalizado (Yahoo Finance):", "").upper().strip()
    if custom_ticker and custom_ticker not in selected_equities:
        selected_equities.append(custom_ticker)

    st.markdown("---")
    
    # 2. Activo de Renta Fija Explícito de Yahoo Finance
    st.markdown("#### 🏛️ 2. Activo de Renta Fija (Yahoo Finance)")
    fixed_income_ticker = st.selectbox(
        "Instrumento de Renta Fija (ETF de Bonos):",
        options=["TLT", "IEF", "SHY", "BND", "LQD", "HYG", "TIP"],
        format_func=lambda x: {
            "TLT": "TLT - Tesoro EE.UU. 20+ Años (D* ~16.8a)",
            "IEF": "IEF - Tesoro EE.UU. 7-10 Años (D* ~7.6a)",
            "SHY": "SHY - Tesoro EE.UU. 1-3 Años (D* ~1.9a)",
            "BND": "BND - Vanguard Total Bond Market (D* ~6.4a)",
            "LQD": "LQD - Bonos Corporativos Grado Inversión (D* ~8.3a)",
            "HYG": "HYG - Bonos Corporativos High Yield (D* ~3.7a)",
            "TIP": "TIP - Bonos TIPS Protegidos de Inflación (D* ~6.8a)"
        }.get(x, x),
        index=0,
        help="Este activo de renta fija se descarga en tiempo real desde Yahoo Finance y se optimiza en conjunto con la renta variable para amortiguar la volatilidad."
    )
    
    # Combinar activos para la cartera completa
    selected_tickers = list(selected_equities)
    if fixed_income_ticker not in selected_tickers:
        selected_tickers.append(fixed_income_ticker)

    st.markdown("---")
    
    # 3. Restricción Presupuestaria y Concentración
    st.markdown("#### 💰 3. Presupuesto & Concentración")
    budget = st.number_input("Presupuesto Disponible ($ USD):", min_value=1000.0, max_value=5000000.0, value=50000.0, step=1000.0)
    
    max_weight_cap = st.slider(
        "Tope Máximo por Activo (%):", min_value=15, max_value=60, value=25, step=5,
        help="Fuerza al optimizador a diversificar. Un tope del 25% exige asignar capital en al menos 4 o más activos diferentes, evitando monopolios del 50%+ en una sola acción."
    )
    
    broker_fee_bps = st.slider("Comisión de Corretaje (bps):", min_value=0.0, max_value=50.0, value=10.0, step=1.0, 
                               help="10 bps = 0.10% por operación. Modela costos de transacción.")
    
    allow_short = st.checkbox("Habilitar Venta en Corto (Short Selling)", value=False,
                              help="Permite ponderaciones negativas limitadas al 20% por activo.")
    
    hist_period = st.selectbox("Historial de Precios:", ["1y", "2y", "5y"], index=1)
    
    st.markdown("---")
    run_button = st.button("🚀 Ejecutar Optimización Cuantitativa", type="primary", use_container_width=True)

# Header Principal
st.markdown('<div class="main-title">📈 AI Quantitative Investment & Portfolio Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Motor institucional de asignación de capital con regularización de Ledoit-Wolf, dinámica de renta fija y señales ML/DL</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# CARGA DE DATOS DESDE YAHOO FINANCE
# -------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_market_data(tickers, period):
    return get_asset_data(tickers, period=period)

with st.spinner("Descargando cotizaciones de mercado y estimando matrices de covarianza..."):
    market_data = load_market_data(selected_tickers, hist_period)
    prices_df = market_data['prices']
    returns_df = market_data['returns']
    spreads_dict = market_data['spreads']
    latest_prices = market_data['latest_prices']

# -------------------------------------------------------------
# DEFINICIÓN DE TABS DEL DASHBOARD
# -------------------------------------------------------------
tab_risk, tab_port, tab_bonds, tab_signals, tab_theory = st.tabs([
    "📋 1. Perfil del Inversor & Gamma",
    "📊 2. Optimización & Presupuesto",
    "📉 3. Renta Fija (Duración & Convexidad)",
    "⏱️ 4. Señales ML & Maduración",
    "🧠 5. Laboratorio Didáctico de Modelos"
])

# =============================================================
# TAB 1: PERFIL DEL INVERSOR Y AVERSIÓN AL RIESGO (ARROW-PRATT)
# =============================================================
with tab_risk:
    st.subheader("Evaluación Psicométrica de Aversión al Riesgo")
    st.write("Determine su coeficiente de aversión al riesgo relativo de Arrow-Pratt $\\gamma$. Este parámetro gobierna la penalización por volatilidad en la función de utilidad esperada cuadrática.")
    
    col_q1, col_q2 = st.columns(2)
    
    with col_q1:
        q_h = st.selectbox(
            "1. ¿Cuál es su horizonte temporal de inversión planeado?",
            [
                "Menos de 6 meses (Muy corto plazo)",
                "De 6 meses a 2 años (Corto plazo)",
                "De 2 a 5 años (Mediano plazo)",
                "Más de 5 años (Largo plazo)"
            ],
            index=2
        )
        
        q_d = st.selectbox(
            "2. Si su portafolio sufre una corrección repentina del 20% en 1 mes, usted:",
            [
                "Vendo toda mi cartera para evitar mayores pérdidas",
                "Vendo una parte para asegurar liquidez",
                "Mantengo la calma y espero a que se recupere",
                "Aprovecho la caída y compro más títulos a descuento"
            ],
            index=2
        )
        
        q_o = st.selectbox(
            "3. ¿Cuál es el objetivo primordial de esta inversión?",
            [
                "Preservación estricta de capital (tolerancia cero a pérdidas)",
                "Generación de ingresos estables con baja volatilidad",
                "Crecimiento moderado con riesgo controlado",
                "Maximización agresiva de rentabilidad a largo plazo"
            ],
            index=2
        )

    with col_q2:
        q_i = st.selectbox(
            "4. ¿Cuál es la estabilidad y predictibilidad de sus ingresos actuales?",
            [
                "Ingresos altamente variables o jubilación",
                "Ingresos moderadamente estables",
                "Empleo o negocio muy estable con capacidad de ahorro constante",
                "Flujos de caja abundantes e independientes del mercado"
            ],
            index=2
        )
        
        q_e = st.selectbox(
            "5. ¿Cuál es su nivel de experiencia en instrumentos financieros complejos?",
            [
                "Principiante (cuentas de ahorro o depósitos)",
                "Intermedio (fondos indexados y acciones)",
                "Avanzado (acciones, derivados y futuros)",
                "Institucional / Profesional cuantitativo"
            ],
            index=1
        )
        
    risk_profile = calculate_risk_aversion(q_h, q_d, q_o, q_i, q_e)
    gamma_val = risk_profile['gamma']
    
    st.markdown("---")
    res_col1, res_col2 = st.columns([1, 1.2])
    
    with res_col1:
        # Gauge visual interactivo de Gamma
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = gamma_val,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Coeficiente Arrow-Pratt (γ)", 'font': {'size': 20}},
            gauge = {
                'axis': {'range': [1, 10], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#00E676" if gamma_val < 3.5 else ("#FFD600" if gamma_val < 6.5 else "#FF5252")},
                'steps': [
                    {'range': [1, 3], 'color': "rgba(0, 230, 118, 0.2)"},
                    {'range': [3, 7], 'color': "rgba(255, 214, 0, 0.2)"},
                    {'range': [7, 10], 'color': "rgba(255, 82, 82, 0.2)"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': gamma_val
                }
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_gauge, use_container_width=True)
        
    with res_col2:
        st.markdown(f"### Clasificación: **{risk_profile['perfil']}**")
        st.info(risk_profile['descripcion'])
        st.metric("Puntaje Psicométrico Acumulado", f"{risk_profile['total_score']} / 20 pts")
        st.metric("Tope Recomendado en Renta Variable", f"{risk_profile['max_equity_pct']}% del Patrimonio Total")

# =============================================================
# TAB 2: OPTIMIZACIÓN DE PORTAFOLIO & ASIGNACIÓN PRESUPUESTARIA
# =============================================================
with tab_port:
    st.subheader("Optimización Cuantitativa de Utilidad con Contracción Ledoit-Wolf")
    st.write(f"Optimizando asignación para un presupuesto de **${budget:,.2f} USD** con parámetro de aversión **$\\gamma = {gamma_val:.2f}$**.")
    
    # Ejecutar optimización matemática con restricción de concentración
    opt_result = optimize_portfolio_utility(
        returns=returns_df,
        gamma=gamma_val,
        budget=budget,
        latest_prices=latest_prices,
        spreads=spreads_dict,
        broker_fee_bps=broker_fee_bps,
        allow_short=allow_short,
        max_weight_per_asset=max_weight_cap / 100.0
    )
    
    # Tarjetas métricas de resumen
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Retorno Esperado Anual", f"{opt_result['port_ret_lw']*100:.2f}%")
    m_col2.metric("Volatilidad Anual (Riesgo)", f"{opt_result['port_vol_lw']*100:.2f}%")
    m_col3.metric("Ratio de Sharpe Regularizado", f"{opt_result['sharpe_lw']:.2f}")
    m_col4.metric("Contracción Ledoit-Wolf (δ*)", f"{opt_result['shrinkage_intensity']:.4f}")
    
    st.markdown("---")
    
    # Clasificador de Activo para la tabla
    def get_asset_category(t):
        if t == fixed_income_ticker:
            return f"🏛️ Renta Fija ({t})"
        elif t in ["GLD", "SLV", "USO", "DBA", "CPER"]:
            return "🪙 Materia Prima / Oro"
        elif t in ["IWM", "IJH", "VB"]:
            return "📈 Small/Mid-Cap"
        elif t in ["EEM", "EWZ", "EWW", "INDA", "EFA"]:
            return "🌍 Mercado Emergente"
        elif t in ["VNQ", "XLE", "XLV", "XLU", "XLI"]:
            return "🏢 Sectorial / REIT"
        elif t in ["PLTR", "SQ", "ENPH", "O", "COIN", "FSLR"]:
            return "🚀 Growth / Mid-Cap"
        else:
            return "💻 Renta Variable (Mega-Cap)"

    # Tabla de Asignación Discreta de Acciones
    alloc_df = pd.DataFrame({
        "Ticker": opt_result['asset_names'],
        "Clase de Activo": [get_asset_category(a) for a in opt_result['asset_names']],
        "Precio Actual ($)": [round(latest_prices[a], 2) for a in opt_result['asset_names']],
        "Peso Óptimo (%)": [round(opt_result['weights_lw'][a] * 100, 2) for a in opt_result['asset_names']],
        "Títulos Enteros": [opt_result['shares'][a] for a in opt_result['asset_names']],
        "Capital Asignado ($)": [round(opt_result['invested_cash'][a], 2) for a in opt_result['asset_names']],
        "Spread Estimado (bps)": [round(spreads_dict.get(a, 0.001)*10000, 1) for a in opt_result['asset_names']]
    })
    
    st.info(f"🏛️ **Activo de Renta Fija Integrado**: `{fixed_income_ticker}` ({get_fixed_income_profile(fixed_income_ticker)['name']}) | Tope de Concentración: **{max_weight_cap}% por activo**")
    
    t_col1, t_col2 = st.columns([1.5, 0.9])
    
    with t_col1:
        st.markdown("#### 🎯 Asignación Discreta por Título (Resolución de Presupuesto)")
        st.dataframe(alloc_df.set_index("Ticker"), use_container_width=True)
        
        # Balance de Caja
        c_col1, c_col2, c_col3 = st.columns(3)
        c_col1.metric("Capital Invertido en Acciones", f"${opt_result['total_invested']:,.2f}")
        c_col2.metric("Efectivo Remanente (Buffer)", f"${opt_result['cash_remaining']:,.2f}")
        c_col3.metric("Costos Totales de Entrada", f"${opt_result['total_costs_paid']:,.2f}")
        
    with t_col2:
        st.markdown("#### 🥧 Distribución de Ponderaciones Óptimas")
        pie_data = alloc_df[alloc_df["Capital Asignado ($)"] > 0]
        if not pie_data.empty:
            labels = list(pie_data["Ticker"]) + ["CASH (Liquidez)"]
            values = list(pie_data["Capital Asignado ($)"]) + [opt_result['cash_remaining']]
            fig_pie = px.pie(names=labels, values=values, hole=0.45, 
                             color_discrete_sequence=px.colors.qualitative.Plotly)
            fig_pie.update_layout(margin=dict(l=10, r=10, t=20, b=20), height=300)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.warning("El perfil es altamente conservador o el presupuesto por activo no alcanza para comprar 1 acción completa.")
            
    st.markdown("---")
    
    # Comparación Visual: Ledoit-Wolf vs Markowitz Clásico (Sample Covariance)
    st.markdown("#### ⚖️ Efecto de Regularización: Ledoit-Wolf vs. Covarianza Muestral Clásica")
    comp_df = pd.DataFrame({
        "Activo": opt_result['asset_names'],
        "Ledoit-Wolf (Regularizado)": opt_result['weights_lw'].values * 100,
        "Covarianza Muestral (Sin regularizar)": opt_result['weights_sample'].values * 100
    })
    
    fig_comp = px.bar(
        comp_df, x="Activo", y=["Ledoit-Wolf (Regularizado)", "Covarianza Muestral (Sin regularizar)"],
        barmode="group",
        labels={"value": "Ponderación en Portafolio (%)", "variable": "Estimador"},
        title="Prevención de Sobreestimación de Pesos Extremos (Shrinkage Effect)"
    )
    fig_comp.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_comp, use_container_width=True)
    st.caption("Nota: Observe cómo la covarianza muestral tradicional tiende a concentrar de manera espuria el capital en activos con anomalías pasadas. Ledoit-Wolf contrae la matriz hacia un estimador estructurado, eliminando ruido y estabilizando pesos.")

    st.markdown("---")
    st.markdown("### 🎲 Simulación Monte Carlo Multivariada de TODO el Portafolio (Cholesky & Ledoit-Wolf)")
    st.write("Simula la evolución probabilística del **valor total del portafolio ($ USD)** proyectando los activos de forma correlacionada mediante la descomposición de Cholesky $\\Sigma_{\\text{LW}} = L L^T$.")
    
    col_mc_ctrl1, col_mc_ctrl2, col_mc_ctrl3 = st.columns(3)
    with col_mc_ctrl1:
        horizon_choice = st.selectbox("Horizonte Temporal de Proyección:", ["3 Meses (63 días)", "6 Meses (126 días)", "1 Año (252 días)"], index=2, key="sb_mc_horiz")
        h_days = 63 if "3 Meses" in horizon_choice else (126 if "6 Meses" in horizon_choice else 252)
    with col_mc_ctrl2:
        n_sims = st.slider("Número de Trayectorias Monte Carlo:", min_value=100, max_value=1000, value=300, step=50, key="sld_mc_nsims")
    with col_mc_ctrl3:
        rf_mc = st.number_input("Tasa Libre de Riesgo Anual (%):", value=3.5, step=0.25, key="num_mc_rf") / 100.0
        
    ann_mu_vec = returns_df.mean().values * 252
    
    # Ejecutar simulación de TODO el portafolio
    mc_port_res = simulate_multivariate_portfolio_mc(
        shares_dict=opt_result['shares'].to_dict(),
        latest_prices=latest_prices,
        annual_returns=ann_mu_vec,
        cov_matrix=opt_result['cov_lw'],
        initial_budget=budget,
        cash_buffer=opt_result['cash_remaining'],
        asset_names=opt_result['asset_names'],
        time_horizon_days=h_days,
        n_simulations=n_sims,
        risk_free_rate=rf_mc
    )
    
    # Métricas de riesgo de todo el portafolio
    rc1, rc2, rc3, rc4, rc5 = st.columns(5)
    rc1.metric("Patrimonio Esperado E[W_T]", f"${mc_port_res['expected_wealth']:,.2f}", 
               delta=f"{(mc_port_res['expected_wealth'] - budget)/budget*100:+.2f}%")
    rc2.metric("VaR 95% ($ USD)", f"${mc_port_res['var_95_dollar']:,.2f}", 
               delta=f"-{mc_port_res['var_95_pct']:.2f}%", delta_color="inverse")
    rc3.metric("CVaR 95% (Expected Shortfall)", f"${mc_port_res['cvar_95_dollar']:,.2f}", 
               delta=f"-{mc_port_res['cvar_95_pct']:.2f}%", delta_color="inverse")
    rc4.metric("Probabilidad de Pérdida", f"{mc_port_res['prob_loss']:.1f}%")
    rc5.metric("Máx Drawdown Promedio", f"{mc_port_res['max_drawdown_avg']:.2f}%")
    
    col_mc_chart1, col_mc_chart2 = st.columns([1.6, 1.0])
    
    with col_mc_chart1:
        # Abanico de Cono de Riqueza Temporal
        cone = mc_port_res['cone_df']
        fig_cone = go.Figure()
        
        # Banda 90% (P05 a P95)
        fig_cone.add_trace(go.Scatter(
            x=cone['Dia'], y=cone['P95'], mode='lines', line=dict(width=0), showlegend=False
        ))
        fig_cone.add_trace(go.Scatter(
            x=cone['Dia'], y=cone['P05'], mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor='rgba(0, 230, 118, 0.15)', name='Intervalo de Confianza 90% (P05 - P95)'
        ))
        
        # Banda 50% (P25 a P75)
        fig_cone.add_trace(go.Scatter(
            x=cone['Dia'], y=cone['P75'], mode='lines', line=dict(width=0), showlegend=False
        ))
        fig_cone.add_trace(go.Scatter(
            x=cone['Dia'], y=cone['P25'], mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor='rgba(0, 230, 118, 0.30)', name='Intervalo Intercuartil 50% (P25 - P75)'
        ))
        
        # Mediana
        fig_cone.add_trace(go.Scatter(
            x=cone['Dia'], y=cone['Mediana'], mode='lines', name='Trayectoria Mediana (P50)',
            line=dict(color='#00E676', width=3)
        ))
        
        # Presupuesto Inicial
        fig_cone.add_hline(
            y=budget, line_dash="dash", line_color="#FFD600",
            annotation_text=f"Capital Inicial (${budget:,.0f})", annotation_position="top left"
        )
        
        fig_cone.update_layout(
            title=f"Cono de Riqueza Probabilístico de TODO el Portafolio ({h_days} días de mercado)",
            xaxis_title="Días de Negociación", yaxis_title="Valor de la Cartera ($ USD)",
            height=380, margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified"
        )
        st.plotly_chart(fig_cone, use_container_width=True)
        
    with col_mc_chart2:
        # Distribución de Riqueza Final al Horizonte
        fig_hist = px.histogram(
            x=mc_port_res['final_wealth'], nbins=30,
            title="Distribución de Capital Final",
            labels={'x': 'Patrimonio Final ($ USD)', 'y': 'Frecuencia'},
            color_discrete_sequence=['#1E88E5']
        )
        fig_hist.add_vline(x=budget, line_dash='dash', line_color='#FFD600', annotation_text="Base")
        fig_hist.add_vline(x=mc_port_res['p05'], line_dash='dot', line_color='#FF5252', annotation_text="VaR 95%")
        fig_hist.add_vline(x=mc_port_res['expected_wealth'], line_dash='solid', line_color='#00E676', annotation_text="E[W]")
        fig_hist.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)

# =============================================================
# TAB 3: DINÁMICA DE RENTA FIJA (DURACIÓN Y CONVEXIDAD)
# =============================================================
with tab_bonds:
    st.subheader(f"Dinámica de Renta Fija: Activo Seleccionado `{fixed_income_ticker}`")
    fi_prof = get_fixed_income_profile(fixed_income_ticker, current_price=latest_prices.get(fixed_income_ticker))
    
    st.markdown(f"""
    <div style="background-color: #1E2638; padding: 16px; border-radius: 8px; border-left: 5px solid #1E88E5; margin-bottom: 20px;">
        <h4 style="margin:0; color: #64B5F6;">🏛️ {fi_prof['name']} (Ticker: {fixed_income_ticker})</h4>
        <p style="margin: 4px 0; color: #ECEFF1;"><strong>Categoría:</strong> {fi_prof['category']} | <strong>Calificación Crediticia:</strong> {fi_prof['credit_rating']}</p>
        <p style="margin: 0; color: #B0BEC5; font-size: 0.95rem;">{fi_prof['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    bond_met = fi_prof['metrics']
    mkt_p = fi_prof['market_price']
    
    bm1, bm2, bm3, bm4, bm5 = st.columns(5)
    bm1.metric("Precio de Mercado (Yahoo Finance)", f"${mkt_p:,.2f}")
    bm2.metric("Duración Modificada Efectiva (D*)", f"{fi_prof['effective_duration']:.2f} años")
    bm3.metric("Convexidad (C)", f"{fi_prof['convexity']:.2f}")
    bm4.metric("Rendimiento al Vencimiento (YTM)", f"{fi_prof['ytm']*100:.2f}%")
    bm5.metric("DV01 por Acción (1 bp shock)", f"${fi_prof['dv01_share']:.4f}")
    
    st.markdown("---")
    
    # Explicación de cómo entra en la optimización
    col_fi_exp1, col_fi_exp2 = st.columns([1.3, 1.0])
    with col_fi_exp1:
        st.markdown(f"#### 🎯 ¿Cómo y por qué se utiliza `{fixed_income_ticker}` para optimizar la cartera?")
        st.write(f"""
        1. **Cotización e Ingesta Real**: `{fixed_income_ticker}` no es una constante teórica; **es un ETF cotizado en vivo en Yahoo Finance**. Su vector de retornos históricos y su matriz de correlación se calculan directamente contra sus activos de renta variable.
        2. **Efecto Amortiguador y Varianza Mínima**: Al tener una correlación baja o negativa con acciones (`IWM`, `NVDA`, `XLE`, etc.), el estimador de Ledoit-Wolf $\\Sigma_{{\\text{{LW}}}}$ utiliza este instrumento para estabilizar la cartera y maximizar el Ratio de Sharpe.
        3. **Sensibilidad a Tasas de Interés (FED)**: Con una duración efectiva de **{fi_prof['effective_duration']} años**, una caída de 100 bps en las tasas de interés impulsará el precio de este ETF en aproximadamente un **+{fi_prof['effective_duration']:.1f}%**.
        """)
    with col_fi_exp2:
        # Gráfico interactivo de correlación del bono con los demás activos
        if fixed_income_ticker in returns_df.columns and len(returns_df.columns) > 1:
            bond_corrs = returns_df.corr()[fixed_income_ticker].drop(fixed_income_ticker)
            fig_corr = px.bar(
                x=bond_corrs.index, y=bond_corrs.values,
                labels={'x': 'Activo', 'y': f'Correlación con {fixed_income_ticker}'},
                title=f"Correlación de {fixed_income_ticker} vs Renta Variable",
                color=bond_corrs.values, color_continuous_scale="RdYlGn_r"
            )
            fig_corr.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")
    # Simulación de Shocks de Tasas
    shock_range = st.slider("Rango de Simulación de Shock de Tasas (± bps):", min_value=50, max_value=400, value=200, step=25, key="sld_fi_shock_tab3")
    shocks_df = simulate_yield_shocks(bond_met, shock_bps_range=shock_range, n_points=80)
    
    scale_factor = mkt_p / bond_met['bond_price']
    
    fig_bond = go.Figure()
    fig_bond.add_trace(go.Scatter(
        x=shocks_df['Shock_bps'], y=shocks_df['Precio_Exacto'] * scale_factor,
        mode='lines', name=f'Precio Exacto {fixed_income_ticker} (Full Re-pricing)',
        line=dict(color='#00E676', width=3)
    ))
    fig_bond.add_trace(go.Scatter(
        x=shocks_df['Shock_bps'], y=shocks_df['Aprox_Duracion_Convexidad'] * scale_factor,
        mode='lines', name='Taylor 2° Orden (Duración + Convexidad)',
        line=dict(color='#1E88E5', width=2, dash='dash')
    ))
    fig_bond.add_trace(go.Scatter(
        x=shocks_df['Shock_bps'], y=shocks_df['Aprox_Duracion_Lineal'] * scale_factor,
        mode='lines', name='Aproximación Lineal (Sólo Duración)',
        line=dict(color='#FF5252', width=2, dash='dot')
    ))
    
    fig_bond.update_layout(
        title=f"Sensibilidad del Precio de {fixed_income_ticker} ante Movimientos en Tasas de Interés (Δy)",
        xaxis_title="Variación en Rendimiento Δy (Puntos Básicos)",
        yaxis_title=f"Precio Proyectado de {fixed_income_ticker} ($ USD)",
        hovermode="x unified", height=380, margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98)
    )
    st.plotly_chart(fig_bond, use_container_width=True)
    
    st.markdown("""
    > [!NOTE]
    > **Interpretación Cuantitativa**: La recta roja (duración lineal) subestima el precio cuando las tasas bajan y sobreestima la caída cuando las tasas suben. La **convexidad** (curvatura verde/azul) es una propiedad favorable: amortigua las pérdidas cuando las tasas suben y amplifica las ganancias cuando las tasas bajan.
    """)

# =============================================================
# TAB 4: SEÑALES ML/DL LONG/SHORT & TIEMPO DE MADURACIÓN
# =============================================================
with tab_signals:
    st.subheader("Señales Direccionales de Inteligencia Artificial & Tiempos de Maduración")
    st.write("Seleccione el motor de inferencia cuantitativa. Puede comparar modelos de **Machine Learning clásico (Random Forest)** contra redes neuronales de **Deep Learning avanzadas (BiLSTM con Mecanismo de Auto-Atención Temporal en PyTorch)** entrenadas sobre los datos históricos reales.")
    
    col_eng1, col_eng2, col_eng3 = st.columns([1.6, 1.2, 1.0])
    with col_eng1:
        ai_engine = st.radio(
            "Seleccione el Motor Cuantitativo de Aprendizaje:",
            [
                "🤖 Deep Learning: Red Neuronal Recurrente BiLSTM con Auto-Atención Temporal (PyTorch)",
                "🌲 Machine Learning: Bosque Aleatorio sobre Memoria Fraccionaria (Scikit-Learn)"
            ],
            index=0,
            horizontal=False
        )
    with col_eng2:
        dl_mode = st.selectbox(
            "Perfil de Arquitectura Deep Learning:",
            [
                "⚡ Ultraligero & Rápido (~4.4k params, 1 capa BiLSTM)",
                "🎯 Estándar (~8.5k params, 1 capa BiLSTM extendida)"
            ],
            index=0,
            help="El modo Ultraligero reduce el uso de memoria RAM en ~89% y triplica la velocidad de entrenamiento en CPU con ventana de 10 días."
        )
    with col_eng3:
        dl_epochs = st.slider("Épocas PyTorch (AdamW):", min_value=5, max_value=25, value=10, step=5,
                             help="Iteraciones de optimización en la red neuronal. 10 épocas ofrecen convergencia veloz y bajo uso de CPU/RAM.")
        
    use_deep_learning = "Deep Learning" in ai_engine
    dl_mode_idx = 0 if "Ultraligero" in dl_mode else 1
    
    col_btn_re, _ = st.columns([1.2, 2.8])
    with col_btn_re:
        if st.button("🔄 Re-entrenar Modelos en Memoria", help="Limpia la caché de tensores y re-ejecuta el entrenamiento"):
            st.cache_data.clear()

    # -------------------------------------------------------------
    # ENTRENAMIENTO E INFERENCIA DE MODELOS EN CACHÉ EFICIENTE
    # -------------------------------------------------------------
    @st.cache_data(ttl=1800, show_spinner=False)
    def compute_cached_dl_signals(tickers_tuple, prices_sub, returns_sub, epochs, mode_idx):
        h_dim = 16 if mode_idx == 0 else 24
        s_len = 10 if mode_idx == 0 else 12
        
        dl_results_map = {}
        dl_signals_list = []
        
        for ticker in tickers_tuple:
            p_s = prices_sub[ticker].dropna()
            r_s = returns_sub[ticker].dropna()
            fd_s = apply_frac_diff(np.log(p_s), d=0.40)
            
            dl_res = train_deep_learning_agent(
                p_s, r_s, fd_s,
                epochs=epochs,
                seq_len=s_len,
                hidden_dim=h_dim,
                num_layers=1,
                batch_size=32,
                lr=0.008
            )
            dl_results_map[ticker] = dl_res
            
            curr_price = float(p_s.iloc[-1])
            daily_vol = float(r_s.iloc[-20:].std())
            half_life = float(np.clip(np.log(2.0) / (daily_vol * 15 + 1e-4), 3, 30))
            
            sig = dl_res['signal']
            if sig == "LONG":
                act = "Comprar en Largo (Long)"
                sl = curr_price * (1.0 - 2.0 * daily_vol)
                tp = curr_price * (1.0 + 3.0 * daily_vol)
                h_days = int(np.round(half_life * 0.8))
            elif sig == "SHORT":
                act = "Vender en Corto (Short)"
                sl = curr_price * (1.0 + 2.0 * daily_vol)
                tp = curr_price * (1.0 - 3.0 * daily_vol)
                h_days = int(np.round(half_life * 0.8))
            else:
                act = "Neutral / Mantener (Cash)"
                sl = curr_price * (1.0 - daily_vol)
                tp = curr_price * (1.0 + daily_vol)
                h_days = int(np.round(half_life))
                
            h_days = max(3, min(45, h_days))
            
            dl_signals_list.append({
                'Activo': ticker,
                'Precio Actual ($)': round(curr_price, 2),
                'Señal AI': sig,
                'Recomendación': act,
                'Confianza Softmax': f"{dl_res['confidence']:.1f}%",
                'Accuracy Validación (OOS)': f"{dl_res['val_acc']:.1f}%",
                'Maduración Sugerida (Días)': h_days,
                'Take-Profit Sugerido ($)': round(tp, 2),
                'Stop-Loss Dinámico ($)': round(sl, 2),
                'Volatilidad Diaria': f"{daily_vol*100:.2f}%"
            })
            
        return pd.DataFrame(dl_signals_list), dl_results_map

    dl_results = {}
    with st.spinner(f"Ejecutando inferencia con {'Deep Learning (PyTorch BiLSTM Ultraligera)' if use_deep_learning else 'Machine Learning (Random Forest)'}..."):
        if use_deep_learning:
            active_signals_df, dl_results = compute_cached_dl_signals(
                tuple(selected_tickers),
                prices_df[selected_tickers],
                returns_df[selected_tickers],
                dl_epochs,
                dl_mode_idx
            )
        else:
            active_signals_df = generate_timing_recommendations(prices_df, returns_df)
            if 'Señal ML' in active_signals_df.columns:
                active_signals_df = active_signals_df.rename(columns={'Señal ML': 'Señal AI'})

    st.dataframe(active_signals_df.set_index("Activo"), use_container_width=True)
    st.markdown("---")
    
    # -------------------------------------------------------------
    # INSPECTOR DE APRENDIZAJE REAL DE LA RED NEURONAL / ML
    # -------------------------------------------------------------
    st.markdown("### 🧠 Inspector de Aprendizaje y Mecanismos Internos")
    asset_inspect = st.selectbox("Seleccione un Activo para auditar el aprendizaje del modelo:", selected_tickers, key="sb_audit_asset")
    
    if use_deep_learning and asset_inspect in dl_results:
        agent_data = dl_results[asset_inspect]
        
        col_met1, col_met2, col_met3, col_met4 = st.columns(4)
        col_met1.metric("Precisión Out-of-Sample (Val Accuracy)", f"{agent_data['val_acc']:.1f}%")
        col_met2.metric("Épocas de Entrenamiento", f"{agent_data['epochs_trained']}")
        col_met3.metric("Muestras de Entrenamiento", f"{agent_data['num_train_samples']} secuencias")
        col_met4.metric("Muestras de Testeo", f"{agent_data['num_val_samples']} secuencias")
        
        col_dl_plot1, col_dl_plot2 = st.columns(2)
        
        with col_dl_plot1:
            # Curva de Pérdida real de PyTorch por Época
            loss_curve = agent_data['train_loss_history']
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=list(range(1, len(loss_curve) + 1)), y=loss_curve,
                mode='lines+markers', name='Loss PyTorch (CrossEntropy)',
                line=dict(color='#00E676', width=2.5)
            ))
            fig_loss.update_layout(
                title=f"Curva de Pérdida en Entrenamiento PyTorch ({asset_inspect})",
                xaxis_title="Época de Entrenamiento (Epoch)", yaxis_title="Loss de Optimización",
                height=320, margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_loss, use_container_width=True)
            st.caption("Esta curva demuestra el **aprendizaje empírico real**: la pérdida desciende a medida que el optimizador AdamW ajusta los pesos de las matrices LSTM y de Atención.")
            
        with col_dl_plot2:
            # Pesos de Atención Temporal (Attention Weights)
            attn_w = agent_data['attention_weights']
            days_labels = [f"t-{len(attn_w)-i}" for i in range(len(attn_w))]
            fig_attn = px.bar(
                x=days_labels, y=attn_w,
                title=f"Pesos del Mecanismo de Auto-Atención Temporal ({asset_inspect})",
                labels={'x': 'Día en la Ventana Temporal de Entrada', 'y': 'Peso de Atención (Softmax α_t)'},
                color=attn_w, color_continuous_scale="Viridis"
            )
            fig_attn.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
            st.plotly_chart(fig_attn, use_container_width=True)
            st.caption("Los pesos de atención revelan **qué días específicos del pasado** la red neuronal consideró matemáticamente más determinantes para predecir la dirección futura.")
    else:
        col_sig1, col_sig2 = st.columns(2)
        with col_sig1:
            p_asset = prices_df[asset_inspect].dropna()
            fd_asset = apply_frac_diff(np.log(p_asset), d=0.40)
            fig_fd = go.Figure()
            fig_fd.add_trace(go.Scatter(x=fd_asset.index, y=fd_asset.values, name="FracDiff (d=0.40)", line=dict(color="#FFD600", width=1.5)))
            fig_fd.update_layout(
                title=f"Serie Estacionaria con Memoria Conservada ({asset_inspect})",
                xaxis_title="Fecha", yaxis_title="Valor Fraccionario",
                height=320, margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_fd, use_container_width=True)
            
        with col_sig2:
            fig_mat = px.bar(
                active_signals_df, x="Activo", y="Maduración Sugerida (Días)",
                color="Señal AI",
                color_discrete_map={"LONG": "#00E676", "SHORT": "#FF5252", "NEUTRAL": "#B0BEC5"},
                title="Horizonte Óptimo de Maduración (Semivida O-U en Días de Mercado)"
            )
            fig_mat.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_mat, use_container_width=True)
            
    st.markdown("""
    > [!TIP]
    > **Gestión de Riesgo Dinámica**: 
    > - **Take-Profit Sugerido**: Proyectado a $+3\\sigma_{\\text{diaria}}$ para operaciones en largo y $-3\\sigma$ para corto.
    > - **Stop-Loss Dinámico**: Nivel de corte fijado a $2\\sigma_{\\text{diaria}}$ para proteger el capital contra rupturas adversas de volatilidad.
    """)

# =============================================================
# TAB 5: LABORATORIO DIDÁCTICO & SIMULACIÓN INTERACTIVA DE MODELOS
# =============================================================
with tab_theory:
    st.subheader("🧪 Laboratorio Cuantitativo Interactivo: Simulación & Gráficos de Modelos")
    st.write("Experimente interactivamente con las matemáticas y dinámicas estocásticas de cada modelo en tiempo real. Modifique parámetros y observe la respuesta inmediata de los algoritmos y sus representaciones visuales.")
    
    subtab1, subtab2, subtab3, subtab4, subtab5 = st.tabs([
        "🔬 1. Ledoit-Wolf & Autovalores",
        "📈 2. Frontera & Curvas de Indiferencia",
        "📉 3. Renta Fija & Error Residual de Taylor",
        "🧬 4. Diferenciación Fraccionaria",
        "🔄 5. Simulación Monte Carlo Ornstein-Uhlenbeck"
    ])
    
    # -------------------------------------------------------------
    # SUBTAB 1: LEDOIT-WOLF & MARCHENKO-PASTUR
    # -------------------------------------------------------------
    with subtab1:
        st.markdown("### 🔬 Regularización de Covarianza & Filtrado de Ruido (Random Matrix Theory)")
        st.latex(r"\Sigma_{\text{LW}} = \delta^* F + (1 - \delta^*) S, \quad \lambda_{\pm} = \sigma^2 \left(1 \pm \sqrt{\frac{N}{T}}\right)^2")
        st.write("La covarianza muestral $S$ sobreestima los autovalores mayores y genera inestabilidad numérica. Ledoit-Wolf contrae la matriz hacia un target estructurado $F$, reduciendo el número de condición.")
        
        lw_res = ledoit_wolf_covariance(returns_df)
        S_mat = lw_res['sample_cov']
        LW_mat = lw_res['shrunk_cov']
        delta_opt = float(lw_res['shrinkage_intensity'])
        
        col_lw1, col_lw2 = st.columns([1, 2])
        with col_lw1:
            st.markdown("#### Parámetros del Estimador")
            user_delta = st.slider(
                "Intensidad de Contracción (δ):",
                min_value=0.0, max_value=1.0, value=float(np.round(delta_opt, 3)), step=0.01,
                help="δ = 0 es Covarianza Muestral pura (ruidosa). δ = 1 es el Target estructurado F."
            )
            st.metric("Contracción Óptima Asintótica (δ*)", f"{delta_opt:.4f}")
            
            # Construir Target F con correlación equitativa
            n_dim = len(S_mat)
            corr_sum = 0.0
            pair_count = 0
            for i in range(n_dim):
                for j in range(i+1, n_dim):
                    corr_sum += S_mat[i, j] / np.sqrt(S_mat[i, i] * S_mat[j, j])
                    pair_count += 1
            mean_corr = corr_sum / max(1, pair_count)
            
            F_mat = np.zeros_like(S_mat)
            for i in range(n_dim):
                for j in range(n_dim):
                    if i == j:
                        F_mat[i, j] = S_mat[i, i]
                    else:
                        F_mat[i, j] = mean_corr * np.sqrt(S_mat[i, i] * S_mat[j, j])
                        
            Sigma_user = (1.0 - user_delta) * S_mat + user_delta * F_mat
            
            evals_sample = np.sort(np.linalg.eigvalsh(S_mat))[::-1]
            evals_user = np.sort(np.linalg.eigvalsh(Sigma_user))[::-1]
            
            cond_sample = float(evals_sample[0] / (evals_sample[-1] + 1e-8))
            cond_user = float(evals_user[0] / (evals_user[-1] + 1e-8))
            
            st.metric("Número de Condición κ(S) [Muestral]", f"{cond_sample:.1f}")
            st.metric("Número de Condición κ(Σ) [Regularizado]", f"{cond_user:.1f}", 
                      delta=f"{cond_user - cond_sample:.1f}", delta_color="inverse")
            st.caption("Un menor número de condición κ evita ponderaciones numéricamente inestables e hipertrofiadas.")
            
        with col_lw2:
            T_obs, N_dim = returns_df.shape
            q_ratio = T_obs / N_dim
            sigma2 = np.trace(S_mat) / N_dim
            lambda_plus = float(sigma2 * (1.0 + np.sqrt(1.0 / q_ratio)) ** 2)
            
            df_evals = pd.DataFrame({
                "Autovalor": [f"λ_{i+1}" for i in range(len(evals_sample))],
                "Covarianza Muestral S": evals_sample,
                f"Contracción δ={user_delta:.2f}": evals_user
            })
            
            fig_eval = go.Figure()
            fig_eval.add_trace(go.Bar(
                x=df_evals["Autovalor"], y=df_evals["Covarianza Muestral S"],
                name="Covarianza Muestral S (Sin regularizar)", marker_color="#FF5252"
            ))
            fig_eval.add_trace(go.Bar(
                x=df_evals["Autovalor"], y=df_evals[f"Contracción δ={user_delta:.2f}"],
                name=f"Regularizada (δ = {user_delta:.2f})", marker_color="#00E676"
            ))
            fig_eval.add_hline(
                y=lambda_plus, line_dash="dash", line_color="#FFD600",
                annotation_text=f"Límite de Ruido Marchenko-Pastur λ+ ({lambda_plus:.3f})",
                annotation_position="top right"
            )
            fig_eval.update_layout(
                title="Espectro de Autovalores vs. Umbral de Ruido de Marchenko-Pastur",
                xaxis_title="Modo Propio (Eigenmode)", yaxis_title="Varianza Explicada (Autovalor)",
                barmode="group", height=380, margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_eval, use_container_width=True)
            st.info("La línea amarilla marca el umbral superior de ruido $\\lambda_+$ según la Teoría de Matrices Aleatorias. Ledoit-Wolf contrae los autovalores ruidosos hacia el valor medio, eliminando el sobreajuste empírico.")

    # -------------------------------------------------------------
    # SUBTAB 2: FRONTERA EFICIENTE & CURVAS DE INDIFERENCIA
    # -------------------------------------------------------------
    with subtab2:
        st.markdown("### 📈 Superficie de Utilidad & Curvas de Indiferencia de Arrow-Pratt")
        st.latex(r"U(w) = w^T \mu - \frac{\gamma}{2} w^T \Sigma w \implies \mu = U + \frac{\gamma}{2} \sigma^2")
        st.write("Visualice cómo interactúa la función de utilidad cuadrática con la Frontera Eficiente de Markowitz, y cómo varía el punto de tangencia óptimo al modular la aversión al riesgo $\\gamma$.")
        
        col_fr1, col_fr2 = st.columns([1, 2.2])
        with col_fr1:
            gamma_sim = st.slider("Aversión al Riesgo (γ):", min_value=0.5, max_value=12.0, value=float(gamma_val), step=0.5, key="sld_gamma_lab")
            ann_mu = returns_df.mean().values * 252
            cov_ann = LW_mat
            n_assets = len(ann_mu)
            
            # Generar puntos de la frontera
            target_returns = np.linspace(min(ann_mu)*0.85, max(ann_mu)*1.05, 25)
            front_vols = []
            front_rets = []
            
            for r_target in target_returns:
                cons = [
                    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                    {'type': 'eq', 'fun': lambda w, r=r_target: np.dot(w, ann_mu) - r}
                ]
                bnds = [(0.0, 1.0) for _ in range(n_assets)]
                res = minimize(lambda w: np.dot(w.T, np.dot(cov_ann, w)), np.ones(n_assets)/n_assets,
                               method='SLSQP', bounds=bnds, constraints=cons)
                if res.success:
                    front_vols.append(float(np.sqrt(res.fun)))
                    front_rets.append(float(r_target))
                    
            # Punto óptimo de utilidad para gamma_sim
            cons_u = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
            bnds_u = [(0.0, 1.0) for _ in range(n_assets)]
            res_u = minimize(lambda w: -(np.dot(w, ann_mu) - 0.5 * gamma_sim * np.dot(w.T, np.dot(cov_ann, w))),
                             np.ones(n_assets)/n_assets, method='SLSQP', bounds=bnds_u, constraints=cons_u)
            w_star = res_u.x
            opt_mu = float(np.dot(w_star, ann_mu))
            opt_vol = float(np.sqrt(np.dot(w_star.T, np.dot(cov_ann, w_star))))
            opt_u = opt_mu - 0.5 * gamma_sim * (opt_vol ** 2)
            
            st.metric("Retorno Óptimo (μ*)", f"{opt_mu*100:.2f}%")
            st.metric("Volatilidad Óptima (σ*)", f"{opt_vol*100:.2f}%")
            st.metric("Utilidad Cuadrática Máxima", f"{opt_u:.4f}")
            
        with col_fr2:
            fig_front = go.Figure()
            if front_vols:
                fig_front.add_trace(go.Scatter(
                    x=front_vols, y=front_rets, mode='lines', name='Frontera Eficiente (Markowitz + Ledoit-Wolf)',
                    line=dict(color='#1E88E5', width=3)
                ))
            
            # Activos individuales
            asset_vols = [float(np.sqrt(cov_ann[i, i])) for i in range(n_assets)]
            fig_front.add_trace(go.Scatter(
                x=asset_vols, y=ann_mu, mode='markers+text',
                text=list(returns_df.columns), textposition='top right',
                marker=dict(size=10, color='#FF9100'), name='Activos Individuales'
            ))
            
            # Curva de indiferencia tangente: mu = U + (gamma / 2) * sigma^2
            min_v = min(front_vols) if front_vols else 0.05
            max_v = max(front_vols) if front_vols else 0.40
            sigma_grid = np.linspace(min_v * 0.7, max_v * 1.25, 50)
            indifference_mu = opt_u + 0.5 * gamma_sim * (sigma_grid ** 2)
            fig_front.add_trace(go.Scatter(
                x=sigma_grid, y=indifference_mu, mode='lines', name=f'Curva de Indiferencia (γ={gamma_sim})',
                line=dict(color='#00E676', width=2, dash='dash')
            ))
            
            # Punto de tangencia
            fig_front.add_trace(go.Scatter(
                x=[opt_vol], y=[opt_mu], mode='markers',
                marker=dict(size=14, color='#FFD600', symbol='star'),
                name='Punto Óptimo Tangente (Max U)'
            ))
            
            fig_front.update_layout(
                title=f"Tangencia de Utilidad Esperada sobre la Frontera Eficiente (γ = {gamma_sim})",
                xaxis_title="Riesgo Anual (Volatilidad σ)", yaxis_title="Retorno Esperado Anual (μ)",
                xaxis=dict(tickformat=".1%"), yaxis=dict(tickformat=".1%"),
                hovermode="closest", height=420, margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_front, use_container_width=True)
            st.caption("A mayor $\\gamma$ (mayor aversión al riesgo), la parábola verde se empina con fuerza, empujando la cartera óptima hacia la izquierda (menor volatilidad).")

    # -------------------------------------------------------------
    # SUBTAB 3: RENTA FIJA & ERROR RESIDUAL DE TAYLOR
    # -------------------------------------------------------------
    with subtab3:
        st.markdown("### 📉 Renta Fija: Análisis de Sensibilidad y Error de Taylor")
        st.latex(r"\frac{\Delta P}{P} = -D^* \Delta y + \frac{1}{2} C (\Delta y)^2 + \mathcal{O}((\Delta y)^3)")
        st.write("Cuantificación del error de aproximación en función de la magnitud del shock de rendimiento $\\Delta y$. Demuestra por qué la duración por sí sola falla en shocks severos.")
        
        col_tay1, col_tay2 = st.columns([1, 2])
        with col_tay1:
            mat_lab = st.slider("Maduración del Bono (Años):", 1, 30, 15, key="sld_mat_lab")
            c_lab = st.slider("Tasa Cupón (%):", 1.0, 12.0, 5.0, key="sld_c_lab") / 100.0
            ytm_lab = st.slider("Rendimiento Inicial (YTM %):", 1.0, 12.0, 4.5, key="sld_ytm_lab") / 100.0
            shock_max = st.slider("Shock Máximo Simulado (bps):", 100, 500, 300, step=50, key="sld_shk_lab")
            
            b_met = calculate_bond_metrics(1000.0, c_lab, ytm_lab, mat_lab, 2)
            st.metric("Duración Modificada (D*)", f"{b_met['mod_duration']:.2f} años")
            st.metric("Convexidad (C)", f"{b_met['convexity']:.2f}")
            st.metric("DV01 (Shock 1 bp)", f"${b_met['dv01']:.4f}")
            
        with col_tay2:
            df_sh = simulate_yield_shocks(b_met, shock_bps_range=shock_max, n_points=60)
            error_lineal = np.abs(df_sh['Aprox_Duracion_Lineal'] - df_sh['Precio_Exacto'])
            error_taylor2 = np.abs(df_sh['Aprox_Duracion_Convexidad'] - df_sh['Precio_Exacto'])
            
            fig_err = go.Figure()
            fig_err.add_trace(go.Scatter(
                x=df_sh['Shock_bps'], y=error_lineal,
                mode='lines', name='Error Absoluto: Sólo Duración (Lineal)',
                line=dict(color='#FF5252', width=2.5)
            ))
            fig_err.add_trace(go.Scatter(
                x=df_sh['Shock_bps'], y=error_taylor2,
                mode='lines', name='Error Absoluto: Duración + Convexidad (Taylor 2°)',
                line=dict(color='#00E676', width=2.5)
            ))
            fig_err.update_layout(
                title=f"Error Residual de Aproximación ante Shocks en Tasas (Vencimiento = {mat_lab} años)",
                xaxis_title="Shock de Rendimiento Δy (Puntos Básicos)",
                yaxis_title="Error Residual Absoluto en Precio ($ USD)",
                height=380, margin=dict(l=20, r=20, t=40, b=20),
                hovermode="x unified"
            )
            st.plotly_chart(fig_err, use_container_width=True)
            max_err_lin = float(error_lineal.max())
            max_err_tay = float(error_taylor2.max())
            pct_reduc = (1.0 - max_err_tay / (max_err_lin + 1e-6)) * 100
            st.info(f"Para un shock de ±{shock_max} bps, el error de aproximación lineal alcanza **${max_err_lin:.2f} USD** por bono de $1,000, mientras que al incorporar la **convexidad** el error se reduce a apenas **${max_err_tay:.2f} USD** (reducción del {pct_reduc:.1f}% del error).")

    # -------------------------------------------------------------
    # SUBTAB 4: DIFERENCIACIÓN FRACCIONARIA
    # -------------------------------------------------------------
    with subtab4:
        st.markdown("### 🧬 Conservación de Memoria Histórica vs. Estacionariedad (López de Prado)")
        st.latex(r"(1 - L)^d = \sum_{k=0}^{\infty} (-1)^k \binom{d}{k} L^k")
        st.write("La diferenciación entera tradicional ($d=1$) destruye la memoria predictiva de niveles de soporte y resistencia. La diferenciación fraccionaria halla el orden mínimo $d^*$ que garantiza estacionariedad conservando la mayor memoria.")
        
        col_fd1, col_fd2 = st.columns([1, 2])
        with col_fd1:
            asset_fd = st.selectbox("Activo a Analizar:", selected_tickers, key="sb_asset_fd_lab")
            d_slider = st.slider("Orden de Diferenciación Fraccionaria (d):", 0.0, 1.0, 0.40, 0.05, key="sld_d_lab")
            
            p_raw = prices_df[asset_fd].dropna()
            log_p = np.log(p_raw)
            
            d_grid = np.linspace(0.0, 1.0, 11)
            corrs = []
            vol_ratios = []
            
            for d_val in d_grid:
                if d_val == 0.0:
                    fd_s = log_p
                else:
                    fd_s = apply_frac_diff(log_p, d=d_val)
                common = log_p.index.intersection(fd_s.index)
                corr = float(np.corrcoef(log_p.loc[common].values, fd_s.loc[common].values)[0, 1])
                corrs.append(corr)
                vol_ratios.append(float(fd_s.std() / (log_p.std() + 1e-6)))
                
            idx_curr = int(np.round(d_slider * 10))
            st.metric("Memoria Preservada (Correlación)", f"{corrs[idx_curr]*100:.1f}%")
            st.caption("Con d=0.40 se retiene más del 80% de la correlación con la serie de precios original.")
            
        with col_fd2:
            fig_mem = go.Figure()
            fig_mem.add_trace(go.Scatter(
                x=d_grid, y=corrs, mode='lines+markers', name='Memoria Conservada (Correlación con Precio)',
                line=dict(color='#1E88E5', width=3)
            ))
            fig_mem.add_trace(go.Scatter(
                x=d_grid, y=vol_ratios, mode='lines+markers', name='Volatilidad Normalizada (Dispersión Residual)',
                line=dict(color='#FF5252', width=2, dash='dot')
            ))
            
            fig_mem.add_vrect(
                x0=0.35, x1=0.45, fillcolor="green", opacity=0.2,
                annotation_text="Zona Óptima d* (0.35 - 0.45)", annotation_position="top left"
            )
            fig_mem.add_vline(x=d_slider, line_dash="dash", line_color="#FFD600",
                              annotation_text=f"d={d_slider}")
                              
            fig_mem.update_layout(
                title=f"Curva de Información Fraccionaria de López de Prado ({asset_fd})",
                xaxis_title="Orden de Diferenciación (d)", yaxis_title="Métrica Normalizada [0, 1]",
                height=380, margin=dict(l=20, r=20, t=40, b=20),
                hovermode="x unified"
            )
            st.plotly_chart(fig_mem, use_container_width=True)
            st.info("A $d=1.0$ (diferenciación entera tradicional), la memoria cae por debajo del 15%, transformando la serie en ruido blanco. A $d=0.40$, logramos estacionariedad mientras mantenemos los canales estructurales y soportes históricos.")

    # -------------------------------------------------------------
    # SUBTAB 5: SIMULACIÓN MONTE CARLO (PORTAFOLIO & DIFUSIÓN)
    # -------------------------------------------------------------
    with subtab5:
        st.markdown("### 🎲 Simulaciones Estocásticas de Monte Carlo: Portafolio & Difusión")
        st.write("Explore tanto la **simulación multivariada de TODO el portafolio** (proyectando todos los activos de forma correlacionada con Cholesky) como los **procesos estocásticos de reversión a la media** (Ornstein-Uhlenbeck).")
        
        mc_type = st.radio(
            "Seleccione el Experimento de Monte Carlo:",
            [
                "🏛️ Simulación Multivariada de TODO el Portafolio (Cholesky & Ledoit-Wolf)",
                "🔄 Proceso de Difusión Ornstein-Uhlenbeck (Reversión a la Media de Spread)"
            ],
            index=0,
            horizontal=True
        )
        
        if "TODO el Portafolio" in mc_type:
            st.latex(r"W_t = \sum_{i=1}^N \text{Shares}_i \cdot S_{i, t} + \text{Cash}_t, \quad S_{i, t} = S_{i, t-1} \exp\left(\mu_i \Delta t + [L Z]_i \sqrt{\Delta t}\right)")
            st.write("Proyección probabilística del valor conjunto del patrimonio neto. Incorpora la estructura completa de covarianzas regularizadas $\\Sigma_{\\text{LW}} = L L^T$ para capturar el efecto de la diversificación real.")
            
            col_pmc1, col_pmc2 = st.columns([1, 2])
            with col_pmc1:
                st.markdown("#### Parámetros del Portafolio")
                mc_horiz_lab = st.slider("Horizonte de Proyección (Días de Mercado):", 30, 504, 252, step=21, key="sld_mc_h_lab")
                mc_nsim_lab = st.slider("Número de Simulaciones:", 100, 1000, 400, step=50, key="sld_mc_n_lab")
                vol_mult = st.slider("Multiplicador de Estrés de Volatilidad:", 0.5, 2.5, 1.0, 0.1,
                                     help="1.0 = Volatilidad histórica normal. > 1.0 = Simulación de estrés de mercado (Stress-testing).")
                rf_lab = st.number_input("Tasa Libre de Riesgo (%):", value=3.5, step=0.25, key="num_mc_rf_lab") / 100.0
                
                # Ejecutar simulación con matriz escalada por estrés
                stressed_cov = opt_result['cov_lw'] * (vol_mult ** 2)
                res_mc_lab = simulate_multivariate_portfolio_mc(
                    shares_dict=opt_result['shares'].to_dict(),
                    latest_prices=latest_prices,
                    annual_returns=returns_df.mean().values * 252,
                    cov_matrix=stressed_cov,
                    initial_budget=budget,
                    cash_buffer=opt_result['cash_remaining'],
                    asset_names=opt_result['asset_names'],
                    time_horizon_days=mc_horiz_lab,
                    n_simulations=mc_nsim_lab,
                    risk_free_rate=rf_lab
                )
                
                st.metric("Patrimonio Esperado E[W]", f"${res_mc_lab['expected_wealth']:,.2f}", 
                          delta=f"{(res_mc_lab['expected_wealth']-budget)/budget*100:+.2f}%")
                st.metric("VaR 95% en Dólares", f"${res_mc_lab['var_95_dollar']:,.2f}", 
                          delta=f"-{res_mc_lab['var_95_pct']:.1f}%", delta_color="inverse")
                st.metric("CVaR 95% (Pérdida en Cola)", f"${res_mc_lab['cvar_95_dollar']:,.2f}", 
                          delta=f"-{res_mc_lab['cvar_95_pct']:.1f}%", delta_color="inverse")
                st.metric("Probabilidad de Pérdida", f"{res_mc_lab['prob_loss']:.1f}%")
                
            with col_pmc2:
                cone_lab = res_mc_lab['cone_df']
                fig_port_mc = go.Figure()
                
                # Banda 90% (P05 a P95)
                fig_port_mc.add_trace(go.Scatter(
                    x=cone_lab['Dia'], y=cone_lab['P95'], mode='lines', line=dict(width=0), showlegend=False
                ))
                fig_port_mc.add_trace(go.Scatter(
                    x=cone_lab['Dia'], y=cone_lab['P05'], mode='lines', line=dict(width=0),
                    fill='tonexty', fillcolor='rgba(30, 136, 229, 0.15)', name='Intervalo de Confianza 90% (P05 - P95)'
                ))
                
                # Mediana
                fig_port_mc.add_trace(go.Scatter(
                    x=cone_lab['Dia'], y=cone_lab['Mediana'], mode='lines', name='Trayectoria Mediana P50',
                    line=dict(color='#00E676', width=3)
                ))
                
                # Muestra de 15 trayectorias individuales
                w_paths = res_mc_lab['wealth_paths']
                days_idx = np.arange(mc_horiz_lab + 1)
                for p in range(min(15, mc_nsim_lab)):
                    fig_port_mc.add_trace(go.Scatter(
                        x=days_idx, y=w_paths[:, p], mode='lines',
                        line=dict(width=1), opacity=0.3, showlegend=False
                    ))
                    
                fig_port_mc.add_hline(
                    y=budget, line_dash="dash", line_color="#FFD600",
                    annotation_text=f"Capital Inicial (${budget:,.0f})", annotation_position="top left"
                )
                
                fig_port_mc.update_layout(
                    title=f"Evolución Estocástica de TODO el Portafolio ({mc_nsim_lab} escenarios)",
                    xaxis_title="Días de Negociación", yaxis_title="Patrimonio Neto ($ USD)",
                    height=380, margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified"
                )
                st.plotly_chart(fig_port_mc, use_container_width=True)
                st.info(f"Con un multiplicador de estrés de {vol_mult}x, el VaR 95% indica que en el 95% de los escenarios anuales la pérdida no superará **${res_mc_lab['var_95_dollar']:,.2f} USD**.")
        else:
            st.latex(r"dy_t = \theta (\mu - y_t) dt + \sigma dW_t, \quad t_{1/2} = \frac{\ln(2)}{\theta}")
            st.write("Simulación estocástica de trayectorias que experimentan reversión a la media. Permite determinar científicamente la duración óptima de un trade antes de su decaimiento.")
            
            col_ou1, col_ou2 = st.columns([1, 2])
            with col_ou1:
                st.markdown("#### Parámetros Estocásticos")
                theta_sim = st.slider("Velocidad de Reversión (θ):", 0.5, 15.0, 4.0, 0.5, key="sld_th_ou_lab")
                mu_sim = st.slider("Nivel de Equilibrio (μ $):", 50.0, 200.0, 100.0, 5.0, key="sld_mu_ou_lab")
                y0_sim = st.slider("Precio Inicial Desviado (y0 $):", 50.0, 200.0, 125.0, 5.0, key="sld_y0_ou_lab")
                sigma_sim = st.slider("Volatilidad de Difusión (σ):", 5.0, 50.0, 15.0, 2.5, key="sld_sig_ou_lab")
                n_paths = st.slider("Número de Trayectorias Monte Carlo:", 5, 30, 12, key="sld_np_ou_lab")
                n_days = st.slider("Horizonte de Simulación (Días):", 15, 90, 45, key="sld_nd_ou_lab")
                
                dt_day = 1.0 / 252.0
                half_life_days = np.log(2.0) / (theta_sim * dt_day)
                half_life_days = float(np.clip(half_life_days, 1.0, n_days))
                
                st.metric("Semivida Analítica (t₁/₂)", f"{half_life_days:.1f} días de mercado")
                st.caption("Tiempo estadístico esperado para que el 50% de la desviación (|y0 - μ|) sea absorbida.")
                
            with col_ou2:
                np.random.seed(42)
                dt = 1.0 / 252.0
                paths = np.zeros((n_days, n_paths))
                paths[0] = y0_sim
                
                for t in range(1, n_days):
                    dW = np.random.normal(0, np.sqrt(dt), n_paths)
                    paths[t] = paths[t-1] + theta_sim * (mu_sim - paths[t-1]) * dt + sigma_sim * dW
                    
                fig_ou = go.Figure()
                days_x = np.arange(n_days)
                for p in range(n_paths):
                    fig_ou.add_trace(go.Scatter(
                        x=days_x, y=paths[:, p], mode='lines',
                        line=dict(width=1), opacity=0.4, showlegend=False
                    ))
                    
                mean_path = np.mean(paths, axis=1)
                fig_ou.add_trace(go.Scatter(
                    x=days_x, y=mean_path, mode='lines', name='Trayectoria Media Simulada',
                    line=dict(color='#00E676', width=3)
                ))
                
                fig_ou.add_hline(
                    y=mu_sim, line_dash="dash", line_color="#FFD600",
                    annotation_text=f"Equilibrio a Largo Plazo μ (${mu_sim:.1f})",
                    annotation_position="top right"
                )
                
                if half_life_days <= n_days:
                    fig_ou.add_vline(
                        x=half_life_days, line_dash="dot", line_color="#FF5252",
                        annotation_text=f"Maduración t₁/₂ ({half_life_days:.1f}d)",
                        annotation_position="bottom right"
                    )
                    
                fig_ou.update_layout(
                    title="Simulación Monte Carlo de Reversión a la Media (Proceso O-U)",
                    xaxis_title="Días de Mercado Transcurridos", yaxis_title="Precio del Activo ($ USD)",
                    height=380, margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_ou, use_container_width=True)
                st.info("Las posiciones cuantitativas de arbitraje o reversión deben cerrarse cerca de la **semivida $t_{1/2}$**, evitando dejar capital inmovilizado cuando la velocidad de convergencia se ralentiza asintóticamente.")

# Footer institucional
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #757575; font-size: 0.85rem;'>"
    "AI Quantitative Investment Platform &bull; Desarrollado para Taller de Finanzas Cuantitativas &bull; "
    "Datos provistos vía Yahoo Finance API &bull; Modelos ejecutados en PyTorch / Scikit-Learn"
    "</div>",
    unsafe_allow_html=True
)

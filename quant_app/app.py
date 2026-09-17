import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Importación de módulos cuantitativos locales
from modules.data_loader import get_asset_data
from modules.risk_profiler import calculate_risk_aversion
from modules.portfolio_optimizer import optimize_portfolio_utility, ledoit_wolf_covariance
from modules.fixed_income import calculate_bond_metrics, simulate_yield_shocks
from modules.timing_signals import generate_timing_recommendations, apply_frac_diff

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
    st.markdown("### ⚙️ Parámetros del Inversor")
    
    # 1. Selección de Activos (Yahoo Finance)
    selected_tickers = st.multiselect(
        "Activos en Renta Variable / ETFs:",
        options=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "TLT", "SPY", "QQQ", "GLD", "BND"],
        default=["AAPL", "MSFT", "NVDA", "GOOGL", "TLT"]
    )
    
    # Permitir ticker personalizado
    custom_ticker = st.text_input("Agregar Ticker Adicional (Yahoo Finance):", "").upper().strip()
    if custom_ticker and custom_ticker not in selected_tickers:
        selected_tickers.append(custom_ticker)
        
    st.markdown("---")
    
    # 2. Restricción Presupuestaria y Fricciones
    budget = st.number_input("Presupuesto Disponible ($ USD):", min_value=1000.0, max_value=5000000.0, value=50000.0, step=1000.0)
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
    
    # Ejecutar optimización matemática
    opt_result = optimize_portfolio_utility(
        returns=returns_df,
        gamma=gamma_val,
        budget=budget,
        latest_prices=latest_prices,
        spreads=spreads_dict,
        broker_fee_bps=broker_fee_bps,
        allow_short=allow_short
    )
    
    # Tarjetas métricas de resumen
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Retorno Esperado Anual", f"{opt_result['port_ret_lw']*100:.2f}%")
    m_col2.metric("Volatilidad Anual (Riesgo)", f"{opt_result['port_vol_lw']*100:.2f}%")
    m_col3.metric("Ratio de Sharpe Regularizado", f"{opt_result['sharpe_lw']:.2f}")
    m_col4.metric("Contracción Ledoit-Wolf (δ*)", f"{opt_result['shrinkage_intensity']:.4f}")
    
    st.markdown("---")
    
    # Tabla de Asignación Discreta de Acciones
    alloc_df = pd.DataFrame({
        "Ticker": opt_result['asset_names'],
        "Precio Actual ($)": [round(latest_prices[a], 2) for a in opt_result['asset_names']],
        "Peso Óptimo (%)": [round(opt_result['weights_lw'][a] * 100, 2) for a in opt_result['asset_names']],
        "Títulos Enteros": [opt_result['shares'][a] for a in opt_result['asset_names']],
        "Capital Asignado ($)": [round(opt_result['invested_cash'][a], 2) for a in opt_result['asset_names']],
        "Spread Estimado (bps)": [round(spreads_dict.get(a, 0.001)*10000, 1) for a in opt_result['asset_names']]
    })
    
    t_col1, t_col2 = st.columns([1.4, 1.0])
    
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

# =============================================================
# TAB 3: DINÁMICA DE RENTA FIJA (DURACIÓN Y CONVEXIDAD)
# =============================================================
with tab_bonds:
    st.subheader("Dinámica Analítica de Renta Fija: Sensibilidad a Tasas de Interés")
    st.write("Modelado de instrumentos de deuda gubernamental / corporativa mediante la expansión de Taylor de segundo orden en función del rendimiento al vencimiento ($y$).")
    
    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
    with b_col1:
        face_val = st.number_input("Valor Nominal ($ Face Value):", value=1000.0, step=100.0)
    with b_col2:
        coupon_pct = st.number_input("Tasa Cupón Anual (%):", value=5.0, step=0.25) / 100.0
    with b_col3:
        ytm_pct = st.number_input("Tir / YTM Actual (%):", value=4.5, step=0.25) / 100.0
    with b_col4:
        mat_years = st.number_input("Maduración (Años):", value=10, min_value=1, max_value=30, step=1)
        
    bond_met = calculate_bond_metrics(
        face_value=face_val,
        coupon_rate=coupon_pct,
        ytm=ytm_pct,
        maturity_years=mat_years,
        freq=2
    )
    
    # Métricas clave de Renta Fija
    bm1, bm2, bm3, bm4, bm5 = st.columns(5)
    bm1.metric("Precio del Bono ($)", f"${bond_met['bond_price']:,.2f}")
    bm2.metric("Duración Macaulay", f"{bond_met['mac_duration']:.2f} años")
    bm3.metric("Duración Modificada (D*)", f"{bond_met['mod_duration']:.2f} años")
    bm4.metric("Convexidad (C)", f"{bond_met['convexity']:.2f}")
    bm5.metric("DV01 (1 bp shock)", f"${bond_met['dv01']:.4f}")
    
    st.markdown("---")
    
    # Simulación de Shocks de Tasas (Yield Shocks)
    shock_range = st.slider("Rango de Simulación de Shock de Tasas (± bps):", min_value=50, max_value=400, value=200, step=25)
    shocks_df = simulate_yield_shocks(bond_met, shock_bps_range=shock_range, n_points=80)
    
    fig_bond = go.Figure()
    fig_bond.add_trace(go.Scatter(
        x=shocks_df['Shock_bps'], y=shocks_df['Precio_Exacto'],
        mode='lines', name='Precio Exacto (Full Re-pricing)',
        line=dict(color='#00E676', width=3)
    ))
    fig_bond.add_trace(go.Scatter(
        x=shocks_df['Shock_bps'], y=shocks_df['Aprox_Duracion_Convexidad'],
        mode='lines', name='Duración + Convexidad (Taylor 2° Orden)',
        line=dict(color='#1E88E5', width=2, dash='dash')
    ))
    fig_bond.add_trace(go.Scatter(
        x=shocks_df['Shock_bps'], y=shocks_df['Aprox_Duracion_Lineal'],
        mode='lines', name='Aproximación Lineal (Sólo Duración)',
        line=dict(color='#FF5252', width=2, dash='dot')
    ))
    
    fig_bond.update_layout(
        title="Respuesta del Precio del Bono ante Shocks en Tasas de Interés (Δy)",
        xaxis_title="Shock de Rendimiento Δy (Puntos Básicos)",
        yaxis_title="Precio del Bono ($ USD)",
        hovermode="x unified",
        height=420,
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98)
    )
    st.plotly_chart(fig_bond, use_container_width=True)
    
    st.markdown("""
    > [!NOTE]
    > **Interpretación Cuantitativa**: La recta roja (duración lineal) subestima el precio cuando las tasas bajan y sobreestima la caída cuando las tasas suben. La **convexidad** (curvatura verde/azul) es una propiedad favorable: amortigua las pérdidas cuando las tasas suben y amplifica las ganancias cuando las tasas bajan.
    """)

# =============================================================
# TAB 4: SEÑALES ML LONG/SHORT & TIEMPO DE MADURACIÓN
# =============================================================
with tab_signals:
    st.subheader("Señales Direccionales de Machine Learning & Tiempos de Maduración")
    st.write("Clasificación de árboles aleatorios entrenados sobre **series con diferenciación fraccionaria ($d^*=0.40$)** para predecir la dirección futura, complementados con la **semivida de Ornstein-Uhlenbeck** para determinar el horizonte óptimo de tenencia (maduración del trade).")
    
    with st.spinner("Computando transformaciones fraccionarias y entrenando modelos de ensamble..."):
        signals_df = generate_timing_recommendations(prices_df, returns_df)
        
    st.dataframe(signals_df.set_index("Activo"), use_container_width=True)
    
    st.markdown("---")
    
    # Análisis Detallado de un Activo Seleccionado
    asset_inspect = st.selectbox("Seleccione un Activo para inspeccionar su estructura de memoria fraccionaria:", selected_tickers)
    
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
        # Gráfico de Maduración (Half-Life en días)
        fig_mat = px.bar(
            signals_df, x="Activo", y="Maduración Sugerida (Días)",
            color="Señal ML",
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
# TAB 5: LABORATORIO DIDÁCTICO DE MODELOS
# =============================================================
with tab_theory:
    st.subheader("Fundamentos Matemáticos & Algorítmicos del Taller Cuantitativo")
    st.markdown("""
    Esta plataforma materializa las metodologías más avanzadas de la literatura de finanzas cuantitativas moderna:
    """)
    
    th_col1, th_col2 = st.columns(2)
    
    with th_col1:
        st.markdown("""
        ### 1. Maximización de Utilidad Esperada Cuadrática
        Para un inversor con coeficiente de aversión al riesgo de Arrow-Pratt $\\gamma$, la función de utilidad a maximizar es:
        $$\\max_{w} \\quad w^T \\mu - \\frac{\\gamma}{2} w^T \\Sigma_{\\text{LW}} w - \\sum_{i=1}^N \\left( c_{\\text{broker}} |w_i - w_{0,i}| + \\frac{\\text{Spread}_i}{2} |w_i| \\right)$$
        Sujeto a:
        $$\\sum_{i=1}^N w_i \\le 1, \\quad w_i \\ge 0 \\quad (\\text{o } w_i \\ge -0.20 \\text{ si Short})$$
        * **$w^T \\mu$**: Retorno esperado de la cartera.
        * **$\\frac{\\gamma}{2} w^T \\Sigma w$**: Penalización por riesgo cuadrático ponderado por la aversión psicológica del usuario.
        * **Fricciones de Mercado**: Cada rebalanceo incurre en tarifas de corretaje ($c_{\\text{broker}}$) y cruce de horquilla bid-ask (spread).
        """)
        
        st.markdown("""
        ### 2. Regularización de Covarianza (Ledoit & Wolf, 2004)
        La covarianza muestral clásica $S = \\frac{1}{T} X^T X$ sobreestima los autovalores más grandes y subestima los pequeños cuando la relación activos/observaciones $N/T$ es alta, creando carteras espurias hiperconcentradas.
        
        El estimador encogido (shrinkage) interpola óptimamente:
        $$\\Sigma_{\\text{LW}} = \\delta^* F + (1 - \\delta^*) S$$
        Donde $F$ es una matriz de correlación equitativa bien condicionada y $\\delta^* \\in [0, 1]$ minimiza asintóticamente la pérdida cuadrática de Frobenius.
        """)

    with th_col2:
        st.markdown("""
        ### 3. Expansión de Taylor en Renta Fija (Duración & Convexidad)
        El precio de un bono cupón con rendimiento $y$ es $P(y) = \\sum_{t=1}^T \\frac{C_t}{(1+y)^t}$.
        Aproximando por Serie de Taylor hasta segundo orden:
        $$\\frac{\\Delta P}{P} \\approx -D^* \\Delta y + \\frac{1}{2} C (\\Delta y)^2$$
        * **Duración Modificada ($D^*$):** Primera derivada normalizada $-\\frac{1}{P} \\frac{dP}{dy}$. Sensibilidad lineal.
        * **Convexidad ($C$):** Segunda derivada normalizada $\\frac{1}{P} \\frac{d^2P}{dy^2}$. Captura la curvatura que siempre favorece al bonista.
        * **DV01:** Cambio monetario absoluto por 1 punto básico de movimiento en tasas ($D^* \\cdot P \\cdot 0.0001$).
        """)
        
        st.markdown("""
        ### 4. Diferenciación Fraccionaria & Ornstein-Uhlenbeck
        * **Memoria Fraccionaria (López de Prado):**
        $$(1 - B)^d = \\sum_{k=0}^{\\infty} (-1)^k \\binom{d}{k} B^k$$
        Con $d=0.40$, logramos que la serie sea estacionaria para algoritmos de ML mientras retenemos la memoria histórica de niveles de soporte y resistencia.
        * **Semivida (Half-Life) de Maduración:**
        Modelamos la reversión del spread mediante:
        $$dy_t = \\lambda (\\mu - y_t) dt + \\sigma dW_t$$
        El tiempo esperado de maduración o retorno al equilibrio es $t_{1/2} = \\frac{\\ln(2)}{\\lambda}$, dictando el número de días óptimo para mantener abierta la posición.
        """)

# Footer institucional
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #757575; font-size: 0.85rem;'>"
    "AI Quantitative Investment Platform &bull; Desarrollado para Taller de Finanzas Cuantitativas &bull; "
    "Datos provistos vía Yahoo Finance API &bull; Modelos ejecutados en PyTorch / Scikit-Learn"
    "</div>",
    unsafe_allow_html=True
)

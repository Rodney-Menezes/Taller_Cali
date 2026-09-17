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
    # SUBTAB 5: SIMULACIÓN ORNSTEIN-UHLENBECK
    # -------------------------------------------------------------
    with subtab5:
        st.markdown("### 🔄 Proceso de Difusión de Ornstein-Uhlenbeck & Semivida de Maduración")
        st.latex(r"dy_t = \theta (\mu - y_t) dt + \sigma dW_t, \quad t_{1/2} = \frac{\ln(2)}{\theta}")
        st.write("Simulación estocástica de trayectorias de precios y spreads que experimentan reversión a la media. Permite determinar científicamente la duración óptima de un trade antes de su decaimiento.")
        
        col_ou1, col_ou2 = st.columns([1, 2])
        with col_ou1:
            st.markdown("#### Parámetros Estocásticos")
            theta_sim = st.slider("Velocidad de Reversión (θ):", 0.5, 15.0, 4.0, 0.5, key="sld_th_ou")
            mu_sim = st.slider("Nivel de Equilibrio (μ $):", 50.0, 200.0, 100.0, 5.0, key="sld_mu_ou")
            y0_sim = st.slider("Precio Inicial Desviado (y0 $):", 50.0, 200.0, 125.0, 5.0, key="sld_y0_ou")
            sigma_sim = st.slider("Volatilidad de Difusión (σ):", 5.0, 50.0, 15.0, 2.5, key="sld_sig_ou")
            n_paths = st.slider("Número de Trayectorias Monte Carlo:", 5, 30, 12, key="sld_np_ou")
            n_days = st.slider("Horizonte de Simulación (Días):", 15, 90, 45, key="sld_nd_ou")
            
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

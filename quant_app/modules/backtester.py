import numpy as np
import pandas as pd
import plotly.graph_objects as go

def run_portfolio_backtest(prices_df, weights_dict, benchmark_ticker="SPY", budget=10000.0, rf=0.04):
    """
    Ejecuta una simulación histórica walk-forward de la cartera con las ponderaciones dadas.
    Calcula métricas institucionales (Sharpe, Sortino, Calmar, Max Drawdown)
    y genera gráficos interactivos de Curva de Riqueza y Underwater Plot.
    """
    valid_assets = [a for a in weights_dict.keys() if a in prices_df.columns and weights_dict[a] > 0]
    
    if not valid_assets:
        return None
        
    p_subset = prices_df[valid_assets].dropna()
    rets_subset = p_subset.pct_change().dropna()
    
    # Vector de pesos normalizado para los activos disponibles
    raw_w = np.array([weights_dict[a] for a in valid_assets], dtype=np.float64)
    total_w = np.sum(raw_w)
    if total_w > 0:
        w_vec = raw_w / total_w
    else:
        w_vec = np.ones(len(valid_assets)) / len(valid_assets)
        
    # Retornos diarios ponderados del portafolio
    port_daily_rets = rets_subset.dot(w_vec)
    
    # Curva de capital acumulado ($ USD)
    cum_growth = (1.0 + port_daily_rets).cumprod()
    equity_series = budget * cum_growth
    # Añadir punto base inicial
    equity_series = pd.concat([pd.Series([budget], index=[rets_subset.index[0] - pd.Timedelta(days=1)]), equity_series])
    
    # Benchmark: SPY si está disponible, o promedio equiponderado del universo
    if benchmark_ticker in prices_df.columns:
        bm_prices = prices_df[benchmark_ticker].reindex(rets_subset.index).ffill().bfill()
        bm_rets = bm_prices.pct_change().fillna(0.0)
    else:
        bm_rets = rets_subset.mean(axis=1)
        
    bm_cum = budget * (1.0 + bm_rets).cumprod()
    bm_cum = pd.concat([pd.Series([budget], index=[rets_subset.index[0] - pd.Timedelta(days=1)]), bm_cum])
    
    # -------------------------------------------------------------
    # CÁLCULO DE MÉTRICAS INSTITUCIONALES
    # -------------------------------------------------------------
    n_days = len(port_daily_rets)
    years = max(n_days / 252.0, 0.1)
    
    final_equity = float(equity_series.iloc[-1])
    cagr = float((final_equity / budget) ** (1.0 / years) - 1.0)
    
    ann_vol = float(port_daily_rets.std() * np.sqrt(252.0))
    
    # Volatilidad a la baja (Semi-desviación / Downside Risk para Sortino)
    downside_rets = port_daily_rets[port_daily_rets < 0.0]
    if len(downside_rets) > 0:
        downside_vol = float(np.sqrt(np.mean(downside_rets ** 2)) * np.sqrt(252.0))
    else:
        downside_vol = ann_vol
        
    # Ratio de Sharpe
    sharpe = float((cagr - rf) / (ann_vol + 1e-8))
    
    # Ratio de Sortino
    sortino = float((cagr - rf) / (downside_vol + 1e-8))
    
    # Serie de Drawdown y Máximo Drawdown
    running_max = equity_series.cummax()
    drawdown_series = (equity_series - running_max) / running_max
    max_drawdown = float(drawdown_series.min()) # Valor negativo e.g. -0.15
    
    # Ratio de Calmar (CAGR / |Max Drawdown|)
    calmar = float(cagr / (abs(max_drawdown) + 1e-8))
    
    # Win rate diario
    win_rate = float(np.mean(port_daily_rets > 0.0) * 100.0)
    
    # Retorno Benchmark
    bm_final = float(bm_cum.iloc[-1])
    bm_cagr = float((bm_final / budget) ** (1.0 / years) - 1.0)
    bm_max_dd = float(((bm_cum - bm_cum.cummax()) / bm_cum.cummax()).min())
    
    # -------------------------------------------------------------
    # GRÁFICOS INTERACTIVOS CON PLOTLY
    # -------------------------------------------------------------
    # 1. Curva de Riqueza Acumulada
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        x=equity_series.index, y=equity_series.values,
        mode='lines', name='Portafolio Cuantitativo Optimizado',
        line=dict(color='#00E676', width=2.5)
    ))
    fig_equity.add_trace(go.Scatter(
        x=bm_cum.index, y=bm_cum.values,
        mode='lines', name=f'Benchmark ({benchmark_ticker if benchmark_ticker in prices_df.columns else "Equiponderado 1/N"})',
        line=dict(color='#90A4AE', width=1.5, dash='dash')
    ))
    fig_equity.add_hline(y=budget, line_dash="dot", line_color="#FFD600", annotation_text=f"Capital Inicial (${budget:,.0f})")
    fig_equity.update_layout(
        title="Curva de Crecimiento Patrimonial Histórico ($ USD)",
        xaxis_title="Fecha", yaxis_title="Patrimonio de la Cartera ($)",
        hovermode="x unified", height=350, margin=dict(l=20, r=20, t=40, b=20)
    )
    
    # 2. Underwater Plot (Drawdown acumulado)
    fig_underwater = go.Figure()
    fig_underwater.add_trace(go.Scatter(
        x=drawdown_series.index, y=drawdown_series.values * 100.0,
        mode='lines', name='Drawdown (%)',
        line=dict(color='#FF5252', width=1.5),
        fill='tozeroy', fillcolor='rgba(255, 82, 82, 0.25)'
    ))
    fig_underwater.add_hline(y=max_drawdown * 100.0, line_dash="dash", line_color="#FF1744",
                             annotation_text=f"Máximo Drawdown ({max_drawdown*100:.1f}%)")
    fig_underwater.update_layout(
        title="Gráfico Submarino de Drawdown (Caídas Pico-a-Valle en %)",
        xaxis_title="Fecha", yaxis_title="Pérdida respecto al Máximo Histórico (%)",
        hovermode="x unified", height=280, margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return {
        'equity_series': equity_series,
        'drawdown_series': drawdown_series,
        'cagr': cagr,
        'ann_vol': ann_vol,
        'downside_vol': downside_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'final_equity': final_equity,
        'bm_cagr': bm_cagr,
        'bm_max_dd': bm_max_dd,
        'fig_equity': fig_equity,
        'fig_underwater': fig_underwater
    }

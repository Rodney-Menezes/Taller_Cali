import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
import plotly.figure_factory as ff
import plotly.graph_objects as go

def get_quasi_diagonal(linkage_matrix):
    """
    Algoritmo de seriación (quasi-diagonalización) de Marcos López de Prado.
    Reordena los activos de forma que activos con correlaciones similares queden adyacentes.
    """
    link = linkage_matrix.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
        
    return sort_ix.tolist()

def get_cluster_variance(cov, cluster_items):
    """
    Calcula la varianza interna de un cluster aplicando ponderaciones inversas a la varianza individual.
    """
    cov_slice = cov[np.ix_(cluster_items, cluster_items)]
    diag = np.diag(cov_slice)
    diag_inv = 1.0 / np.clip(diag, 1e-8, None)
    weights = diag_inv / np.sum(diag_inv)
    cluster_var = np.dot(np.dot(weights, cov_slice), weights)
    return float(cluster_var)

def compute_recursive_bisection(cov, sorted_indices):
    """
    Bisección recursiva: distribuye el peso de capital (1.0) entre subgrupos
    de forma inversamente proporcional a la varianza de cada sub-cluster.
    """
    weights = pd.Series(1.0, index=sorted_indices)
    clusters = [sorted_indices]
    
    while len(clusters) > 0:
        clusters = [
            c[start:end] 
            for c in clusters 
            for start, end in ((0, len(c) // 2), (len(c) // 2, len(c))) 
            if len(c) > 1
        ]
        for i in range(0, len(clusters), 2):
            c0 = clusters[i]
            c1 = clusters[i + 1]
            var0 = get_cluster_variance(cov, c0)
            var1 = get_cluster_variance(cov, c1)
            
            # Factor de asignación alpha: menor varianza recibe mayor peso
            alpha = 1.0 - var0 / (var0 + var1 + 1e-12)
            weights[c0] *= alpha
            weights[c1] *= (1.0 - alpha)
            
    return weights

def optimize_hrp_portfolio(returns_df, latest_prices=None, budget=10000.0, max_weight_cap=0.35):
    """
    Ejecuta el pipeline completo de Hierarchical Risk Parity (HRP):
    1. Distancia de correlación métrica
    2. Clustering jerárquico aglomerativo (single linkage)
    3. Quasi-diagonalización
    4. Bisección recursiva
    5. Asignación discreta de acciones y buffer de liquidez
    """
    assets = list(returns_df.columns)
    n_assets = len(assets)
    
    if n_assets <= 1:
        w_dict = {assets[0]: 1.0} if n_assets == 1 else {}
        return {
            'weights': w_dict,
            'dendrogram_fig': None,
            'expected_return': 0.05,
            'volatility': 0.15,
            'sharpe': 0.33,
            'shares': {},
            'cash_buffer': budget
        }
        
    cov_df = returns_df.cov()
    corr_df = returns_df.corr()
    
    cov = cov_df.values
    corr = corr_df.values
    
    # 1. Matriz de Distancia de Correlación angular
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)
    
    # Condensar distancia simétrica para scipy
    condensed_dist = squareform(dist)
    
    # 2. Linkage jerárquico
    linkage = sch.linkage(condensed_dist, method='single')
    
    # 3. Quasi-diagonalización
    sorted_idx = get_quasi_diagonal(linkage)
    
    # 4. Bisección recursiva
    raw_weights_series = compute_recursive_bisection(cov, sorted_idx)
    raw_weights_series = raw_weights_series.sort_index()
    raw_weights = raw_weights_series.values
    
    # Normalizar y aplicar tope máximo opcional
    if max_weight_cap is not None and max_weight_cap < 1.0:
        raw_weights = np.minimum(raw_weights, max_weight_cap)
        raw_weights /= np.sum(raw_weights)
        
    weights_dict = {assets[i]: float(raw_weights[i]) for i in range(n_assets)}
    
    # Métricas anualizadas del portafolio HRP
    mean_rets = returns_df.mean().values * 252.0
    cov_annual = cov * 252.0
    
    port_ret = float(np.dot(raw_weights, mean_rets))
    port_vol = float(np.sqrt(np.dot(raw_weights, np.dot(cov_annual, raw_weights))))
    rf = 0.04
    port_sharpe = float((port_ret - rf) / (port_vol + 1e-8))
    
    # 5. Generar Dendrograma con Plotly
    try:
        fig_dendro = ff.create_dendrogram(
            dist,
            orientation='bottom',
            labels=assets
        )
        fig_dendro.update_layout(
            title="Dendrograma de Clustering Jerárquico de Activos (López de Prado)",
            xaxis_title="Activos Agrupados por Similitud de Covarianza",
            yaxis_title="Distancia Métrica de Correlación d(i, j)",
            height=340,
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
    except Exception:
        fig_dendro = None
        
    # 6. Asignación discreta de capital
    shares_dict = {}
    invested_total = 0.0
    if latest_prices is not None:
        for a in assets:
            p = latest_prices.get(a, 0.0)
            target_alloc = budget * weights_dict[a]
            if p > 0:
                sh = int(np.floor(target_alloc / p))
            else:
                sh = 0
            shares_dict[a] = sh
            invested_total += sh * p
            
    cash_rem = max(0.0, budget - invested_total)
    
    return {
        'weights': weights_dict,
        'dendrogram_fig': fig_dendro,
        'expected_return': port_ret,
        'volatility': port_vol,
        'sharpe': port_sharpe,
        'shares': shares_dict,
        'invested_capital': invested_total,
        'cash_buffer': cash_rem
    }

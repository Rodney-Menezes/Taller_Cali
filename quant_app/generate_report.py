# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from scipy.optimize import minimize
from datetime import datetime

# Rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from modules.data_loader import get_asset_data
from modules.risk_profiler import calculate_risk_aversion
from modules.portfolio_optimizer import optimize_portfolio_utility
from modules.hrp_optimizer import optimize_hrp_portfolio
from modules.black_litterman import calculate_black_litterman
from modules.portfolio_monte_carlo import simulate_multivariate_portfolio_mc
from modules.backtester import run_portfolio_backtest
from modules.stress_testing import simulate_crisis_stress
from modules.market_regimes import detect_market_regimes
from modules.fixed_income import get_fixed_income_profile, simulate_yield_shocks
from modules.deep_learning_model import train_deep_learning_agent
from modules.timing_signals import apply_frac_diff

print('Imports and modules loaded successfully.')

def draw_page_header(fig, title, subtitle):
    ax_h = fig.add_axes([0.05, 0.915, 0.90, 0.065])
    ax_h.set_facecolor('#0D1B2A')
    for sp in ax_h.spines.values():
        sp.set_visible(False)
    ax_h.text(0.02, 0.68, 'PONTIFICIA UNIVERSIDAD JAVERIANA DE CALI - COLOMBIA', 
              color='#FFFFFF', fontsize=11, fontweight='bold', va='center')
    ax_h.text(0.02, 0.28, 'Taller de Finanzas Cuantitativas • Propiedad Intelectual: Rodney Menezes © 2026', 
              color='#90CAF9', fontsize=8.5, va='center')
    ax_h.set_xticks([])
    ax_h.set_yticks([])
    
    # Titulo de seccion
    fig.text(0.05, 0.890, title, fontsize=13, fontweight='bold', color='#0D1B2A')
    fig.text(0.05, 0.872, subtitle, fontsize=9, fontstyle='italic', color='#4A5568')

def draw_page_footer(fig, page_num, total=10):
    fig.text(0.05, 0.030, '—' * 95, color='#CBD5E0', fontsize=8)
    fig.text(0.05, 0.018, 'Manual Técnico y Metodológico de Modelos Cuantitativos • Taller de Finanzas Cuantitativas', 
             color='#718096', fontsize=7.5)
    fig.text(0.95, 0.018, f'Página {page_num} de {total}', color='#1A202C', fontsize=8, fontweight='bold', ha='right')

def draw_analysis_box(fig, x, y, w, h, sections, bg='#F8F9FA'):
    ax_b = fig.add_axes([x, y, w, h])
    ax_b.set_facecolor(bg)
    for sp in ax_b.spines.values():
        sp.set_color('#CBD5E0')
        sp.set_linewidth(1.0)
    ax_b.set_xticks([])
    ax_b.set_yticks([])
    
    curr_y = 0.96
    for title, text in sections:
        ax_b.text(0.02, curr_y, title, color='#0D1B2A', fontsize=8.5, fontweight='bold', va='top')
        curr_y -= 0.055
        # Wrap text lines
        import textwrap
        lines = textwrap.wrap(text, width=105)
        for line in lines:
            if curr_y < 0.04:
                break
            ax_b.text(0.02, curr_y, line, color='#2D3748', fontsize=7.2, va='top')
            curr_y -= 0.038
        curr_y -= 0.020

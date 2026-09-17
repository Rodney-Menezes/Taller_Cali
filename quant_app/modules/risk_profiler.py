import numpy as np

def calculate_risk_aversion(q_horizon, q_drawdown, q_objective, q_income, q_experience):
    """
    Calcula el coeficiente de aversión al riesgo relativo de Arrow-Pratt: gamma in [1.0, 10.0]
    A partir de 5 preguntas psicométricas y financieras.
    """
    # 1. Horizonte temporal
    score_h = {
        "Menos de 6 meses (Muy corto plazo)": 1,
        "De 6 meses a 2 años (Corto plazo)": 2,
        "De 2 a 5 años (Mediano plazo)": 3,
        "Más de 5 años (Largo plazo)": 4
    }.get(q_horizon, 3)

    # 2. Reacción ante caída del 20%
    score_d = {
        "Vendo toda mi cartera para evitar mayores pérdidas": 1,
        "Vendo una parte para asegurar liquidez": 2,
        "Mantengo la calma y espero a que se recupere": 3,
        "Aprovecho la caída y compro más títulos a descuento": 4
    }.get(q_drawdown, 3)

    # 3. Objetivo principal
    score_o = {
        "Preservación estricta de capital (tolerancia cero a pérdidas)": 1,
        "Generación de ingresos estables con baja volatilidad": 2,
        "Crecimiento moderado con riesgo controlado": 3,
        "Maximización agresiva de rentabilidad a largo plazo": 4
    }.get(q_objective, 3)

    # 4. Estabilidad de ingresos
    score_i = {
        "Ingresos altamente variables o jubilación": 1,
        "Ingresos moderadamente estables": 2,
        "Empleo o negocio muy estable con capacidad de ahorro constante": 3,
        "Flujos de caja abundantes e independientes del mercado": 4
    }.get(q_income, 3)

    # 5. Experiencia previa
    score_e = {
        "Principiante (cuentas de ahorro o depósitos)": 1,
        "Intermedio (fondos indexados y acciones)": 2,
        "Avanzado (acciones, derivados y futuros)": 3,
        "Institucional / Profesional cuantitativo": 4
    }.get(q_experience, 2)

    total_score = score_h + score_d + score_o + score_i + score_e  # Min: 5, Max: 20
    
    # Mapeo inverso: a mayor puntaje, MENOR aversión al riesgo (gamma)
    # Score 5 -> gamma = 10.0 (Ultra-conservador)
    # Score 20 -> gamma = 1.0 (Agresivo puro)
    gamma = 10.0 - ((total_score - 5) / 15.0) * 9.0
    gamma = float(np.clip(gamma, 1.0, 10.0))

    if gamma <= 2.5:
        perfil = "Agresivo / Máximo Crecimiento"
        descripcion = "Elevada tolerancia a la volatilidad. Enfoque en maximizar retorno esperado aun con drawdowns significativos."
        max_equity_pct = 95
    elif gamma <= 4.5:
        perfil = "Crecimiento Moderado"
        descripcion = "Tolerancia moderada a la volatilidad. Busca crecimiento sostenido con diversificación inteligente."
        max_equity_pct = 80
    elif gamma <= 6.5:
        perfil = "Equilibrado / Balanceado"
        descripcion = "Preferencia por un balance simétrico entre crecimiento patrimonial y control estricto del riesgo a la baja."
        max_equity_pct = 60
    elif gamma <= 8.5:
        perfil = "Conservador"
        descripcion = "Prioridad en proteger el capital. Se toleran fluctuaciones menores a cambio de mayor estabilidad."
        max_equity_pct = 40
    else:
        perfil = "Ultra-Conservador / Preservación"
        descripcion = "Aversión máxima a pérdidas. Alta asignación en activos defensivos, renta fija de corta duración y liquidez."
        max_equity_pct = 20

    return {
        'gamma': gamma,
        'total_score': total_score,
        'perfil': perfil,
        'descripcion': descripcion,
        'max_equity_pct': max_equity_pct
    }

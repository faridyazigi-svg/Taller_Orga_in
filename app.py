import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Concentración de mercado", layout="wide")

# ---------- 1. INDICADORES (s tiene forma (..., N) y suma 1) ----------
def cr(s, k):
    return np.sort(s, axis=-1)[..., ::-1][..., :k].sum(-1)

def ihh(s):  # escala 0-1 (se muestra x10.000 al interpretar)
    return (s ** 2).sum(-1)

def idom(s):  # índice de dominancia: suma de (s_i^2 / IHH)^2
    h = s ** 2 / ihh(s)[..., None]
    return (h ** 2).sum(-1)

def ent(s):  # entropía de Shannon (logaritmo natural)
    return -(s * np.log(np.where(s > 0, s, 1))).sum(-1)

def valor(ind, s, k=4):
    return {"CRk": lambda: cr(s, k), "IHH": lambda: ihh(s),
            "ID": lambda: idom(s), "IE": lambda: ent(s)}[ind]()

# ---------- 2. UMBRALES (cámbialos por los de tu clase si son distintos) ----------
def nivel(ind, s, k=4):
    v = float(valor(ind, s, k))
    if ind == "CRk":
        return ("Baja" if v < 0.40 else "Moderada" if v <= 0.60 else "Alta"), "menos de 40% = baja, 40% a 60% = moderada, más de 60% = alta"
    if ind == "IHH":
        v *= 10000
        return ("Baja" if v < 1500 else "Moderada" if v <= 2500 else "Alta"), "menos de 1.500 = baja, 1.500 a 2.500 = moderada, más de 2.500 = alta"
    if ind == "ID":
        return ("Baja" if v < 0.25 else "Moderada" if v <= 0.50 else "Alta"), "menos de 0,25 = baja, 0,25 a 0,50 = moderada, más de 0,50 = alta"
    e = v / np.log(len(s))  # entropía normalizada (0 a 1): MÁS alta = MENOS concentración
    return ("Baja" if e > 0.80 else "Moderada" if e >= 0.50 else "Alta"), "entropía normalizada (IE / ln N): más de 0,80 = baja, 0,50 a 0,80 = moderada, menos de 0,50 = alta"

# ---------- 3. SIMULACIÓN MONTE CARLO ----------
@st.cache_data(show_spinner="Simulando...")
def simular(n, iters, ind, k):
    # Dirichlet(1,...,1): cada fila suma exactamente 1
    cuotas = np.random.default_rng().dirichlet(np.ones(n), size=iters)
    return valor(ind, cuotas, k)

# ---------- 4. INTERFAZ ----------
st.title("Simulador de concentración de mercado (Monte Carlo)")

with st.sidebar:
    st.header("Parámetros")
    ind = st.selectbox("Indicador", ["CRk", "IHH", "ID", "IE"])
    N = int(st.number_input("Número de empresas (N)", 2, 100, 5, 1))
    k = int(st.number_input("k (solo para CRk)", 1, N, min(4, N), 1)) if ind == "CRk" else 4
    iters = int(st.number_input("Iteraciones", 100, 50000, 1000, 100,
                help="Mínimo 100 (con menos el gráfico es muy irregular) y máximo 50.000 (con más la app se vuelve lenta)."))
    if iters > 10000:
        st.warning("Muchas iteraciones aumentan el tiempo de espera y el uso de memoria y CPU del servidor.")
    else:
        st.info("Más iteraciones = gráfico más preciso, pero más lento y con más consumo de recursos.")

st.subheader("Caso particular")
modo = st.radio("¿Cómo definir las cuotas?", ["Generar al azar", "Ingresar manualmente"], horizontal=True)
caso = None
if modo == "Generar al azar":
    nuevo = st.button("Generar nuevo caso")
    if nuevo or "caso" not in st.session_state or len(st.session_state.caso) != N:
        st.session_state.caso = np.random.default_rng().dirichlet(np.ones(N))
    caso = st.session_state.caso
else:
    txt = st.text_input(f"{N} cuotas en % separadas por coma (deben sumar 100)", ", ".join([f"{100 / N:.2f}"] * N))
    try:
        vals = np.array([float(x) for x in txt.replace(";", ",").split(",")]) / 100
        if len(vals) != N:
            st.error(f"Debes ingresar exactamente {N} cuotas (ingresaste {len(vals)}).")
        elif (vals < 0).any():
            st.error("Las cuotas no pueden ser negativas.")
        elif abs(vals.sum() - 1) > 1e-3:
            st.error(f"Las cuotas suman {vals.sum() * 100:.2f}%, deben sumar 100%.")
        else:
            caso = vals / vals.sum()
    except ValueError:
        st.error("Escribe solo números separados por comas.")

if caso is not None:
    st.write("Cuotas (%):", np.round(np.sort(caso)[::-1] * 100, 2).tolist())
    sim = simular(N, iters, ind, k)
    v = float(valor(ind, caso, k))
    pct = float((sim <= v).mean() * 100)

    fig = go.Figure(go.Histogram(x=sim, nbinsx=40, name="Simulación"))
    fig.add_vline(x=v, line_color="red", line_width=3,
                  annotation_text=f"Tu caso: {v:.4f} (percentil {pct:.1f})")
    nombre = f"CR{k}" if ind == "CRk" else ind
    fig.update_layout(xaxis_title=f"Valor del indicador {nombre}", yaxis_title="Frecuencia (nº de simulaciones)",
                      title=f"Distribución de {nombre} con N={N} y {iters:,} iteraciones")
    st.plotly_chart(fig, use_container_width=True)
    st.metric(f"{nombre} del caso particular", f"{v:.4f}", f"Percentil {pct:.1f}", delta_color="off")

    # ---------- 5. EVALUADOR ----------
    st.subheader("Pregunta de evaluación")
    resp = st.radio(f"Según {nombre}, ¿qué nivel de concentración tiene tu caso?",
                    ["Baja", "Moderada", "Alta"], index=None, horizontal=True)
    if st.button("Responder") and resp:
        correcta, regla = nivel(ind, caso, k)
        txt_v = f"{v * 10000:,.0f}" if ind == "IHH" else f"{v:.4f}"
        msg = (f"El {nombre} del caso es {txt_v}. Regla usada: {regla}. "
               f"Por lo tanto la concentración es **{correcta}**. "
               f"Además, el caso está en el percentil {pct:.1f} de la simulación.")
        (st.success if resp == correcta else st.error)(("¡Correcto! " if resp == correcta else f"Incorrecto. Tu respuesta fue {resp}. ") + msg)

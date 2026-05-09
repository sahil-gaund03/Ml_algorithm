# =========================================================
# ML ALGORITHM HANDBOOK — Interactive Streamlit Explorer
# =========================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="ML Algorithm Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# GLOBAL CSS
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,400&display=swap');

:root {
    --bg:         #06080f;
    --bg-card:    #0c1020;
    --bg-card2:   #101525;
    --border:     rgba(255,255,255,0.07);
    --violet:     #8b5cf6;
    --blue:       #3b82f6;
    --cyan:       #06b6d4;
    --green:      #22c55e;
    --amber:      #f59e0b;
    --red:        #ef4444;
    --text:       #e2e8f0;
    --muted:      #64748b;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border);
}
#MainMenu, footer, header { visibility: hidden; }

/* Typography */
h1,h2,h3,.hero-title { font-family: 'Syne', sans-serif; }

/* Hero */
.hero {
    text-align: center;
    padding: 2.4rem 1rem 1.2rem;
}
.hero-badge {
    display: inline-block;
    font-size: 11px; font-weight:600; letter-spacing:3px; text-transform:uppercase;
    color: var(--violet);
    background: rgba(139,92,246,0.1);
    border: 1px solid rgba(139,92,246,0.25);
    border-radius: 20px; padding: 5px 16px; margin-bottom: 16px;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.2rem, 4vw, 3.4rem); font-weight:800;
    background: linear-gradient(135deg, #f1f5f9 30%, var(--violet) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.15; margin: 0 0 10px;
}
.hero-sub { color: var(--muted); font-size: .95rem; font-weight:300; max-width: 480px; margin: 0 auto; }

/* Cards */
.glass {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px; padding: 22px 24px; margin-bottom: 16px;
    position: relative; overflow: hidden;
}
.glass::before {
    content: ''; position: absolute; top:0; left:0; right:0; height:1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
}

/* Chapter pill nav */
.chapter-pill {
    display:inline-block; padding: 6px 18px; border-radius:20px;
    font-size:0.8rem; font-weight:600; letter-spacing:.5px;
    margin: 4px; cursor:pointer;
}

/* Section label */
.slabel {
    font-family:'Syne',sans-serif; font-size:10px; font-weight:700;
    letter-spacing:3px; text-transform:uppercase; color:var(--muted); margin-bottom:8px;
}

/* Metric row */
.mcard {
    background: var(--bg-card); border:1px solid var(--border);
    border-radius:12px; padding:16px; text-align:center;
    position:relative; overflow:hidden;
}
.mcard-num { font-family:'Syne',sans-serif; font-size:1.8rem; font-weight:800; }
.mcard-lbl { color:var(--muted); font-size:.72rem; letter-spacing:1.5px; text-transform:uppercase; margin-top:4px; }
.mcard::after { content:''; position:absolute; bottom:0; left:0; right:0; height:2px; }
.v  { color: var(--violet); } .v::after { background: linear-gradient(90deg, var(--violet), var(--blue)); }
.b  { color: var(--blue);   } .b::after { background: linear-gradient(90deg, var(--blue), var(--cyan)); }
.c  { color: var(--cyan);   } .c::after { background: linear-gradient(90deg, var(--cyan), var(--green)); }
.g  { color: var(--green);  } .g::after { background: linear-gradient(90deg, var(--green), var(--cyan)); }
.a  { color: var(--amber);  } .a::after { background: linear-gradient(90deg, var(--amber), var(--red)); }

/* Formula box */
.formula {
    background: rgba(139,92,246,0.08); border:1px solid rgba(139,92,246,0.2);
    border-radius:10px; padding:14px 18px; font-family:monospace;
    font-size:.88rem; color:#c4b5fd; margin:10px 0;
}

/* Insight list */
.insight { display:flex; gap:10px; align-items:flex-start; padding:9px 0;
    border-bottom:1px solid var(--border); font-size:.85rem; color:#cbd5e1; }
.insight:last-child { border-bottom:none; }
.dot { width:7px; height:7px; border-radius:50%; margin-top:5px; flex-shrink:0; }
.dv { background:var(--violet); } .db { background:var(--blue); }
.dc { background:var(--cyan);   } .dg { background:var(--green); }
.da { background:var(--amber);  } .dr { background:var(--red); }

/* Divider */
.hr { height:1px; background:var(--border); margin:20px 0; }

/* Sidebar items */
.sb-algo {
    display:flex; align-items:center; gap:10px; padding:9px 10px;
    border-radius:10px; margin-bottom:4px; cursor:pointer;
    font-size:.85rem; color:#94a3b8; border:1px solid transparent;
    transition: all .2s;
}
.sb-algo.active {
    background: rgba(139,92,246,0.12); border-color: rgba(139,92,246,0.3);
    color: #c4b5fd; font-weight:600;
}

/* Stapp override */
.stButton>button {
    background: linear-gradient(135deg, var(--violet), var(--blue)) !important;
    color:#fff !important; border:none !important; border-radius:9px !important;
    font-family:'Syne',sans-serif !important; font-weight:700 !important;
    font-size:.88rem !important; letter-spacing:.5px !important;
    transition: all .2s !important;
}
.stButton>button:hover { opacity:.85 !important; transform:translateY(-1px) !important; }
.stSlider [data-baseweb="slider"] div[role="slider"] { background: var(--violet) !important; }
label, .stSelectbox label, .stSlider label { color: var(--muted) !important; font-size:.78rem !important; letter-spacing:1px !important; text-transform:uppercase !important; }
.stSelectbox [data-baseweb="select"] { background: var(--bg-card2) !important; border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_iris_data():
    ds = load_iris(as_frame=True)
    X = ds['data'][['sepal length (cm)', 'sepal width (cm)']]
    X_full = ds['data']
    y = ds['target']
    return X, X_full, y

X_iris, X_iris_full, y_iris = load_iris_data()

# =========================================================
# SIDEBAR NAV
# =========================================================

ALGORITHMS = {
    "🔵 K-Means Clustering":    "kmeans",
    "🟣 PCA":                   "pca",
    "🟡 Linear Regression":     "linreg",
    "🟠 Logistic Regression":   "logreg",
    "🔴 Decision Trees":        "dtree",
}

with st.sidebar:
    st.markdown("""
    <div style='padding:6px 0 18px;'>
        <div style='font-family:Syne,sans-serif;font-size:1.25rem;font-weight:800;color:#f1f5f9;'>🧠 ML Explorer</div>
        <div style='font-size:.72rem;color:#475569;letter-spacing:2px;text-transform:uppercase;margin-top:2px;'>Algorithm Handbook</div>
    </div>
    """, unsafe_allow_html=True)

    algo_choice = st.radio(
        "Choose Algorithm",
        list(ALGORITHMS.keys()),
        label_visibility="collapsed"
    )
    algo_key = ALGORITHMS[algo_choice]

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:.72rem;color:#475569;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;'>About</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:.82rem;color:#475569;line-height:1.6;'>
    Interactive explorer for core ML algorithms. Adjust parameters and see results update in real time.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:.72rem;color:#334155;text-align:center;'>Built with scikit-learn + Streamlit</div>", unsafe_allow_html=True)

# =========================================================
# HERO
# =========================================================

st.markdown(f"""
<div class="hero">
    <div class="hero-badge">Interactive ML Learning</div>
    <div class="hero-title">{algo_choice}</div>
    <div class="hero-sub">Adjust parameters, visualize results, and build intuition for how this algorithm works.</div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# PAGES
# =========================================================

# ─── K-MEANS ──────────────────────────────────────────────
if algo_key == "kmeans":

    col_ctrl, col_main = st.columns([1, 2], gap="large")

    with col_ctrl:
        st.markdown('<div class="slabel">Controls</div>', unsafe_allow_html=True)
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        k = st.slider("Number of Clusters (K)", 2, 8, 3)
        max_iter = st.slider("Max Iterations", 1, 20, 10)
        random_state = st.slider("Random Seed", 0, 99, 42)
        run_btn = st.button("▶  Run K-Means", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="slabel" style="margin-top:10px;">How it works</div>', unsafe_allow_html=True)
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        steps = [
            ("db", "Choose K and a distance metric"),
            ("dv", "Randomly initialise K centroids"),
            ("dc", "Assign each point to nearest centroid"),
            ("dg", "Recompute centroids as cluster means"),
            ("da", "Repeat until convergence"),
        ]
        for dot, text in steps:
            st.markdown(f'<div class="insight"><span class="dot {dot}"></span><span>{text}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_main:
        st.markdown('<div class="slabel">Clustering Result — Iris Dataset (Sepal features)</div>', unsafe_allow_html=True)

        model = KMeans(n_clusters=k, max_iter=max_iter, random_state=random_state, n_init=10)
        model.fit(X_iris)
        labels = model.predict(X_iris)
        centers = model.cluster_centers_

        palette = px.colors.qualitative.Vivid
        fig = go.Figure()
        for i in range(k):
            mask = labels == i
            fig.add_trace(go.Scatter(
                x=X_iris[mask]['sepal length (cm)'], y=X_iris[mask]['sepal width (cm)'],
                mode='markers', name=f'Cluster {i}',
                marker=dict(color=palette[i % len(palette)], size=8, opacity=0.8,
                            line=dict(color='white', width=0.5))
            ))
        fig.add_trace(go.Scatter(
            x=centers[:, 0], y=centers[:, 1], mode='markers', name='Centroids',
            marker=dict(symbol='x', size=16, color='white',
                        line=dict(color='white', width=2))
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,17,32,1)',
            height=380, margin=dict(t=20, b=20, l=10, r=10),
            xaxis=dict(title='Sepal Length (cm)', gridcolor='#1e293b', color='#64748b'),
            yaxis=dict(title='Sepal Width (cm)',  gridcolor='#1e293b', color='#64748b'),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8')),
            font=dict(family='DM Sans')
        )
        st.plotly_chart(fig, use_container_width=True)

        # Elbow
        st.markdown('<div class="slabel">Elbow Method — Inertia vs K</div>', unsafe_allow_html=True)
        inertias = []
        for ki in range(1, 12):
            km = KMeans(n_clusters=ki, n_init=10, random_state=42)
            km.fit(X_iris)
            inertias.append(km.inertia_)

        fig_elbow = go.Figure(go.Scatter(
            x=list(range(1, 12)), y=inertias, mode='lines+markers',
            line=dict(color='#8b5cf6', width=2),
            marker=dict(color='#c4b5fd', size=7)
        ))
        fig_elbow.add_vline(x=k, line=dict(color='#06b6d4', width=1.5, dash='dash'))
        fig_elbow.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,17,32,1)',
            height=230, margin=dict(t=10, b=20, l=10, r=10),
            xaxis=dict(title='K', gridcolor='#1e293b', color='#64748b'),
            yaxis=dict(title='Inertia', gridcolor='#1e293b', color='#64748b'),
            font=dict(family='DM Sans')
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

    # Metrics
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    cluster_sizes = pd.Series(labels).value_counts()
    with m1:
        st.markdown(f'<div class="mcard v"><div class="mcard-num v">{k}</div><div class="mcard-lbl">Clusters</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="mcard b"><div class="mcard-num b">{max_iter}</div><div class="mcard-lbl">Max Iterations</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="mcard c"><div class="mcard-num c">{round(model.inertia_, 1)}</div><div class="mcard-lbl">Inertia</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="mcard g"><div class="mcard-num g">{len(X_iris)}</div><div class="mcard-lbl">Data Points</div></div>', unsafe_allow_html=True)


# ─── PCA ──────────────────────────────────────────────────
elif algo_key == "pca":

    col_ctrl, col_main = st.columns([1, 2], gap="large")

    with col_ctrl:
        st.markdown('<div class="slabel">Controls</div>', unsafe_allow_html=True)
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        n_comp = st.slider("Number of Components", 2, 4, 2)
        use_all_features = st.toggle("Use all 4 Iris features", value=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="slabel" style="margin-top:10px;">What PCA does</div>', unsafe_allow_html=True)
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        pca_steps = [
            ("dv", "Centre the data (subtract mean)"),
            ("db", "Compute the covariance matrix"),
            ("dc", "Find eigenvectors & eigenvalues"),
            ("dg", "Sort by explained variance (desc.)"),
            ("da", "Project data onto top K eigenvectors"),
        ]
        for dot, text in pca_steps:
            st.markdown(f'<div class="insight"><span class="dot {dot}"></span><span>{text}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_main:
        X_input = X_iris_full if use_all_features else X_iris
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_input)

        pca = PCA(n_components=min(n_comp, X_input.shape[1]))
        X_pca = pca.fit_transform(X_scaled)
        ev = pca.explained_variance_ratio_

        pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
        pca_df['Species'] = y_iris.map({0:'Setosa',1:'Versicolor',2:'Virginica'})

        st.markdown('<div class="slabel">PCA Projection — Iris Species</div>', unsafe_allow_html=True)
        fig_pca = px.scatter(
            pca_df, x='PC1', y='PC2', color='Species',
            color_discrete_sequence=px.colors.qualitative.Vivid,
            labels={'PC1': f'PC1 ({ev[0]*100:.1f}% var)', 'PC2': f'PC2 ({ev[1]*100:.1f}% var)'}
        )
        fig_pca.update_traces(marker=dict(size=8, opacity=0.85, line=dict(color='white', width=0.4)))
        fig_pca.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,17,32,1)',
            height=340, margin=dict(t=10, b=20, l=10, r=10),
            xaxis=dict(gridcolor='#1e293b', color='#64748b'),
            yaxis=dict(gridcolor='#1e293b', color='#64748b'),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8')),
            font=dict(family='DM Sans')
        )
        st.plotly_chart(fig_pca, use_container_width=True)

        # Scree
        st.markdown('<div class="slabel">Scree Plot — Explained Variance per Component</div>', unsafe_allow_html=True)
        pca_full = PCA()
        pca_full.fit(X_scaled)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)

        fig_scree = go.Figure()
        fig_scree.add_trace(go.Bar(
            x=[f'PC{i+1}' for i in range(len(pca_full.explained_variance_ratio_))],
            y=pca_full.explained_variance_ratio_ * 100,
            name='Individual', marker_color='#8b5cf6', opacity=0.8
        ))
        fig_scree.add_trace(go.Scatter(
            x=[f'PC{i+1}' for i in range(len(cumvar))],
            y=cumvar * 100, mode='lines+markers',
            name='Cumulative', line=dict(color='#06b6d4', width=2),
            marker=dict(color='#67e8f9', size=7)
        ))
        fig_scree.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,17,32,1)',
            height=240, margin=dict(t=10, b=20, l=10, r=10),
            xaxis=dict(gridcolor='#1e293b', color='#64748b'),
            yaxis=dict(title='Variance Explained (%)', gridcolor='#1e293b', color='#64748b'),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8')),
            font=dict(family='DM Sans')
        )
        st.plotly_chart(fig_scree, use_container_width=True)

    # Metrics
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    mc = st.columns(4)
    for i, col in enumerate(mc):
        if i < len(ev):
            col.markdown(f'<div class="mcard v"><div class="mcard-num v">{ev[i]*100:.1f}%</div><div class="mcard-lbl">PC{i+1} Variance</div></div>', unsafe_allow_html=True)
        else:
            col.markdown(f'<div class="mcard b"><div class="mcard-num b">—</div><div class="mcard-lbl">PC{i+1}</div></div>', unsafe_allow_html=True)


# ─── LINEAR REGRESSION ────────────────────────────────────
elif algo_key == "linreg":

    # Anscombe's Quartet
    x_vals = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
    y1 = [8.04,6.95,7.58,8.81,8.33,9.96,7.24,4.26,10.84,4.82,5.68]
    y2 = [9.14,8.14,8.74,8.77,9.26,8.10,6.13,3.10,9.13,7.26,4.74]
    y3 = [7.46,6.77,12.74,7.11,7.81,8.84,6.08,5.39,8.15,6.42,5.73]
    anscombe = pd.DataFrame({'x': x_vals, 'y1': y1, 'y2': y2, 'y3': y3})

    col_ctrl, col_main = st.columns([1, 2], gap="large")

    with col_ctrl:
        st.markdown('<div class="slabel">Controls</div>', unsafe_allow_html=True)
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        series = st.selectbox("Anscombe Series", ["y1 — Linear", "y2 — Curved", "y3 — Outlier"])
        noise = st.slider("Add Noise σ", 0.0, 3.0, 0.0, 0.1)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="slabel" style="margin-top:10px;">Formula</div>', unsafe_allow_html=True)
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('<div class="formula">ŷ = β₀ + β₁x₁ + β₂x₂ + … + βₙxₙ</div>', unsafe_allow_html=True)
        st.markdown('<div style="color:#475569;font-size:.82rem;margin-top:8px;">Minimises the sum of squared residuals between predicted and actual values.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_main:
        y_col = series.split(' ')[0]
        x_data = np.array(x_vals)
        y_data = np.array(anscombe[y_col]) + np.random.normal(0, noise, len(x_vals))

        # fit
        lr = LinearRegression()
        lr.fit(x_data.reshape(-1,1), y_data)
        x_line = np.linspace(3, 15, 100)
        y_line = lr.predict(x_line.reshape(-1,1))

        slope    = lr.coef_[0]
        intercept = lr.intercept_
        r2 = lr.score(x_data.reshape(-1,1), y_data)

        st.markdown(f'<div class="slabel">Regression Line — Anscombe Series {y_col}</div>', unsafe_allow_html=True)
        fig_lr = go.Figure()
        fig_lr.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers', name='Data',
            marker=dict(color='#8b5cf6', size=10, opacity=0.9, line=dict(color='white', width=0.5))))
        fig_lr.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name='Regression Line',
            line=dict(color='#06b6d4', width=2.5)))
        # Residuals
        y_pred = lr.predict(x_data.reshape(-1,1))
        for xi, yi, ypi in zip(x_data, y_data, y_pred):
            fig_lr.add_shape(type='line', x0=xi, y0=yi, x1=xi, y1=ypi,
                line=dict(color='rgba(239,68,68,0.35)', width=1.5, dash='dot'))
        fig_lr.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,17,32,1)',
            height=360, margin=dict(t=10, b=20, l=10, r=10),
            xaxis=dict(title='x', gridcolor='#1e293b', color='#64748b'),
            yaxis=dict(title='y', gridcolor='#1e293b', color='#64748b'),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8')),
            font=dict(family='DM Sans'),
            annotations=[dict(
                x=0.02, y=0.97, xref='paper', yref='paper',
                text=f'ŷ = {slope:.2f}x + {intercept:.2f}  |  R² = {r2:.3f}',
                showarrow=False, font=dict(color='#67e8f9', size=12, family='DM Sans'),
                align='left'
            )]
        )
        st.plotly_chart(fig_lr, use_container_width=True)

    # Metrics
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    residuals = y_data - y_pred
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals**2))
    with m1:
        st.markdown(f'<div class="mcard v"><div class="mcard-num v">{slope:.3f}</div><div class="mcard-lbl">Slope β₁</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="mcard b"><div class="mcard-num b">{intercept:.3f}</div><div class="mcard-lbl">Intercept β₀</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="mcard c"><div class="mcard-num c">{r2:.3f}</div><div class="mcard-lbl">R² Score</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="mcard a"><div class="mcard-num a">{mae:.3f}</div><div class="mcard-lbl">MAE</div></div>', unsafe_allow_html=True)


# ─── LOGISTIC REGRESSION ──────────────────────────────────
elif algo_key == "logreg":

    col_ctrl, col_main = st.columns([1, 2], gap="large")

    with col_ctrl:
        st.markdown('<div class="slabel">Controls</div>', unsafe_allow_html=True)
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        C_val = st.select_slider("Regularisation C", options=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0], value=1.0)
        test_size = st.slider("Test Split %", 10, 40, 20)
        seed = st.slider("Random Seed", 0, 99, 42)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="slabel" style="margin-top:10px;">Sigmoid Function</div>', unsafe_allow_html=True)
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('<div class="formula">σ(z) = 1 / (1 + e⁻ᶻ)</div>', unsafe_allow_html=True)
        st.markdown('<div style="color:#475569;font-size:.82rem;margin-top:8px;">Squashes any real number into [0, 1] — interpreted as a class probability.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_main:
        # Sigmoid plot
        st.markdown('<div class="slabel">Sigmoid Curve</div>', unsafe_allow_html=True)
        z = np.linspace(-8, 8, 300)
        sig = 1 / (1 + np.exp(-z))
        fig_sig = go.Figure()
        fig_sig.add_trace(go.Scatter(x=z, y=sig, mode='lines', line=dict(color='#f59e0b', width=2.5), name='σ(z)'))
        fig_sig.add_hline(y=0.5, line=dict(color='rgba(255,255,255,0.2)', width=1, dash='dash'))
        fig_sig.add_vline(x=0,   line=dict(color='rgba(255,255,255,0.2)', width=1, dash='dash'))
        fig_sig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,17,32,1)',
            height=220, margin=dict(t=10, b=10, l=10, r=10),
            xaxis=dict(title='z', gridcolor='#1e293b', color='#64748b'),
            yaxis=dict(title='Probability', gridcolor='#1e293b', color='#64748b', range=[-0.05, 1.05]),
            font=dict(family='DM Sans'),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8')),
        )
        st.plotly_chart(fig_sig, use_container_width=True)

        # Iris binary classification (Setosa vs rest)
        st.markdown('<div class="slabel">Iris Classification (Setosa vs Others)</div>', unsafe_allow_html=True)
        X_lr = X_iris_full.values
        y_bin = (y_iris == 0).astype(int)
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X_lr)
        X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y_bin, test_size=test_size/100, random_state=seed)
        log_model = LogisticRegression(C=C_val, max_iter=1000)
        log_model.fit(X_tr, y_tr)
        train_acc = log_model.score(X_tr, y_tr)
        test_acc  = log_model.score(X_te, y_te)

        # PCA to 2D for visualization
        pca2 = PCA(n_components=2)
        X_vis = pca2.fit_transform(X_sc)
        y_pred_all = log_model.predict(X_sc)

        fig_lc = px.scatter(
            x=X_vis[:,0], y=X_vis[:,1], color=y_pred_all.astype(str),
            color_discrete_map={'0':'#8b5cf6','1':'#22c55e'},
            labels={'x':'PC1','y':'PC2','color':'Predicted'},
            symbol=y_bin.astype(str),
        )
        fig_lc.update_traces(marker=dict(size=8, opacity=0.85, line=dict(color='white', width=0.4)))
        fig_lc.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,17,32,1)',
            height=260, margin=dict(t=10, b=10, l=10, r=10),
            xaxis=dict(gridcolor='#1e293b', color='#64748b'),
            yaxis=dict(gridcolor='#1e293b', color='#64748b'),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8')),
            font=dict(family='DM Sans')
        )
        st.plotly_chart(fig_lc, use_container_width=True)

    # Metrics
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="mcard v"><div class="mcard-num v">{train_acc*100:.1f}%</div><div class="mcard-lbl">Train Accuracy</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="mcard g"><div class="mcard-num g">{test_acc*100:.1f}%</div><div class="mcard-lbl">Test Accuracy</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="mcard b"><div class="mcard-num b">{C_val}</div><div class="mcard-lbl">Regularisation C</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="mcard c"><div class="mcard-num c">{len(X_te)}</div><div class="mcard-lbl">Test Samples</div></div>', unsafe_allow_html=True)


# ─── DECISION TREES ───────────────────────────────────────
elif algo_key == "dtree":

    col_ctrl, col_main = st.columns([1, 2], gap="large")

    with col_ctrl:
        st.markdown('<div class="slabel">Controls</div>', unsafe_allow_html=True)
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        max_depth = st.slider("Max Tree Depth", 1, 15, 4)
        test_pct  = st.slider("Test Split %", 10, 40, 25)
        seed2     = st.slider("Random Seed", 0, 99, 42)
        show_tree = st.toggle("Show Tree Diagram", value=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="slabel" style="margin-top:10px;">Overfitting Guide</div>', unsafe_allow_html=True)
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        guides = [
            ("dr", "Depth 1–2: Underfitting — too simple"),
            ("da", "Depth 3–6: Good generalisation zone"),
            ("dg", "Depth 7–10: Watch test score carefully"),
            ("dr", "Depth 11+: Likely overfitting"),
        ]
        for dot, text in guides:
            st.markdown(f'<div class="insight"><span class="dot {dot}"></span><span>{text}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_main:
        X_dt = X_iris_full.values
        y_dt = y_iris.values
        X_tr, X_te, y_tr, y_te = train_test_split(X_dt, y_dt, test_size=test_pct/100, random_state=seed2)

        # Depth sweep
        train_scores, test_scores = [], []
        depths = list(range(1, 16))
        for d in depths:
            dt = DecisionTreeClassifier(max_depth=d, random_state=seed2)
            dt.fit(X_tr, y_tr)
            train_scores.append(dt.score(X_tr, y_tr))
            test_scores.append(dt.score(X_te, y_te))

        st.markdown('<div class="slabel">Train vs Test Accuracy — Depth Sweep</div>', unsafe_allow_html=True)
        fig_sweep = go.Figure()
        fig_sweep.add_trace(go.Scatter(x=depths, y=train_scores, mode='lines+markers', name='Train',
            line=dict(color='#3b82f6', width=2), marker=dict(color='#93c5fd', size=6)))
        fig_sweep.add_trace(go.Scatter(x=depths, y=test_scores, mode='lines+markers', name='Test',
            line=dict(color='#22c55e', width=2), marker=dict(color='#86efac', size=6)))
        fig_sweep.add_vline(x=max_depth, line=dict(color='#f59e0b', width=2, dash='dash'))
        fig_sweep.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,17,32,1)',
            height=300, margin=dict(t=10, b=20, l=10, r=10),
            xaxis=dict(title='Max Depth', gridcolor='#1e293b', color='#64748b'),
            yaxis=dict(title='Accuracy', gridcolor='#1e293b', color='#64748b', range=[0.7, 1.02]),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8')),
            font=dict(family='DM Sans')
        )
        st.plotly_chart(fig_sweep, use_container_width=True)

        # Fit chosen tree
        chosen_dt = DecisionTreeClassifier(max_depth=max_depth, random_state=seed2)
        chosen_dt.fit(X_tr, y_tr)
        tr_acc = chosen_dt.score(X_tr, y_tr)
        te_acc = chosen_dt.score(X_te, y_te)

        if show_tree:
            st.markdown('<div class="slabel">Tree Diagram</div>', unsafe_allow_html=True)
            fig_tree, ax = plt.subplots(figsize=(12, max(3, max_depth * 1.4)))
            fig_tree.patch.set_facecolor('#0c1020')
            ax.set_facecolor('#0c1020')
            plot_tree(chosen_dt, ax=ax,
                      feature_names=['sepal len','sepal wid','petal len','petal wid'],
                      class_names=['Setosa','Versicolor','Virginica'],
                      filled=True, fontsize=8, rounded=True)
            st.pyplot(fig_tree, use_container_width=True)
            plt.close(fig_tree)

    # Metrics
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="mcard b"><div class="mcard-num b">{tr_acc*100:.1f}%</div><div class="mcard-lbl">Train Accuracy</div></div>', unsafe_allow_html=True)
    with m2:
        clr = "g" if te_acc >= 0.9 else ("a" if te_acc >= 0.8 else "r")
        st.markdown(f'<div class="mcard {clr}"><div class="mcard-num {clr}">{te_acc*100:.1f}%</div><div class="mcard-lbl">Test Accuracy</div></div>', unsafe_allow_html=True)
    with m3:
        ovf = "Overfitting ⚠" if tr_acc - te_acc > 0.08 else "Healthy ✓"
        clr2 = "a" if tr_acc - te_acc > 0.08 else "g"
        st.markdown(f'<div class="mcard {clr2}"><div class="mcard-num {clr2}" style="font-size:1.1rem;">{ovf}</div><div class="mcard-lbl">Fit Status</div></div>', unsafe_allow_html=True)
    with m4:
        n_leaves = chosen_dt.get_n_leaves()
        st.markdown(f'<div class="mcard v"><div class="mcard-num v">{n_leaves}</div><div class="mcard-lbl">Leaf Nodes</div></div>', unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;padding:10px 0 18px;color:#334155;font-size:.76rem;letter-spacing:1px;'>
ML ALGORITHM EXPLORER &nbsp;·&nbsp; scikit-learn + Plotly + Streamlit &nbsp;·&nbsp; Sahil Gaund
</div>
""", unsafe_allow_html=True)

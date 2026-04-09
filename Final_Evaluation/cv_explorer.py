"""
cv_explorer.py  –  Interactive hyperparameter CV-score explorer
Run:  streamlit run cv_explorer.py
python -m streamlit run cv_explorer.py

On Colab:
    !pip install streamlit plotly pyngrok -q
    from pyngrok import ngrok
    import subprocess, threading
    threading.Thread(target=lambda: subprocess.run(["streamlit","run","cv_explorer.py","--server.port","8501"])).start()
    print(ngrok.connect(8501))
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR = os.path.join(_HERE, '..', 'Results', 'metrics')

MODEL_FILES = {
    'KNN': 'KNN_cv_results.csv',
    'DT':  'DT_cv_results.csv',
    'RF':  'RF_cv_results.csv',
    'NB':  'NB_cv_results.csv',
    'SVM': 'SVM_cv_results.csv',
    'ANN': 'ANN_cv_results.csv',
}

TASK_OPTIONS = {
    'Binary / Raw':          ('Binary',     'Raw'),
    'Binary / Reduced':      ('Binary',     'Reduced'),
    'Multiclass / Raw':      ('Multiclass', 'Raw'),
    'Multiclass / Reduced':  ('Multiclass', 'Reduced'),
}

# Params that span orders of magnitude → log x-axis
LOG_SCALE_PARAMS = {'var_smoothing', 'lr', 'weight_decay', 'C'}

# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_data
def load_csv(model: str) -> pd.DataFrame:
    path = os.path.join(METRICS_DIR, MODEL_FILES[model])
    df = pd.read_csv(path)
    # Normalise NaN class_weight → string 'None'
    for col in df.columns:
        if 'class_weight' in col:
            df[col] = df[col].fillna('None')
    return df


def param_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith('param_')]


def display_name(col: str) -> str:
    """Strip param_ prefix and simplify ANN param names."""
    n = col.removeprefix('param_')
    n = n.replace('net__module__', '').replace('net__optimizer__', '').replace('net__', '')
    return n


def sorted_vals(series: pd.Series) -> list:
    """Sort unique values: numbers numerically, strings alphabetically."""
    vals = series.dropna().unique().tolist()
    try:
        return sorted(vals, key=float)
    except (TypeError, ValueError):
        return sorted(vals, key=str)


def use_log(name: str) -> bool:
    return any(k in name for k in LOG_SCALE_PARAMS)


# ── Plot helpers ───────────────────────────────────────────────────────────────
def plot_1d(df: pd.DataFrame, x_col: str, title: str):
    name = display_name(x_col)
    agg = (df.groupby(x_col, sort=False)['mean_test_score']
             .agg(['mean', 'std'])
             .reset_index()
             .rename(columns={'mean': 'CV Score', 'std': 'std'}))
    agg = agg.sort_values(x_col, key=lambda s: s.map(lambda v: float(v) if str(v).replace('.','',1).replace('e-','',1).lstrip('-').isdigit() else str(v)))

    fig = px.line(
        agg, x=x_col, y='CV Score', error_y='std',
        markers=True,
        title=title,
        labels={x_col: name, 'CV Score': 'CV Score (balanced_acc)'},
    )
    fig.update_traces(marker_size=8)
    if use_log(name):
        fig.update_xaxes(type='log', title=name)
    fig.update_layout(yaxis_tickformat='.4f', height=450)
    st.plotly_chart(fig, use_container_width=True)


def plot_2d(df: pd.DataFrame, x_col: str, y_col: str, title: str):
    xname, yname = display_name(x_col), display_name(y_col)
    agg = (df.groupby([x_col, y_col], sort=False)['mean_test_score']
             .mean()
             .reset_index()
             .rename(columns={'mean_test_score': 'CV Score'}))

    # Heatmap
    try:
        pivot = agg.pivot(index=y_col, columns=x_col, values='CV Score')
        fig = px.imshow(
            pivot,
            title=title,
            labels={'color': 'CV Score', 'x': xname, 'y': yname},
            color_continuous_scale='RdYlGn',
            text_auto='.3f',
            aspect='auto',
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        # Fallback: scatter
        fig = px.scatter(agg, x=x_col, y=y_col, color='CV Score',
                         color_continuous_scale='RdYlGn',
                         title=title, labels={x_col: xname, y_col: yname})
        st.plotly_chart(fig, use_container_width=True)

    # Optional 3D surface
    with st.expander("Show 3D surface"):
        try:
            pivot = agg.pivot(index=y_col, columns=x_col, values='CV Score')
            fig3d = go.Figure(go.Surface(
                z=pivot.values,
                x=[str(v) for v in pivot.columns],
                y=[str(v) for v in pivot.index],
                colorscale='RdYlGn',
                showscale=True,
            ))
            fig3d.update_layout(
                title=title,
                scene=dict(
                    xaxis_title=xname,
                    yaxis_title=yname,
                    zaxis_title='CV Score',
                ),
                height=550,
            )
            st.plotly_chart(fig3d, use_container_width=True)
        except Exception as e:
            st.warning(f"3D plot failed: {e}")


# ── Main app ───────────────────────────────────────────────────────────────────
st.set_page_config(page_title='CV Explorer', layout='wide')
st.title('EEG ML Benchmark – Hyperparameter CV Explorer')

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header('Select')
    model      = st.selectbox('Model', list(MODEL_FILES.keys()))
    task_label = st.selectbox('Task / Features', list(TASK_OPTIONS.keys()))

task, features = TASK_OPTIONS[task_label]

# Load & filter by task
df_all = load_csv(model)
df = df_all[(df_all['Task'] == task) & (df_all['Features'] == features)].copy()

if df.empty:
    st.error(f"No data found for {model} / {task_label}.")
    st.stop()

pcols = param_cols(df)
st.caption(f"**{model}** · {task_label} · {len(df)} parameter combinations · "
           f"scoring: balanced_accuracy (5-fold CV)")

# ── NB: single hyperparameter, show directly ──────────────────────────────────
if model == 'NB':
    st.subheader(f'NB: {display_name(pcols[0])} vs CV Score')
    plot_1d(df, pcols[0], f'Naïve Bayes – {task_label}')
    best = df.loc[df['mean_test_score'].idxmax()]
    st.success(f"Best: {display_name(pcols[0])} = **{best[pcols[0]]}**  →  "
               f"CV Score = **{best['mean_test_score']:.4f}** ± {best['std_test_score']:.4f}")
    st.stop()

# ── Multi-param models ─────────────────────────────────────────────────────────
st.subheader('Fix Hyperparameters')
st.caption('Check a parameter to fix its value. Unfixed parameters become plot axes.')

fixed: dict[str, object] = {}

cols_ui = st.columns(min(len(pcols), 3))
for i, col in enumerate(pcols):
    name = display_name(col)
    vals = sorted_vals(df[col])
    with cols_ui[i % len(cols_ui)]:
        if st.checkbox(f'Fix **{name}**', key=f'fix_{col}'):
            chosen = st.selectbox(f'{name} =', vals, key=f'val_{col}')
            fixed[col] = chosen

# Apply fixed filters
df_plot = df.copy()
for col, val in fixed.items():
    df_plot = df_plot[df_plot[col].astype(str) == str(val)]

if df_plot.empty:
    st.warning('No data matches the selected filters.')
    st.stop()

free = [c for c in pcols if c not in fixed]

# ── Choose plot axes (when > 2 free params) ───────────────────────────────────
x_col = y_col = None

if len(free) == 0:
    score = df_plot['mean_test_score'].mean()
    std   = df_plot['std_test_score'].mean()
    st.metric('CV Score (all params fixed)', f'{score:.4f}', delta=f'±{std:.4f}')
    st.stop()

elif len(free) == 1:
    x_col = free[0]

elif len(free) == 2:
    x_col, y_col = free[0], free[1]

else:
    # 3+ free: let user pick axes; aggregate over the rest
    free_names = [display_name(c) for c in free]
    st.info(f'{len(free)} parameters are free. Choose up to 2 axes; '
            f'the rest will be averaged over.')
    name_to_col = {display_name(c): c for c in free}

    c1, c2 = st.columns(2)
    with c1:
        xn = st.selectbox('X axis', free_names, key='ax_x')
        x_col = name_to_col[xn]
    with c2:
        yn_opts = ['(none – 1D plot)'] + [n for n in free_names if n != xn]
        yn = st.selectbox('Y axis', yn_opts, key='ax_y')
        y_col = name_to_col[yn] if yn != '(none – 1D plot)' else None

# ── Draw plot ──────────────────────────────────────────────────────────────────
fixed_str = ', '.join(f'{display_name(k)}={v}' for k, v in fixed.items())
title_suffix = f'  [{fixed_str}]' if fixed_str else ''
title = f'{model} – {task_label}{title_suffix}'

st.divider()
st.subheader('Result')
st.caption(f"Showing mean CV score over {len(df_plot)} combinations"
           + (f" (averaged over free params not on axes)" if len(free) > 2 else ""))

if y_col is None:
    plot_1d(df_plot, x_col, title)
else:
    plot_2d(df_plot, x_col, y_col, title)

# ── Best combo summary ─────────────────────────────────────────────────────────
st.divider()
st.subheader('Best Combination (within current filter)')
best = df_plot.loc[df_plot['mean_test_score'].idxmax()]
cols_show = pcols + ['mean_test_score', 'std_test_score']
best_df = best[cols_show].to_frame().T
best_df.columns = [display_name(c) if c.startswith('param_') else c for c in best_df.columns]
best_df = best_df.rename(columns={'mean_test_score': 'CV Score', 'std_test_score': 'Std'})
st.dataframe(best_df.reset_index(drop=True), use_container_width=True)

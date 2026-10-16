"""
Streamlit Quantitative Finance Application (Clean White Theme)
BSE SENSEX Dynamic Trajectory vs Geometric Brownian Motion (GBM)
Palette: Pure White, Emerald Jade (#059669), and Royal Violet (#7c3aed)
"""

import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import plotly.graph_objects as go
import quant_models as qm

# Configure page layout and light styling
st.set_page_config(
    page_title="BSE SENSEX Dynamic Trajectory | Quantitative Model",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Clean White aesthetic
st.markdown("""
<style>
    .reportview-container, .main, header, footer {
        background-color: #f8fafc;
        color: #0f172a;
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 16px 20px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04);
        margin-bottom: 12px;
        border-left: 4px solid #059669;
    }
    .metric-title {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-sub {
        font-size: 0.78rem;
        margin-top: 4px;
        color: #64748b;
    }
    .text-emerald { color: #059669 !important; }
    .text-violet { color: #7c3aed !important; }
    .text-blue { color: #2563eb !important; }
    .text-crimson { color: #e11d48 !important; }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0px 0px;
        padding: 10px 18px;
        background-color: #ffffff;
        color: #64748b;
        border: 1px solid #e2e8f0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f1f5f9 !important;
        color: #059669 !important;
        border-bottom: 2px solid #059669 !important;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar settings
with st.sidebar:
    st.title("📈 Trajectory Engine")
    st.caption("Stochastic Diffusion vs Empirical SENSEX Dynamics")
    
    ticker_choice = st.selectbox(
        "Index / Asset",
        options=["^BSESN", "^NSEI", "^GSPC"],
        format_func=lambda x: {
            "^BSESN": "BSE SENSEX (India)",
            "^NSEI": "NIFTY 50 (India)",
            "^GSPC": "S&P 500 (US)"
        }.get(x, x),
        index=0
    )
    
    n_sims = st.slider("GBM Monte Carlo Paths", min_value=50, max_value=500, value=150, step=25)
    ci_level = st.select_slider("Diffusion Cone (CI %)", options=[80, 90, 95, 99], value=90)
    
    st.markdown("---")
    st.markdown("### Modeling Extensions")
    show_student_t = st.checkbox("Student-t Fit (Power-Law Tail)", value=True)
    show_garch = st.checkbox("GARCH(1,1) Volatility Standardized", value=True)
    scale_log = st.checkbox("Logarithmic Scale", value=True)
    
    st.markdown("---")
    st.caption("Empirical Quantitative Research on Indian Equity Markets & Stochastic Modeling.")

# Data loading
@st.cache_data(ttl=3600)
def load_data(ticker):
    return qm.fetch_index_data(ticker=ticker)

try:
    df = load_data(ticker_choice)
except Exception as e:
    st.error(f"Error loading index data: {e}")
    st.stop()

returns = df["Log_Return"]
price = df["Normalized_Price"]
dates = df.index
summary = qm.calculate_summary_stats(returns, price)
cond_vol, std_resid, garch_kurt = qm.fit_garch_model(returns)

# Header
st.title("📈 BSE SENSEX: Dynamic Growth & Stochastic Diffusion")
st.markdown(
    f"Evaluating **{summary['n_days']:,}** daily observations "
    f"({dates[0].strftime('%b %Y')} – {dates[-1].strftime('%b %Y')}) "
    "against Geometric Brownian Motion (GBM) diffusion, Student-t MLE, and GARCH(1,1)."
)

# Live Metric Strip
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #059669;">
        <div class="metric-title">Compounded Growth (CAGR)</div>
        <div class="metric-value text-emerald">{summary['ann_geo_return']*100:.2f}%</div>
        <div class="metric-sub text-violet">Base 100 to {price.iloc[-1]:,.0f} pts</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #7c3aed;">
        <div class="metric-title">Annualized Volatility (σ)</div>
        <div class="metric-value text-violet">{summary['ann_vol']*100:.2f}%</div>
        <div class="metric-sub text-blue">Daily σ: {summary['daily_std']*100:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #2563eb;">
        <div class="metric-title">Excess Kurtosis (Leptokurtic)</div>
        <div class="metric-value text-blue">{summary['excess_kurtosis']:.2f}</div>
        <div class="metric-sub text-emerald">GARCH Standardized: {garch_kurt:.2f} (-57%)</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #e11d48;">
        <div class="metric-title">4-Sigma Black Swan Days</div>
        <div class="metric-value text-crimson">{summary['outliers_4s']} <span style="font-size:1rem;color:#64748b;">days</span></div>
        <div class="metric-sub text-violet">Gaussian Expects: {summary['expected_4s']:.2f} days</div>
    </div>
    """, unsafe_allow_html=True)

# Tabs
tab_traj, tab_dist, tab_qq, tab_history = st.tabs([
    "Compounded Trajectory & Cone",
    "Return Density & Leptokurtosis",
    "Q-Q Normality Diagnostics",
    "Historical Black Swan Days"
])

# 1. Trajectory
with tab_traj:
    half_alpha = (100 - ci_level) / 2.0
    lower_pct = half_alpha
    upper_pct = 100 - half_alpha
    
    paths, p_low, p_med, p_high = qm.simulate_gbm_ensemble(
        s0=price.iloc[0],
        drift=summary["ann_geo_return"],
        vol=summary["ann_vol"],
        n_steps=len(df) - 1,
        n_paths=n_sims,
        seed=101
    )
    
    p_low = np.percentile(paths, lower_pct, axis=0)
    p_high = np.percentile(paths, upper_pct, axis=0)
    
    fig_traj = go.Figure()
    
    for i in range(min(30, len(paths))):
        fig_traj.add_trace(go.Scatter(
            x=dates, y=paths[i],
            mode='lines',
            line=dict(color='rgba(124, 58, 237, 0.08)', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))
        
    fig_traj.add_trace(go.Scatter(
        x=dates, y=p_high,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    fig_traj.add_trace(go.Scatter(
        x=dates, y=p_low,
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(124, 58, 237, 0.14)',
        name=f'GBM {lower_pct:.0f}–{upper_pct:.0f}% Diffusion Cone'
    ))
    
    fig_traj.add_trace(go.Scatter(
        x=dates, y=p_med,
        mode='lines',
        line=dict(color='#7c3aed', width=2, dash='dash'),
        name='GBM Median Drift Path'
    ))
    
    fig_traj.add_trace(go.Scatter(
        x=dates, y=price,
        mode='lines',
        line=dict(color='#059669', width=2.6),
        name=f'Realized {ticker_choice} (Base 100)'
    ))
    
    fig_traj.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        height=540,
        margin=dict(l=50, r=30, t=30, b=40),
        yaxis=dict(
            type="log" if scale_log else "linear",
            gridcolor="#f1f5f9",
            title="Index Level (Base 100)"
        ),
        xaxis=dict(gridcolor="#f1f5f9", title="Timeline"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.95)"
        )
    )
    st.plotly_chart(fig_traj, use_container_width=True)

# 2. Return Distribution
with tab_dist:
    ret_pct = returns * 100.0
    x_dense = np.linspace(-6.5, 6.5, 800)
    pdf_norm = stats.norm.pdf(x_dense, loc=ret_pct.mean(), scale=ret_pct.std())
    t_df = summary["student_t_df"]
    pdf_t = stats.t.pdf(x_dense, df=t_df, loc=summary["student_t_loc"] * 100.0, scale=summary["student_t_scale"] * 100.0)
    
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=ret_pct,
        histnorm='probability density',
        nbinsx=90,
        marker_color='rgba(5, 150, 105, 0.70)',
        name='Realized Daily Returns'
    ))
    fig_dist.add_trace(go.Scatter(
        x=x_dense, y=pdf_norm,
        mode='lines',
        line=dict(color='#7c3aed', width=2.2),
        name='Gaussian Normal (GBM)'
    ))
    if show_student_t:
        fig_dist.add_trace(go.Scatter(
            x=x_dense, y=pdf_t,
            mode='lines',
            line=dict(color='#0284c7', width=2.2),
            name=f'Student-t Fit (ν = {t_df:.2f})'
        ))
        
    fig_dist.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        height=480,
        xaxis=dict(range=[-6.2, 6.2], gridcolor="#f1f5f9", title="Daily Log Return (%)"),
        yaxis=dict(gridcolor="#f1f5f9", title="Probability Density"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_dist, use_container_width=True)

# 3. Q-Q Plot
with tab_qq:
    z_act = (returns - returns.mean()) / returns.std()
    osm_a, osr_a = stats.probplot(z_act, dist="norm", fit=False)
    
    fig_qq = go.Figure()
    q_x = np.linspace(-4.5, 4.5, 150)
    fig_qq.add_trace(go.Scatter(
        x=q_x, y=q_x,
        mode='lines',
        line=dict(color='#94a3b8', width=1.5, dash='dash'),
        name='Normal Reference (45°)'
    ))
    fig_qq.add_trace(go.Scatter(
        x=osm_a[::2], y=osr_a[::2],
        mode='markers',
        marker=dict(color='#059669', size=5, opacity=0.8),
        name='Realized SENSEX'
    ))
    if show_garch:
        z_g = std_resid.dropna()
        osm_g, osr_g = stats.probplot(z_g, dist="norm", fit=False)
        fig_qq.add_trace(go.Scatter(
            x=osm_g[::2], y=osr_g[::2],
            mode='markers',
            marker=dict(color='#e11d48', size=5, opacity=0.6),
            name=f'GARCH Standardized (Kurt: {garch_kurt:.2f})'
        ))
        
    fig_qq.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        height=500,
        xaxis=dict(range=[-4.2, 4.2], gridcolor="#f1f5f9", title="Theoretical Normal Quantiles (σ)"),
        yaxis=dict(range=[-15.0, 9.0], gridcolor="#f1f5f9", title="Sample Quantiles (σ)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_qq, use_container_width=True)

# 4. Black Swan Table
with tab_history:
    st.markdown("### Historical Outlier Days Beyond $\\pm 4\\sigma$")
    z_all = (returns - returns.mean()) / returns.std()
    extreme_mask = np.abs(z_all) >= 4.0
    extreme_df = pd.DataFrame({
        "Date": dates[extreme_mask].strftime("%Y-%m-%d"),
        "Daily Return (%)": (returns[extreme_mask] * 100.0).round(2),
        "Z-Score (Sigma)": z_all[extreme_mask].round(2),
        "Event Type": np.where(returns[extreme_mask] > 0, "🟢 Surge", "🔴 Crash")
    }).sort_values(by="Daily Return (%)")
    
    st.dataframe(extreme_df, use_container_width=True, hide_index=True)

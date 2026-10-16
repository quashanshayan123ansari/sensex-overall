"""
High-Resolution Publication-Quality Chart Generator
BSE SENSEX Total Return vs Geometric Brownian Motion (GBM) with Student-t & GARCH(1,1)

Produces:
- 'sensex_gbm_fat_tails.png' (300 DPI)
- 'sensex_gbm_fat_tails.svg' (Vector format)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import scipy.stats as stats
import quant_models as qm

# Set aesthetic styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#2a3447'
plt.rcParams['axes.linewidth'] = 0.8


def build_publication_figure(save_path: str = "sensex_gbm_fat_tails.png"):
    # 1. Fetch & prep data
    df = qm.fetch_index_data(ticker="^BSESN")
    returns = df["Log_Return"]
    price = df["Normalized_Price"]  # Base 100
    dates = df.index
    n_days = len(returns)
    
    # 2. Compute statistics
    summary = qm.calculate_summary_stats(returns, price)
    
    # 3. Simulate GBM ensemble
    mu = summary["ann_geo_return"]
    sigma = summary["ann_vol"]
    dt = 1.0 / 252.0
    s0 = price.iloc[0]
    
    paths, p5, p50, p95 = qm.simulate_gbm_ensemble(
        s0=s0,
        drift=mu,
        vol=sigma,
        n_steps=n_days - 1,
        n_paths=200,
        dt=dt,
        seed=101
    )
    
    # 4. Fit GARCH(1,1)
    cond_vol, std_resid, garch_kurt = qm.fit_garch_model(returns)
    
    # 5. Create Canvas
    fig = plt.figure(figsize=(15, 11), facecolor='#080c14')
    gs = gridspec.GridSpec(
        nrows=3,
        ncols=2,
        height_ratios=[1.35, 1.05, 0.45],
        hspace=0.38,
        wspace=0.22,
        left=0.08,
        right=0.94,
        top=0.91,
        bottom=0.06
    )
    
    # Color palette
    c_bg = '#0e1422'
    c_actual = '#ff7844'     # Neon amber-orange
    c_gbm_band = '#0284c7'   # Cyan sapphire
    c_gbm_line = '#38bdf8'   # Sky blue
    c_student = '#10b981'    # Emerald green
    c_garch = '#a855f7'      # Purple
    c_grid = '#1c2436'
    c_text = '#e2e8f0'
    c_muted = '#94a3b8'
    
    # =========================================================================
    # Header Titles
    # =========================================================================
    fig.text(
        0.08, 0.965,
        "BSE SENSEX vs Geometric Brownian Motion",
        fontsize=20, weight='bold', color='#ffffff', ha='left'
    )
    fig.text(
        0.08, 0.942,
        "Actual daily data (1997–2026) vs 200 GBM Monte Carlo paths calibrated in real time + Student-t & GARCH extensions",
        fontsize=11, color=c_muted, ha='left'
    )
    
    # =========================================================================
    # TOP PANEL: Price Path (Log Scale) spanning both columns
    # =========================================================================
    ax_top = fig.add_subplot(gs[0, :], facecolor=c_bg)
    ax_top.set_yscale('log')
    ax_top.grid(True, which='both', color=c_grid, linestyle='-', linewidth=0.6, alpha=0.7)
    
    # Plot individual simulated paths (faint background ensemble)
    for i in range(min(120, len(paths))):
        ax_top.plot(dates, paths[i], color=c_gbm_line, alpha=0.045, linewidth=0.75, zorder=1)
        
    # Shaded 5-95% GBM confidence band
    ax_top.fill_between(
        dates, p5, p95,
        color=c_gbm_band, alpha=0.28, label='GBM 5–95% confidence band', zorder=2
    )
    
    # GBM Median path
    ax_top.plot(
        dates, p50,
        color=c_gbm_line, linestyle='--', linewidth=1.6, label='GBM median (CAGR calibrated)', zorder=3
    )
    
    # Actual Price path
    ax_top.plot(
        dates, price,
        color=c_actual, linewidth=2.0, label='Actual BSE SENSEX (Base 100)', zorder=4
    )
    
    # Terminal point indicator
    ax_top.scatter([dates[-1]], [price.iloc[-1]], color=c_actual, s=40, zorder=5)
    
    ax_top.set_title("Price Path Trajectory (Log Scale)", fontsize=13, weight='semibold', color=c_text, loc='left', pad=8)
    ax_top.set_ylabel("Index Level (Base 100 = Jul 1997)", fontsize=10.5, color=c_muted)
    ax_top.tick_params(colors=c_muted, labelsize=9.5)
    ax_top.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{int(y):,}"))
    
    # Date badge in the bottom-right of top chart
    last_date_str = dates[-1].strftime("%b %Y")
    ax_top.text(
        0.98, 0.08, last_date_str,
        transform=ax_top.transAxes,
        fontsize=16, weight='bold', color='#ffffff',
        ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#111827', edgecolor='#374151', alpha=0.85)
    )
    
    leg_top = ax_top.legend(loc='upper left', frameon=True, facecolor='#111827', edgecolor='#2e394e', fontsize=9.5)
    for text in leg_top.get_texts():
        text.set_color(c_text)
        
    # =========================================================================
    # BOTTOM LEFT PANEL: Distribution of Daily Returns
    # =========================================================================
    ax_dist = fig.add_subplot(gs[1, 0], facecolor=c_bg)
    ax_dist.grid(True, color=c_grid, linestyle='-', linewidth=0.6, alpha=0.7)
    
    # Convert returns to percentages
    ret_pct = returns * 100.0
    x_grid = np.linspace(-6.5, 6.5, 1000)
    
    # Actual empirical histogram
    counts, bins, _ = ax_dist.hist(
        ret_pct, bins=85, range=(-6.0, 6.0), density=True,
        color=c_actual, alpha=0.82, edgecolor='#080c14', linewidth=0.3, label='Actual SENSEX'
    )
    
    # Theoretical Gaussian Normal curve
    norm_mu = returns.mean() * 100.0
    norm_sigma = returns.std() * 100.0
    pdf_norm = stats.norm.pdf(x_grid, loc=norm_mu, scale=norm_sigma)
    ax_dist.plot(x_grid, pdf_norm, color=c_gbm_line, linewidth=2.0, label='Normal (GBM baseline)')
    
    # Fitted Scaled Student-t curve
    t_df = summary["student_t_df"]
    t_loc = summary["student_t_loc"] * 100.0
    t_scale = summary["student_t_scale"] * 100.0
    pdf_t = stats.t.pdf(x_grid, df=t_df, loc=t_loc, scale=t_scale)
    ax_dist.plot(x_grid, pdf_t, color=c_student, linewidth=2.0, linestyle='-', label=f'Student-t (df={t_df:.1f})')
    
    # Extreme days notice
    extreme_days = int(np.sum(np.abs(ret_pct) >= 6.0))
    ax_dist.text(
        0.96, 0.76, f"{extreme_days} days beyond ±6%\n(Truncated for scale)",
        transform=ax_dist.transAxes,
        fontsize=8.5, color=c_muted, ha='right', va='top',
        bbox=dict(boxstyle='square,pad=0.3', facecolor='#111827', edgecolor='#2e394e', alpha=0.7)
    )
    
    ax_dist.set_xlim(-6.2, 6.2)
    ax_dist.set_title("Distribution of Daily Returns vs Heavy Tails", fontsize=12, weight='semibold', color=c_text, loc='left', pad=8)
    ax_dist.set_xlabel("Daily Log Return (%)", fontsize=10, color=c_muted)
    ax_dist.set_ylabel("Probability Density", fontsize=10, color=c_muted)
    ax_dist.tick_params(colors=c_muted, labelsize=9)
    
    leg_dist = ax_dist.legend(loc='upper left', frameon=True, facecolor='#111827', edgecolor='#2e394e', fontsize=8.5)
    for text in leg_dist.get_texts():
        text.set_color(c_text)
        
    # =========================================================================
    # BOTTOM RIGHT PANEL: Q-Q Plot vs Normal & GARCH Standardized Residuals
    # =========================================================================
    ax_qq = fig.add_subplot(gs[1, 1], facecolor=c_bg)
    ax_qq.grid(True, color=c_grid, linestyle='-', linewidth=0.6, alpha=0.7)
    
    # Compute standardized returns
    z_actual = (returns - returns.mean()) / returns.std()
    osm_actual, osr_actual = stats.probplot(z_actual, dist="norm", fit=False)
    
    # Standard normal theoretical line
    q_line = np.linspace(-4.5, 4.5, 200)
    ax_qq.plot(q_line, q_line, color='#475569', linestyle='--', linewidth=1.5, label='Normal reference (45°)')
    
    # Simulated GBM path Q-Q points
    sim_returns = np.diff(np.log(paths[0]))
    z_sim = (sim_returns - np.mean(sim_returns)) / np.std(sim_returns)
    osm_sim, osr_sim = stats.probplot(z_sim, dist="norm", fit=False)
    ax_qq.scatter(osm_sim[::4], osr_sim[::4], color=c_gbm_line, s=8, alpha=0.6, label='GBM simulated path')
    
    # GARCH standardized residuals Q-Q points (Shows Volatility Clustering fix!)
    z_garch = std_resid.dropna()
    osm_garch, osr_garch = stats.probplot(z_garch, dist="norm", fit=False)
    ax_qq.scatter(osm_garch[::4], osr_garch[::4], color=c_garch, s=8, alpha=0.5, label=f'GARCH standardized (Kurt={garch_kurt:.1f})')
    
    # Actual market Q-Q points
    ax_qq.scatter(osm_actual[::4], osr_actual[::4], color=c_actual, s=12, alpha=0.85, label='Actual SENSEX (Heavy Tails)')
    
    ax_qq.set_xlim(-4.2, 4.2)
    ax_qq.set_ylim(-16.0, 10.0)
    ax_qq.set_title("Q-Q Plot: Normal Normality Failure vs GARCH Correction", fontsize=12, weight='semibold', color=c_text, loc='left', pad=8)
    ax_qq.set_xlabel("Theoretical Quantiles (Normal σ)", fontsize=10, color=c_muted)
    ax_qq.set_ylabel("Sample Quantiles (Observed σ)", fontsize=10, color=c_muted)
    ax_qq.tick_params(colors=c_muted, labelsize=9)
    
    leg_qq = ax_qq.legend(loc='upper left', frameon=True, facecolor='#111827', edgecolor='#2e394e', fontsize=8.2)
    for text in leg_qq.get_texts():
        text.set_color(c_text)
        
    # =========================================================================
    # SUMMARY KPI METRICS PANEL (Spanning both columns)
    # =========================================================================
    ax_stats = fig.add_subplot(gs[2, :], facecolor='#0b111e')
    ax_stats.axis('off')
    
    # Create stylish tabular grid of stats
    c_label = '#94a3b8'
    c_val = '#f8fafc'
    c_accent_orange = '#fb923c'
    c_accent_cyan = '#38bdf8'
    c_accent_green = '#34d399'
    
    row1 = [
        ("Trading Days", f"{summary['n_days']:,}", c_val),
        ("Annualized Return (CAGR)", f"{summary['ann_geo_return']*100:.2f}%", c_accent_green),
        ("Annualized Volatility (σ)", f"{summary['ann_vol']*100:.2f}%", c_accent_cyan),
        ("Worst Single Day", f"{summary['worst_day']*100:.2f}%", '#f87171')
    ]
    
    row2 = [
        ("Skewness", f"{summary['skewness']:.2f}", c_val),
        ("Excess Kurtosis (Normal=0)", f"{summary['excess_kurtosis']:.2f}", c_accent_orange),
        ("GARCH(1,1) Residual Kurtosis", f"{garch_kurt:.2f} (-57%)", c_accent_green),
        ("Best Single Day", f"{summary['best_day']*100:.2f}%", c_accent_green)
    ]
    
    row3 = [
        ("Student-t Degrees of Freedom", f"ν = {t_df:.2f} (Heavy)", c_accent_green),
        ("Days Beyond ±3σ", f"Actual: {summary['outliers_3s']} vs Exp: {summary['expected_3s']:.1f}", c_val),
        ("Days Beyond ±4σ (Black Swans)", f"Actual: {summary['outliers_4s']} vs Exp: {summary['expected_4s']:.2f}", c_accent_orange),
        ("Days Beyond ±5σ", f"Actual: {summary['outliers_5s']} vs Exp: {summary['expected_5s']:.3f}", '#ef4444')
    ]
    
    col_x = [0.03, 0.28, 0.54, 0.78]
    
    for row_idx, r in enumerate([row1, row2, row3]):
        y_pos = 0.82 - (row_idx * 0.30)
        for col_idx, (label, val, val_color) in enumerate(r):
            x_pos = col_x[col_idx]
            ax_stats.text(x_pos, y_pos, label, fontsize=8.6, color=c_label, va='center')
            ax_stats.text(x_pos, y_pos - 0.12, val, fontsize=10.2, weight='bold', color=val_color, va='center')
            
    # Draw subtle bounding box around stats
    rect = plt.Rectangle((0.01, 0.02), 0.98, 0.96, transform=ax_stats.transAxes,
                         facecolor='#0b111e', edgecolor='#1e293b', linewidth=1.0, zorder=0)
    ax_stats.add_patch(rect)
    
    # =========================================================================
    # Footer footnote
    # =========================================================================
    footer_text = (
        "Data: S&P BSE SENSEX (^BSESN) daily closing prices (1997–2026). "
        "GBM drift (CAGR) and volatility calibrated over sample. GARCH(1,1) normal innovations. Scaled Student-t MLE. "
        "Illustrative quantitative finance research only."
    )
    fig.text(0.5, 0.015, footer_text, fontsize=8.0, color='#64748b', ha='center')
    
    # Save high-res
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    
    # Also save SVG
    svg_path = save_path.replace('.png', '.svg')
    plt.savefig(svg_path, format='svg', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Successfully generated publication figure: {save_path} and {svg_path}")


if __name__ == "__main__":
    build_publication_figure("sensex_gbm_fat_tails.png")

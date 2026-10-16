"""
Dynamic Time-Evolution Animation Generator
BSE SENSEX vs Geometric Brownian Motion (1997–2026)

Produces:
- 'sensex_gbm_evolution.gif': Animated loop showing expanding trajectory,
  distribution fattening, Q-Q tail tearing, and dynamic KPI metric updates.
"""

import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import scipy.stats as stats
from PIL import Image
import quant_models as qm

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#2a3447'
plt.rcParams['axes.linewidth'] = 0.8


def create_animation(
    gif_path: str = "sensex_gbm_evolution.gif",
    step_days: int = 100,
    fps: int = 8
):
    print("Initializing animation data...")
    df = qm.fetch_index_data(ticker="^BSESN")
    returns_full = df["Log_Return"]
    price_full = df["Normalized_Price"]
    dates_full = df.index
    total_len = len(df)
    
    # 1-year burn-in period before beginning animation
    burn_in = 252
    frame_indices = list(range(burn_in, total_len, step_days))
    if frame_indices[-1] != total_len - 1:
        frame_indices.append(total_len - 1)
        
    print(f"Total trading days: {total_len}. Generating {len(frame_indices)} frames...")
    
    # Pre-simulate full ensemble for trajectory backdrop
    full_stats = qm.calculate_summary_stats(returns_full, price_full)
    paths_full, p5_full, p50_full, p95_full = qm.simulate_gbm_ensemble(
        s0=price_full.iloc[0],
        drift=full_stats["ann_geo_return"],
        vol=full_stats["ann_vol"],
        n_steps=total_len - 1,
        n_paths=100,
        seed=101
    )
    
    # Styling colors
    c_bg = '#0e1422'
    c_actual = '#ff7844'
    c_gbm_band = '#0284c7'
    c_gbm_line = '#38bdf8'
    c_student = '#10b981'
    c_garch = '#a855f7'
    c_grid = '#1c2436'
    c_text = '#e2e8f0'
    c_muted = '#94a3b8'
    
    frames = []
    
    for idx_num, cur_idx in enumerate(frame_indices):
        cur_dates = dates_full[:cur_idx + 1]
        cur_price = price_full.iloc[:cur_idx + 1]
        cur_returns = returns_full.iloc[:cur_idx]
        cur_date_str = cur_dates[-1].strftime("%b %Y")
        
        # Calculate current window statistics
        summary = qm.calculate_summary_stats(cur_returns, cur_price)
        
        fig = plt.figure(figsize=(13.5, 9.8), facecolor='#080c14')
        gs = gridspec.GridSpec(
            nrows=3, ncols=2,
            height_ratios=[1.35, 1.05, 0.45],
            hspace=0.38, wspace=0.22,
            left=0.08, right=0.94, top=0.91, bottom=0.06
        )
        
        # Header
        fig.text(0.08, 0.965, "BSE SENSEX vs Geometric Brownian Motion", fontsize=18, weight='bold', color='#ffffff')
        fig.text(0.08, 0.940, "Expanding window calibration: watch fat tails and volatility regimes emerge in real time", fontsize=10, color=c_muted)
        
        # 1. TOP PANEL: Price Path
        ax_top = fig.add_subplot(gs[0, :], facecolor=c_bg)
        ax_top.set_yscale('log')
        ax_top.grid(True, which='both', color=c_grid, linestyle='-', linewidth=0.5, alpha=0.7)
        ax_top.set_xlim(dates_full[0], dates_full[-1])
        ax_top.set_ylim(40, 200000)
        
        # Plot GBM ensemble up to cur_idx
        for i in range(min(45, len(paths_full))):
            ax_top.plot(cur_dates, paths_full[i, :cur_idx + 1], color=c_gbm_line, alpha=0.06, linewidth=0.75)
            
        ax_top.fill_between(
            cur_dates, p5_full[:cur_idx + 1], p95_full[:cur_idx + 1],
            color=c_gbm_band, alpha=0.25, label='GBM 5–95% band'
        )
        ax_top.plot(cur_dates, p50_full[:cur_idx + 1], color=c_gbm_line, linestyle='--', linewidth=1.5, label='GBM median')
        ax_top.plot(cur_dates, cur_price, color=c_actual, linewidth=2.0, label='Actual SENSEX (Base 100)')
        ax_top.scatter([cur_dates[-1]], [cur_price.iloc[-1]], color=c_actual, s=40, zorder=5)
        
        ax_top.set_title("Price Path Trajectory (Log Scale)", fontsize=12, weight='semibold', color=c_text, loc='left', pad=6)
        ax_top.set_ylabel("Index Level (Base 100)", fontsize=9.5, color=c_muted)
        ax_top.tick_params(colors=c_muted, labelsize=9)
        ax_top.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{int(y):,}"))
        
        # Date badge
        ax_top.text(
            0.98, 0.08, cur_date_str,
            transform=ax_top.transAxes,
            fontsize=15, weight='bold', color='#ffffff',
            ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#111827', edgecolor='#374151', alpha=0.9)
        )
        leg_top = ax_top.legend(loc='upper left', frameon=True, facecolor='#111827', edgecolor='#2e394e', fontsize=8.5)
        for t in leg_top.get_texts():
            t.set_color(c_text)
            
        # 2. BOTTOM LEFT PANEL: Return distribution
        ax_dist = fig.add_subplot(gs[1, 0], facecolor=c_bg)
        ax_dist.grid(True, color=c_grid, linestyle='-', linewidth=0.5, alpha=0.7)
        ret_pct = cur_returns * 100.0
        x_grid = np.linspace(-6.0, 6.0, 500)
        
        ax_dist.hist(ret_pct, bins=60, range=(-6.0, 6.0), density=True, color=c_actual, alpha=0.82, edgecolor='#080c14', linewidth=0.3)
        pdf_norm = stats.norm.pdf(x_grid, loc=cur_returns.mean() * 100.0, scale=cur_returns.std() * 100.0)
        ax_dist.plot(x_grid, pdf_norm, color=c_gbm_line, linewidth=1.8, label='Normal (GBM)')
        
        t_df = summary["student_t_df"]
        t_loc = summary["student_t_loc"] * 100.0
        t_scale = summary["student_t_scale"] * 100.0
        pdf_t = stats.t.pdf(x_grid, df=t_df, loc=t_loc, scale=t_scale)
        ax_dist.plot(x_grid, pdf_t, color=c_student, linewidth=1.8, label=f'Student-t (df={t_df:.1f})')
        
        ax_dist.set_xlim(-6.0, 6.0)
        ax_dist.set_ylim(0, 0.48)
        ax_dist.set_title("Daily Return Distribution", fontsize=11, weight='semibold', color=c_text, loc='left', pad=6)
        ax_dist.set_xlabel("Daily Log Return (%)", fontsize=9, color=c_muted)
        ax_dist.set_ylabel("Density", fontsize=9, color=c_muted)
        ax_dist.tick_params(colors=c_muted, labelsize=8.5)
        leg_dist = ax_dist.legend(loc='upper left', frameon=True, facecolor='#111827', edgecolor='#2e394e', fontsize=8.0)
        for t in leg_dist.get_texts():
            t.set_color(c_text)
            
        # 3. BOTTOM RIGHT PANEL: Q-Q Plot
        ax_qq = fig.add_subplot(gs[1, 1], facecolor=c_bg)
        ax_qq.grid(True, color=c_grid, linestyle='-', linewidth=0.5, alpha=0.7)
        z_cur = (cur_returns - cur_returns.mean()) / cur_returns.std()
        osm_act, osr_act = stats.probplot(z_cur, dist="norm", fit=False)
        
        q_ref = np.linspace(-4.0, 4.0, 100)
        ax_qq.plot(q_ref, q_ref, color='#475569', linestyle='--', linewidth=1.2, label='Normal reference')
        ax_qq.scatter(osm_act[::2], osr_act[::2], color=c_actual, s=10, alpha=0.85, label='Actual SENSEX')
        
        ax_qq.set_xlim(-4.0, 4.0)
        ax_qq.set_ylim(-15.0, 9.0)
        ax_qq.set_title("Q-Q Plot vs Normal", fontsize=11, weight='semibold', color=c_text, loc='left', pad=6)
        ax_qq.set_xlabel("Theoretical Quantiles (σ)", fontsize=9, color=c_muted)
        ax_qq.set_ylabel("Sample Quantiles (σ)", fontsize=9, color=c_muted)
        ax_qq.tick_params(colors=c_muted, labelsize=8.5)
        leg_qq = ax_qq.legend(loc='upper left', frameon=True, facecolor='#111827', edgecolor='#2e394e', fontsize=8.0)
        for t in leg_qq.get_texts():
            t.set_color(c_text)
            
        # 4. KPI Table
        ax_stats = fig.add_subplot(gs[2, :], facecolor='#0b111e')
        ax_stats.axis('off')
        
        row1 = [
            ("Trading Days", f"{summary['n_days']:,}", '#f8fafc'),
            ("Ann. Return (CAGR)", f"{summary['ann_geo_return']*100:.2f}%", '#34d399'),
            ("Ann. Volatility (σ)", f"{summary['ann_vol']*100:.2f}%", '#38bdf8'),
            ("Worst Single Day", f"{summary['worst_day']*100:.2f}%", '#f87171')
        ]
        row2 = [
            ("Skewness", f"{summary['skewness']:.2f}", '#f8fafc'),
            ("Excess Kurtosis", f"{summary['excess_kurtosis']:.2f}", '#fb923c'),
            ("Student-t df", f"ν = {t_df:.2f}", '#34d399'),
            ("Best Single Day", f"{summary['best_day']*100:.2f}%", '#34d399')
        ]
        row3 = [
            ("Days Beyond ±3σ", f"Act: {summary['outliers_3s']} | Exp: {summary['expected_3s']:.1f}", '#f8fafc'),
            ("Days Beyond ±4σ", f"Act: {summary['outliers_4s']} | Exp: {summary['expected_4s']:.2f}", '#fb923c'),
            ("Days Beyond ±5σ", f"Act: {summary['outliers_5s']} | Exp: {summary['expected_5s']:.3f}", '#ef4444'),
            ("Current Date", cur_date_str, '#ffffff')
        ]
        
        col_x = [0.03, 0.28, 0.54, 0.78]
        for r_i, r_data in enumerate([row1, row2, row3]):
            y_p = 0.82 - (r_i * 0.30)
            for c_i, (lbl, val, col) in enumerate(r_data):
                ax_stats.text(col_x[c_i], y_p, lbl, fontsize=8.2, color=c_muted, va='center')
                ax_stats.text(col_x[c_i], y_p - 0.12, val, fontsize=9.8, weight='bold', color=col, va='center')
                
        rect = plt.Rectangle((0.01, 0.02), 0.98, 0.96, transform=ax_stats.transAxes,
                             facecolor='#0b111e', edgecolor='#1e293b', linewidth=1.0, zorder=0)
        ax_stats.add_patch(rect)
        
        fig.text(0.5, 0.015, "Expanding window SENSEX data vs dynamic GBM & Student-t calibration. Antigravity Quantitative Research.", fontsize=7.5, color='#64748b', ha='center')
        
        # Render to in-memory PIL image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, facecolor=fig.get_facecolor(), edgecolor='none')
        buf.seek(0)
        frame_img = Image.open(buf).convert("RGB")
        frames.append(frame_img)
        plt.close(fig)
        
        if (idx_num + 1) % 10 == 0 or idx_num == len(frame_indices) - 1:
            print(f"Rendered frame {idx_num + 1}/{len(frame_indices)} ({cur_date_str})")
            
    # Save as animated GIF with 3-second hold on the final frame
    print("Compiling frames into animated GIF...")
    durations = [int(1000 / fps)] * len(frames)
    durations[-1] = 2500  # Pause at final frame
    
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True
    )
    print(f"Animation saved successfully to: {gif_path}")


if __name__ == "__main__":
    create_animation("sensex_gbm_evolution.gif", step_days=100, fps=8)

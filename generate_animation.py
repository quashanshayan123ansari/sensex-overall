"""
Dynamic Monthly Growth Animation Generator (Clean White Background)
BSE SENSEX vs Geometric Brownian Motion (July 1997 – October 2026)

Palette: Pure White (#ffffff), Emerald Jade (#059669), and Royal Violet (#7c3aed).
Grows month-by-month through 29+ years of empirical market history.
"""

import os
import io
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from PIL import Image

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 0.8


def generate_monthly_growth_gif(
    json_path: str = "monthly_data.json",
    gif_path: str = "sensex_gbm_evolution.gif",
    step_months: int = 4,
    fps: int = 10
):
    print("Loading monthly dataset...")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    timeline = data['timeline']
    total_months = len(timeline)
    gbm_p5 = data['gbm']['p5']
    gbm_p50 = data['gbm']['p50']
    gbm_p95 = data['gbm']['p95']
    gbm_paths = data['gbm']['paths']
    
    frame_indices = list(range(4, total_months, step_months))
    if frame_indices[-1] != total_months - 1:
        frame_indices.append(total_months - 1)
        
    print(f"Generating {len(frame_indices)} monthly growth animation frames on white background...")
    
    # White Mode Palette
    c_bg = '#ffffff'
    c_card = '#f8fafc'
    c_sensex = '#059669'       # Vibrant Emerald Green
    c_gbm_band = '#7c3aed'     # Royal Violet
    c_gbm_line = '#6d28d9'     # Deep Violet
    c_crimson = '#e11d48'      # Crisp Crimson
    c_grid = '#e2e8f0'
    c_text = '#0f172a'
    c_muted = '#64748b'
    
    frames = []
    
    for f_idx, m_idx in enumerate(frame_indices):
        item = timeline[m_idx]
        cur_month_str = item['month']
        
        fig = plt.figure(figsize=(14, 9.2), facecolor=c_bg)
        gs = gridspec.GridSpec(
            nrows=3, ncols=2,
            height_ratios=[1.35, 0.95, 0.42],
            hspace=0.38, wspace=0.22,
            left=0.08, right=0.94, top=0.865, bottom=0.055
        )
        
        # Header
        fig.text(0.08, 0.955, "BSE SENSEX Dynamic Trajectory Growth", fontsize=18, weight='bold', color=c_text)
        fig.text(0.08, 0.922, f"Empirical Monthly Evolution: July 1997 \u2192 {cur_month_str} vs Calibrated GBM Diffusion", fontsize=10.5, color=c_muted)
        
        # 1. Main Hero Panel: Monthly Price Growth (Log Scale)
        ax_main = fig.add_subplot(gs[0, :], facecolor=c_card)
        ax_main.set_yscale('log')
        ax_main.grid(True, which='both', color=c_grid, linestyle='-', linewidth=0.7, alpha=0.9)
        ax_main.set_xlim(0, total_months - 1)
        ax_main.set_ylim(40, 200000)
        
        tick_pos = list(range(0, total_months, 48))
        tick_lbls = [timeline[p]['month'].split()[-1] for p in tick_pos]
        ax_main.set_xticks(tick_pos)
        ax_main.set_xticklabels(tick_lbls, color=c_muted, fontsize=9.5)
        
        # Background GBM diffusion paths
        for p_i in range(min(20, len(gbm_paths))):
            ax_main.plot(range(m_idx + 1), gbm_paths[p_i][:m_idx + 1], color='#8b5cf6', alpha=0.10, linewidth=0.8)
            
        # 5-95% Diffusion Cone
        ax_main.fill_between(
            range(m_idx + 1), gbm_p5[:m_idx + 1], gbm_p95[:m_idx + 1],
            color=c_gbm_band, alpha=0.15, label='GBM 5–95% Diffusion Cone'
        )
        
        # GBM Median
        ax_main.plot(
            range(m_idx + 1), gbm_p50[:m_idx + 1],
            color=c_gbm_line, linestyle='--', linewidth=1.6, label='GBM Median Drift'
        )
        
        # Growing SENSEX Line
        sensex_sub = [timeline[k]['price'] for k in range(m_idx + 1)]
        ax_main.plot(
            range(m_idx + 1), sensex_sub,
            color=c_sensex, linewidth=2.5, label='Realized BSE SENSEX (Base 100)'
        )
        
        # Moving Head Dot
        ax_main.scatter([m_idx], [sensex_sub[-1]], color=c_sensex, s=40, zorder=6, edgecolors='#ffffff', linewidths=2)
        
        ax_main.set_title("Compounded Price Trajectory (Log Scale)", fontsize=12, weight='semibold', color=c_text, loc='left', pad=6)
        ax_main.set_ylabel("Index Level (Base 100)", fontsize=9.5, color=c_muted)
        ax_main.tick_params(colors=c_muted, labelsize=9)
        ax_main.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{int(y):,}"))
        
        # Month HUD Badge
        ax_main.text(
            0.98, 0.10, cur_month_str.upper(),
            transform=ax_main.transAxes,
            fontsize=16, weight='bold', color=c_sensex,
            ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffffff', edgecolor=c_sensex, alpha=0.95)
        )
        
        leg = ax_main.legend(loc='upper left', frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=8.8)
        for t in leg.get_texts():
            t.set_color(c_text)
            
        # 2. Bottom Left: Return Density
        ax_dist = fig.add_subplot(gs[1, 0], facecolor=c_card)
        ax_dist.grid(True, color=c_grid, linestyle='-', linewidth=0.7, alpha=0.9)
        bins = data['distribution']['bins']
        dens = data['distribution']['density']
        bar_w = (bins[-1] - bins[0]) / len(bins)
        
        ax_dist.bar(bins, dens, width=bar_w * 0.9, color=c_sensex, alpha=0.65, label='Actual Returns')
        
        x_norm = np.linspace(-6, 6, 200)
        norm_std = data['distribution']['std']
        p_norm = (1.0 / (norm_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x_norm / norm_std)**2)
        ax_dist.plot(x_norm, p_norm, color=c_gbm_line, linewidth=1.8, label='GBM Normal')
        
        ax_dist.set_xlim(-6.0, 6.0)
        ax_dist.set_ylim(0, 0.44)
        ax_dist.set_title("Return Density & Leptokurtosis", fontsize=11, weight='semibold', color=c_text, loc='left', pad=6)
        ax_dist.set_xlabel("Daily Return (%)", fontsize=9, color=c_muted)
        ax_dist.set_ylabel("Density", fontsize=9, color=c_muted)
        ax_dist.tick_params(colors=c_muted, labelsize=8.5)
        leg_d = ax_dist.legend(loc='upper left', frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=8.0)
        for t in leg_d.get_texts():
            t.set_color(c_text)
            
        # 3. Bottom Right: Expanding Excess Kurtosis
        ax_kurt = fig.add_subplot(gs[1, 1], facecolor=c_card)
        ax_kurt.grid(True, color=c_grid, linestyle='-', linewidth=0.7, alpha=0.9)
        
        kurt_history = [timeline[k]['kurt'] for k in range(m_idx + 1)]
        ax_kurt.axhline(0, color='#64748b', linestyle='--', linewidth=1.2, label='Normal Baseline (0.0)')
        if m_idx > 12:
            ax_kurt.plot(range(12, m_idx + 1), kurt_history[12:], color=c_crimson, linewidth=2.0, label='Realized Excess Kurtosis')
            ax_kurt.scatter([m_idx], [kurt_history[-1]], color=c_crimson, s=25, edgecolors='#ffffff', zorder=5)
            
        ax_kurt.set_xlim(0, total_months - 1)
        ax_kurt.set_ylim(-2, 16)
        ax_kurt.set_xticks(tick_pos)
        ax_kurt.set_xticklabels(tick_lbls, color=c_muted, fontsize=9.5)
        ax_kurt.set_title("Fat-Tail Accumulation (Excess Kurtosis)", fontsize=11, weight='semibold', color=c_text, loc='left', pad=6)
        ax_kurt.set_ylabel("Kurtosis (Fisher)", fontsize=9, color=c_muted)
        ax_kurt.tick_params(colors=c_muted, labelsize=8.5)
        leg_k = ax_kurt.legend(loc='upper left', frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=8.0)
        for t in leg_k.get_texts():
            t.set_color(c_text)
            
        # 4. Summary Telemetry Row
        ax_stat = fig.add_subplot(gs[2, :], facecolor=c_bg)
        ax_stat.axis('off')
        
        stats_row = [
            ("Current Month", cur_month_str.upper(), c_sensex),
            ("SENSEX Level", f"{int(item['raw_price']):,} pts", c_text),
            ("Realized CAGR", f"{item['cagr']:.2f}%", '#2563eb'),
            ("Realized Volatility (\u03c3)", f"{item['vol']:.2f}%", c_gbm_line),
            ("4-Sigma Black Swans", f"{item['outliers']} days", c_crimson)
        ]
        
        col_xs = [0.03, 0.22, 0.43, 0.64, 0.84]
        for c_i, (lbl, val, col) in enumerate(stats_row):
            ax_stat.text(col_xs[c_i], 0.68, lbl, fontsize=8.5, color=c_muted, va='center', weight='bold')
            ax_stat.text(col_xs[c_i], 0.25, val, fontsize=11.5, weight='bold', color=col, va='center')
            
        rect = plt.Rectangle((0.005, 0.05), 0.99, 0.90, transform=ax_stat.transAxes,
                             facecolor='#f8fafc', edgecolor='#e2e8f0', linewidth=1.0, zorder=0)
        ax_stat.add_patch(rect)
        
        # Save frame to PIL
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, facecolor=fig.get_facecolor(), edgecolor='none')
        buf.seek(0)
        frame_img = Image.open(buf).convert("RGB")
        frames.append(frame_img)
        plt.close(fig)
        
        if (f_idx + 1) % 20 == 0 or f_idx == len(frame_indices) - 1:
            print(f"Rendered frame {f_idx + 1}/{len(frame_indices)} ({cur_month_str})")
            
    print("Compiling monthly growth animation GIF on white background...")
    durations = [int(1000 / fps)] * len(frames)
    durations[-1] = 3000
    
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True
    )
    print(f"White background monthly animation saved to: {gif_path}")


if __name__ == "__main__":
    generate_monthly_growth_gif()

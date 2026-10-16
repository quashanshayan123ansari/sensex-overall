# BSE SENSEX vs Geometric Brownian Motion (GBM)
### Quantitative Modeling of Fat Tails, Volatility Clustering, and Stochastic Drift (1997–2026)

Inspired by the classical debate on mathematical models in finance (such as [Bruce Tan's analysis](https://lnkd.in/p/g9F9QXnp) of the S&P 500), this repository replicates and extends the quantitative benchmarking of empirical market returns against **Geometric Brownian Motion (GBM)** using nearly 30 years of daily data from the **BSE SENSEX** (India's premier benchmark index).

---

## 📌 Core Quantitative Findings

| Metric | Empirical SENSEX | Gaussian GBM Expectation | Student-t MLE / GARCH(1,1) |
| :--- | :---: | :---: | :---: |
| **Trading Days** | **7,216** (1997–2026) | 7,216 | 7,216 |
| **Annualized Return (CAGR)** | **10.21%** | 10.21% (calibrated drift) | — |
| **Annualized Volatility ($\sigma$)** | **22.38%** | 22.38% (calibrated vol) | Conditional $\sigma_t$ |
| **Excess Kurtosis** | **8.76** (Leptokurtic) | **0.00** (Mesokurtic) | Standardized Residuals: **2.07** (-57%) |
| **Tail Index / Shape Parameter** | Heavy Power-Law | $\nu \to \infty$ | **$\nu = 3.11$** (Extreme Fat Tails) |
| **Days Beyond $\pm 3\sigma$** | **113 days** | **19.5 days** | Captures empirical tail frequency |
| **Days Beyond $\pm 4\sigma$ (Black Swans)**| **45 days** | **0.46 days** (~1 day in 60 yrs)| ~100x more frequent than GBM |
| **Days Beyond $\pm 5\sigma$** | **18 days** | **0.004 days** (~1 in 1,000 yrs)| Structural market crashes |
| **Worst Single Day** | **-14.10%** (May 17, 2004) | ~$3.5\sigma$ limit expected | Actual realized: $-9.8\sigma$ |

### Key Takeaway:
> **"You don't learn GBM because markets follow it. You learn it because you'll spend your career measuring how they don't."**
>
> 1. **Long-Run Horizon**: The empirical SENSEX trajectory stays comfortably within the **GBM 5%–95% confidence ribbon** over nearly 30 years. GBM roughly holds for long-run geometric compounded growth.
> 2. **Short-Run Failure**: At daily horizons, GBM drastically underestimates extreme risk. 45 days breach $4\sigma$ when Gaussian statistics predict less than half a day.
> 3. **The GARCH & Student-t Link**: When returns are scaled by daily conditional volatility $\sigma_t$ via **GARCH(1,1)**, excess kurtosis collapses from **8.76 down to 2.07** (-57%). This proves that "fat tails" and "volatility clustering" are largely two perspectives of the same underlying dynamic. Fitting a **Student-t with $\nu \approx 3.1$** models the empirical density with remarkable precision.

---

## 🎨 Deliverables & Visualizations

1. **`sensex_gbm_fat_tails.png`**: High-resolution 300-DPI publication chart with modern dark-mode terminal styling, 3 primary panels (Log Price Trajectory, Return Distribution with Student-t, and Q-Q Plot with GARCH correction), and tabular KPI tiles.
2. **`sensex_gbm_evolution.gif`**: Animated dynamic simulation looping through historical time (1998 to 2026), demonstrating the expanding trajectory, distribution fattening, and tail divergence in real time.
3. **`app.py`**: Streamlit interactive quantitative web dashboard with customizable parameters, Monte Carlo paths, confidence bands, and Plotly graphics.
4. **`index.html`**: Zero-dependency standalone HTML5/Canvas dashboard with interactive scrubbing slider, play/pause controls, model toggles, and live metrics.
5. **`quant_models.py`**: Clean, modular Python engine for data caching, GBM ensemble simulation, Student-t MLE fitting, GARCH(1,1), and tail statistics.

---

## 🚀 How to Run

### 1. One-Click Launcher (Windows)
Double-click `run_dashboard.bat` and select:
- `1` for Streamlit Interactive Dashboard
- `2` for Standalone HTML Web Visualizer
- `3` to re-generate publication charts and animated GIF

### 2. Standalone HTML Visualizer
Simply open `index.html` directly in any web browser:
```bash
start index.html
```

### 3. Streamlit Dashboard
```bash
streamlit run app.py
```

### 4. Re-generate Charts & Animation
```bash
python generate_publication_charts.py
python generate_animation.py
```

---

## 📐 Mathematical Formulation

### 1. Geometric Brownian Motion (GBM)
$$dS_t = \mu S_t dt + \sigma S_t dW_t$$

Under Ito's Lemma for discrete trading steps $\Delta t = 1/252$:
$$r_t = \ln\left(\frac{S_t}{S_{t-1}}\right) \sim \mathcal{N}\left(\left(\mu - \frac{1}{2}\sigma^2\right)\Delta t, \; \sigma^2 \Delta t\right)$$

### 2. Student-t Heavy Tail Formulation
$$f(r; \nu, \mu, s) = \frac{\Gamma\left(\frac{\nu + 1}{2}\right)}{\sqrt{\pi \nu} \, s \, \Gamma\left(\frac{\nu}{2}\right)} \left(1 + \frac{1}{\nu}\left(\frac{r - \mu}{s}\right)^2\right)^{-\frac{\nu+1}{2}}$$
With empirical SENSEX estimate $\nu \approx 3.11$, the theoretical 4th moment is near divergence, reflecting severe leptokurtosis.

### 3. GARCH(1,1) Volatility Clustering
$$r_t = \mu + \epsilon_t, \quad \epsilon_t = \sigma_t z_t, \quad z_t \sim \mathcal{N}(0, 1)$$
$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$
Standardized innovations $z_t = \frac{\epsilon_t}{\sigma_t}$ demonstrate that adjusting for time-varying conditional variance removes the majority of excess kurtosis.

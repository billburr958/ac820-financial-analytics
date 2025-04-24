# 📈 AC820 Financial Analytics

A Streamlit-based interactive dashboard for end-to-end portfolio analysis, optimization, performance measurement, fraud detection, and 10-K exploration.

---

## 🚀 Features

- **Portfolio Overview**  
  Historical price charts, technical indicators (SMA, volatility, cumulative return), and company logos.
- **Performance & Metrics**  
  • Simulate portfolio growth for any allocation  
  • Compute Annual Return, Volatility, Sharpe, Sortino, Calmar, Max Drawdown, VaR/CVaR  
  • Calculate Beta & R² vs. a benchmark (default: SPY)
- **Optimizer**  
  • Monte-Carlo Markowitz frontier with Max-Sharpe / Min-Vol portfolios  
  • Bayesian weight optimization
- **Financials**  
  Pull and display key financial statement items
- **Fraud Detection**  
  • Custom-trained CatBoost model with SHAP explainability  
  • Beneish M-Score & Piotroski F-Score rule-based screens
- **10-K Explorer**  
  Browse and search SEC 10-K filings by company

---

## 📦 Installation

1. **Clone this repository**  
   ```bash
   git clone https://github.com/billburr958/ac820-financial-analytics.git
   cd ac820-financial-analytics
   ```

2. **Create & activate a virtual environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate        # macOS/Linux
   venv\Scripts\activate         # Windows
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Usage

```bash
streamlit run app/app.py
```

- Use the **sidebar** to select tickers and date range.  
- Navigate the **tabs** to explore each module:
  1. Portfolio Overview  
  2. Financials  
  3. Optimizer  
  4. Performance & Metrics  
  5. Fraud Detection  
  6. 10-K Explorer

---

## 📂 Project Structure

```
ac820-financial-analytics/
├── app/
│   └── app.py               # Entrypoint for Streamlit
├── assets/
│   └── style.css            # Custom CSS
├── tabs/
│   ├── portfolio.py
│   ├── performance.py
│   ├── optimizer.py
│   ├── financials.py
│   ├── fraud_detection.py
│   └── tenk_explorer.py
├── utils/
│   ├── data.py              # Data loaders & helpers
│   └── logger.py            # Logging setup
├── config.py                # Global CSS/theme constants
├── requirements.txt         # Python dependencies
└── .streamlit/
    └── config.toml          # Streamlit theming
```

---

## 🛠 Development & Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repo**  
2. **Create a feature branch**  
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**  
   - Follow PEP8 for Python  
   - Update `requirements.txt` if you add dependencies  
4. **Test locally**  
   ```bash
   streamlit run app/app.py
   ```
5. **Stage & commit**  
   ```bash
   git add .
   git commit -m "Feature: describe your change"
   ```
6. **Push & open a Pull Request**  
   ```bash
   git push -u origin feature/your-feature-name
   ```
   Then create a PR against `main`.

---

```markdown
📊 Real-Time Portfolio Optimization Engine

A fully automated and advanced portfolio management system designed to simulate and optimize investment strategies using a wide suite of quantitative finance and AI/ML techniques. This system dynamically rebalances a portfolio in real-time based on market data, risk measures, liquidity constraints, and regime awareness.

---

🚀 Key Highlights

- 💰 $100K simulated capital with weekly rebalancing
- 📈 22.5% portfolio return in 12 weeks (with lower drawdown than major benchmarks)
- 🧠 Combines traditional finance with cutting-edge AI & optimization tools

---

🧠 Algorithms & Features

🧮 Core Optimization Techniques
- NSGA-II (Genetic Multi-Objective Optimization): Maximizes return while minimizing risk
- Black-Litterman Model: Incorporates both market equilibrium and custom views
- Bayesian Optimization: Smart hyperparameter tuning with turnover penalties
- Monte Carlo Simulations: Stress-tests the portfolio across thousands of futures
- Walk-Forward Optimization: Prevents overfitting via realistic backtesting

📊 Market Intelligence
- Hidden Markov Models (HMM): Regime detection (bull/bear/neutral markets)
- Liquidity Constraints: Filters for tradeable, high-volume assets
- Transaction Cost Modeling: Real-world cost-aware allocation logic
- Risk Metrics: Sharpe, Sortino, Beta, Value-at-Risk (VaR), Expected Shortfall (ES)

📉 Stress Testing
- 2008 Financial Crisis
- 2020 COVID Crash

---

📁 Folder Structure

```
.
├── data/                    Output CSVs and historical data
├── images/                  Plots and charts (efficient frontier, drawdown, etc.)
├── core/                    Core optimization logic (modular functions)
├── simulations/             Crisis simulations, Monte Carlo, regime analysis
├── notebooks/               Optional Jupyter demos (coming soon)
├── main.py                  Master script for full pipeline
├── requirements.txt         All dependencies
└── README.md                This file
```

---

📥 Installation

```bash
git clone https://github.com/tofaelttk/AI-portfolio-optimization.git
cd AI-portfolio-optimization
pip install -r requirements.txt
```

---

⚙️ Usage

Run the full pipeline:

```bash
python main.py \
    --tickers AAPL MSFT GOOGL AMZN TSLA META NVDA BRK-B JPM V \
    --investment 100000 \
    --tx_cost 0.001 \
    --init_date 2025-01-01
```

Outputs:
- Initial vs Final Portfolio (`CSV`)
- Weekly performance and equity curve
- Efficient frontier and optimal portfolio visualization
- Risk stats (Sharpe, Sortino, Beta, Drawdown, VaR)
- Simulated market stress performance
- Benchmark comparison with S&P500, NASDAQ, Dow

---

📈 Sample Results

![Equity Curve](images/portfolio_equity_curve.png)
![Efficient Frontier](images/efficient_frontier.png)
![Drawdown](images/drawdown.png)

---

🛠️ Built With

- Python 3.10+
- `yfinance`, `PyPortfolioOpt`, `pymoo`, `hmmlearn`, `scikit-learn`, `matplotlib`, `scipy`, `skopt`
- Modular, extensible architecture for plug-and-play strategies

---

🤝 Contributions

Open to enhancements, factor models, or integration with APIs (e.g., Alpaca, IBKR).

Feel free to:
- Fork the repo
- Submit pull requests
- Suggest features in [issues](https://github.com/tofaelttk/AI-portfolio-optimization/issues)

---

📫 Contact

Want to collaborate or integrate this with your fintech product?

Reach out on [LinkedIn](https://www.linkedin.com/in/toifaelttk) or open an issue.

---

🧾 License

MIT License. Free to use, modify, and distribute — with attribution.



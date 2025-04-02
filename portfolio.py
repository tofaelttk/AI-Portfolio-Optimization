import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from urllib3.exceptions import NotOpenSSLWarning
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time
import datetime as dt
from datetime import datetime, timedelta
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize as pymoo_minimize
from pypfopt import BlackLittermanModel
from sklearn.model_selection import TimeSeriesSplit
from skopt import gp_minimize
from hmmlearn import hmm
from scipy.stats import norm
from dateutil.relativedelta import relativedelta
import logging
import argparse

# -----------------------------
# Set up logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# -----------------------------
# Parse user parameters (or use defaults)
# -----------------------------
parser = argparse.ArgumentParser(description="Advanced Portfolio Optimization and Analysis")
parser.add_argument("--tickers", type=str, nargs="+", default=['AAPL','MSFT','GOOGL','AMZN','TSLA','META','NVDA','BRK-B','JPM','V'],
                    help="List of ticker symbols")
parser.add_argument("--investment", type=float, default=100000, help="Initial investment amount")
parser.add_argument("--tx_cost", type=float, default=0.001, help="Transaction cost rate (e.g., 0.001 for 0.1%)")
parser.add_argument("--init_date", type=str, default="2025-01-01", help="Initial investment date (YYYY-MM-DD)")
args = parser.parse_args()

tickers = args.tickers
initial_investment = args.investment
transaction_cost_rate = args.tx_cost
initial_investment_date = args.init_date

# -----------------------------
# Dynamic Date Handling using relativedelta
# -----------------------------
end_date = dt.datetime.today().strftime('%Y-%m-%d')
start_date = (dt.datetime.today() - relativedelta(years=10)).strftime('%Y-%m-%d')

# -----------------------------
# Risk-Free Rate Calculation (10-year Treasury yield)
# -----------------------------
def get_risk_free_rate():
    try:
        treasury_data = yf.download("^TNX", period="1d", auto_adjust=False)
        if 'Adj Close' in treasury_data.columns:
            return treasury_data['Adj Close'].iloc[-1] / 100
        elif 'Close' in treasury_data.columns:
            return treasury_data['Close'].iloc[-1] / 100
        else:
            raise KeyError("No valid column found for risk-free rate")
    except Exception as e:
        logging.error(f"Error fetching risk-free rate: {e}")
        return 0.04

risk_free_rate = get_risk_free_rate()
rf_scalar = float(risk_free_rate)  # Ensure it's a plain float

# -----------------------------
# Data Retrieval Functions
# -----------------------------
def get_historical_data(tickers, end_date, years=10):
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_date_dt = end_date_dt - relativedelta(years=years)
    start_date_local = start_date_dt.strftime("%Y-%m-%d")
    data = yf.download(tickers, start=start_date_local, end=end_date, auto_adjust=False)
    if 'Adj Close' in data.columns:
        return data['Adj Close']
    else:
        return data['Close']

def get_historical_data_with_retry(tickers, end_date, years=10, max_retries=3, sleep_seconds=10):
    for attempt in range(max_retries):
        try:
            data = get_historical_data(tickers, end_date, years)
            if data.empty:
                raise ValueError("Downloaded data is empty.")
            return data
        except Exception as e:
            logging.warning(f"Attempt {attempt+1}/{max_retries} failed: {e}")
            time.sleep(sleep_seconds * (attempt + 1))
    raise Exception("Failed to download data after several attempts.")

def get_prices_on_date(tickers, date):
    df = yf.download(tickers, start=date, end=(pd.to_datetime(date) + timedelta(days=1)).strftime('%Y-%m-%d'))
    if 'Adj Close' in df.columns:
        return df['Adj Close'].iloc[-1] if not df['Adj Close'].empty else pd.Series(dtype=float)
    return pd.Series(dtype=float)

# -----------------------------
# Portfolio Metrics Functions
# -----------------------------
def calculate_portfolio_stats(weights, returns, cov_matrix, rf=risk_free_rate):
    if isinstance(returns, np.ndarray):
        ret_mean = np.mean(returns, axis=0)
    elif isinstance(returns, pd.DataFrame):
        ret_mean = returns.mean()
    else:
        ret_mean = np.mean(returns)
    annual_return = np.sum(ret_mean * weights) * 252
    annual_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe_ratio = (annual_return - rf) / annual_vol if annual_vol != 0 else np.nan
    if isinstance(returns, pd.DataFrame):
        betas = calculate_betas(returns)
    else:
        betas = np.full(len(weights), np.nan)
    portfolio_beta = np.nansum(betas * weights)
    return {'return': annual_return, 'std_dev': annual_vol, 'sharpe': sharpe_ratio, 'beta': portfolio_beta}

def calculate_betas(returns):
    if not isinstance(returns, pd.DataFrame):
        return np.full((1,), np.nan)
    try:
        market_df = yf.download('SPY', start=returns.index[0], end=returns.index[-1])
        if 'Adj Close' in market_df.columns:
            market_data = market_df['Adj Close']
        elif 'Close' in market_df.columns:
            market_data = market_df['Close']
        else:
            raise KeyError("No valid price column found for SPY.")
        market_returns = market_data.pct_change().dropna()
        aligned_data = pd.concat([market_returns, returns], axis=1).dropna()
        market_returns = aligned_data.iloc[:, 0]
        stock_returns = aligned_data.iloc[:, 1:]
        market_var = market_returns.var()
        betas = [market_returns.cov(stock_returns[ticker]) / market_var for ticker in stock_returns.columns]
        return np.array(betas)
    except Exception as e:
        logging.error(f"Error calculating betas: {e}")
        return np.ones(returns.shape[1])

def get_avg_daily_volume(tickers, end_date, years=5):
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_date_dt = end_date_dt - relativedelta(years=years)
    start_date_local = start_date_dt.strftime("%Y-%m-%d")
    knowledge_cutoff = datetime.strptime("2024-10-31", "%Y-%m-%d")
    historical_end = min(end_date_dt, knowledge_cutoff)
    historical_end_str = historical_end.strftime("%Y-%m-%d")
    volume_data = yf.download(tickers, start=start_date_local, end=historical_end_str)['Volume']
    avg_volume = volume_data.mean() / 1e6
    return np.nan_to_num(avg_volume, nan=0.1)

def objective_function(weights, returns, cov_matrix, rf=risk_free_rate):
    stats = calculate_portfolio_stats(weights, returns, cov_matrix, rf)
    return -stats['sharpe']

def constraint_sum(weights):
    return np.sum(weights) - 1

def calculate_max_drawdown(series):
    cum_max = series.cummax()
    drawdown = (series - cum_max) / cum_max
    return drawdown.min()

def calculate_sortino_ratio(weights, returns, rf=risk_free_rate):
    portfolio_daily_returns = (returns * weights).sum(axis=1)
    rf_daily = float(rf) / 252
    downside_mask = portfolio_daily_returns < rf_daily
    downside_returns = rf_daily - portfolio_daily_returns[downside_mask]
    if len(downside_returns) == 0:
        return np.inf
    downside_std = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(252)
    annual_return = np.sum(returns.mean() * weights) * 252
    return (annual_return - float(rf)) / downside_std if downside_std != 0 else np.nan

# -----------------------------
# Dynamic Liquidity Constraint
# -----------------------------
def liquidity_constraint(weights, avg_volume, prices, current_value, max_turnover=0.2):
    dollar_weights = weights * current_value
    position_size = dollar_weights / prices
    liquidity_ratio = position_size / (avg_volume * 1e6)
    if liquidity_ratio.size == 0:
        return 1
    return max_turnover - liquidity_ratio.max()

# -----------------------------
# Advanced Enhancements & Optimization Functions
# -----------------------------
if pymoo_minimize is not None:
    class PortfolioProblem(ElementwiseProblem):
        def __init__(self, returns, cov_matrix, rf=risk_free_rate):
            self.returns = returns
            self.cov_matrix = cov_matrix
            self.rf = rf
            n_var = returns.shape[1]
            super().__init__(n_var=n_var, n_obj=2, n_constr=1, xl=0, xu=1)
    
        def _evaluate(self, x, out, *args, **kwargs):
            weights = np.array(x, dtype=float) / np.sum(x)
            port_return = np.sum(self.returns.mean() * weights) * 252
            port_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
            out["F"] = [-port_return, port_vol]
            out["G"] = np.sum(weights) - 1

    def pareto_frontier_optimization(returns, cov_matrix, rf=risk_free_rate):
        problem = PortfolioProblem(returns, cov_matrix, rf)
        algorithm = NSGA2(pop_size=100)
        res = pymoo_minimize(problem, algorithm, ('n_gen', 100), verbose=False)
        return res
else:
    def pareto_frontier_optimization(*args, **kwargs):
        logging.warning("NSGA2 optimization is disabled because pymoo is not available.")
        return None

def implementation_shortfall(prev_weights, new_weights, prices, volumes):
    turnover = np.abs(new_weights - prev_weights)
    market_impact = 0.01 * (turnover / volumes) ** 0.5
    return np.sum(turnover * prices * market_impact)

def walk_forward_optimization(returns, n_splits, initial_guess, bounds, constraints, rf):
    try:
        from sklearn.model_selection import TimeSeriesSplit
    except ImportError:
        logging.error("Walk-forward optimization disabled; install scikit-learn.")
        return []
    tscv = TimeSeriesSplit(n_splits=n_splits)
    opt_results = []
    for train_idx, test_idx in tscv.split(returns):
        train_returns = returns.iloc[train_idx]
        train_cov = train_returns.cov()
        res = minimize(objective_function, initial_guess, args=(train_returns, train_cov, rf),
                       method='SLSQP', bounds=bounds, constraints=constraints)
        if not res.success:
            logging.warning("Optimization failed; using equal weights.")
            res.x = np.ones(len(initial_guess)) / len(initial_guess)
        res.x = res.x / np.sum(res.x)
        opt_results.append(res)
    return opt_results

def get_ewm_covariance(returns, halflife=63):
    ewm_cov = returns.ewm(halflife=halflife).cov()
    last_date = returns.index[-1]
    return ewm_cov.loc[last_date].values

def get_black_litterman_returns(cov_matrix, returns):
    if BlackLittermanModel is None:
        logging.info("Black–Litterman integration is disabled.")
        return None
    n = cov_matrix.shape[0]
    P = np.eye(n)
    Q = pd.Series(np.zeros(n), index=returns.columns)
    bl = BlackLittermanModel(cov_matrix, pi="equal", P=P, Q=Q)
    return bl.bl_returns()

def crisis_simulation(weights, returns):
    crisis_periods = returns[returns.index.strftime("%Y-%m").isin(["2008-09", "2020-03"])]
    if crisis_periods.empty:
        logging.info("No crisis periods found in the dataset. Skipping stress test.")
        return None
    crisis_cov = crisis_periods.cov()
    return calculate_portfolio_stats(weights, crisis_periods, crisis_cov)

def detect_market_regimes(returns, n_regimes=3):
    if hmm is None:
        logging.info("Market regime detection is disabled because hmmlearn is not available.")
        return np.zeros(len(returns))
    model = hmm.GaussianHMM(n_components=n_regimes, covariance_type="diag", n_iter=1000)
    model.fit(returns)
    regimes = model.predict(returns)
    return regimes

def regime_aware_optimization(returns, current_regime, initial_guess, bounds, constraints, rf):
    regimes = detect_market_regimes(returns)
    regime_returns = returns[regimes == current_regime]
    regime_cov = regime_returns.cov()
    res = minimize(objective_function, initial_guess, args=(regime_returns, regime_cov, rf),
                   method='SLSQP', bounds=bounds, constraints=constraints)
    return res

def bayesian_optimization(returns, cov_matrix, initial_weights, max_turnover, tickers, rf):
    if gp_minimize is None:
        logging.info("Bayesian optimization is disabled because skopt is not available.")
        return initial_weights
    space = [(0, 1) for _ in range(len(tickers))]
    
    def objective(params):
        new_weights = np.array(params, dtype=float)
        turnover = np.abs(new_weights - initial_weights).sum()
        stats = calculate_portfolio_stats(new_weights, returns, cov_matrix, rf)
        sharpe_value = float(stats['sharpe'])  # force scalar conversion
        if np.isnan(sharpe_value):
            return 1e6
        return -sharpe_value + 100 * max(0, turnover - max_turnover)
    
    res = gp_minimize(objective, space, n_calls=100, random_state=0)
    return res.x

def calculate_var(weights, returns, confidence=0.95):
    portfolio_returns = returns.dot(weights)
    var = np.percentile(portfolio_returns, 100 * (1 - confidence))
    return var

def expected_shortfall(weights, returns, confidence=0.95):
    portfolio_returns = returns.dot(weights)
    var = calculate_var(weights, returns, confidence)
    es = portfolio_returns[portfolio_returns <= var].mean()
    return es

def monte_carlo_validation(optimal_weights, n_simulations, returns, cov_matrix, rf):
    simulated_returns = np.random.multivariate_normal(
        returns.mean(), 
        cov_matrix, 
        size=(n_simulations, returns.shape[0])
    )
    sharpe_ratios = []
    for sim in simulated_returns:
        sim_cov = np.cov(sim.T)
        stats = calculate_portfolio_stats(optimal_weights, sim.mean(axis=0), sim_cov, rf)
        sharpe_ratios.append(stats['sharpe'])
    return np.percentile(sharpe_ratios, [5, 50, 95])

# def factor_risk_decomposition(weights, returns):
#     try:
#         from pyanomaly import factors
#     except ImportError:
#         logging.info("Factor risk analysis is disabled.")
#         return None
#     fr_model = factors(
#         factors=['MKT-RF', 'SMB', 'HML', 'RMW', 'CMA', 'UMD'],
#         freq='D'
#     )
#     fr_model.fit(returns)
#     return fr_model.risk_decomposition(weights)

def get_optimized_weights(returns_subset, initial_guess, bounds, constraints, rf, current_value):
    dynamic_liquidity = {'type': 'ineq', 'fun': lambda w: liquidity_constraint(w, avg_volume_data, initial_prices_data, current_value, max_turnover=0.2)}
    dynamic_constraints = constraints.copy()
    dynamic_constraints.append(dynamic_liquidity)
    cov_matrix_subset = returns_subset.cov()
    res = minimize(objective_function, initial_guess, args=(returns_subset, cov_matrix_subset, rf),
                   method='SLSQP', bounds=bounds, constraints=dynamic_constraints)
    if not res.success:
        logging.warning("Rebalancing optimization failed; using equal weights.")
        return np.ones(len(initial_guess)) / len(initial_guess)
    return res.x / np.sum(res.x)

def simulate_portfolio(initial_investment, transaction_cost_rate, returns, historical_prices,
                       initial_guess, bounds, constraints, rf):
    current_value = initial_investment
    portfolio_values = []
    
    weekly_prices = historical_prices.resample('W-FRI').last()
    weekly_returns = weekly_prices.pct_change().dropna()
    weekly_dates = weekly_returns.index
    
    prev_weights = None
    for i, date in enumerate(weekly_dates):
        current_returns = returns.loc[:date]
        new_weights = get_optimized_weights(current_returns, initial_guess, bounds, constraints, rf, current_value)
        if prev_weights is not None:
            turnover = np.abs(new_weights - prev_weights).sum()
            transaction_cost = current_value * transaction_cost_rate * turnover
            current_value -= transaction_cost
        growth_factor = 1 + weekly_returns.iloc[i].dot(new_weights)
        current_value *= growth_factor
        portfolio_values.append(current_value)
        prev_weights = new_weights
    sim_dates = weekly_dates[:len(portfolio_values)]
    return pd.Series(portfolio_values, index=sim_dates)

def plot_additional_charts(weekly_returns, portfolio_values, drawdown_series):
    plt.figure(figsize=(10, 6))
    plt.hist(weekly_returns.dot(optimal_weights), bins=30, color='skyblue', edgecolor='black')
    plt.title("Histogram of Weekly Portfolio Returns")
    plt.xlabel("Weekly Return")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig("histogram_weekly_returns.png")
    plt.show()
    
    plt.figure(figsize=(10, 6))
    cumulative_returns = portfolio_values / portfolio_values.iloc[0] - 1
    plt.plot(cumulative_returns.index, cumulative_returns.values, label="Cumulative Return")
    plt.title("Cumulative Portfolio Return")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.savefig("cumulative_returns.png")
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.plot(drawdown_series.index, drawdown_series.values, label="Drawdown", color='red')
    plt.title("Portfolio Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.grid(True)
    plt.savefig("drawdown.png")
    plt.show()

# -----------------------------
# Benchmark Functions
# -----------------------------
def get_benchmark_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    return data['Adj Close']

def compute_benchmark_returns(ticker, start_date, end_date):
    bench_data = get_benchmark_data(ticker, start_date, end_date)
    bench_returns = bench_data.pct_change().dropna()
    cumulative = (bench_returns + 1).cumprod() - 1
    return bench_returns, cumulative

# -----------------------------
# Main Execution Block
# -----------------------------
if __name__ == "__main__":
    # Retrieve portfolio data
    historical_data = get_historical_data_with_retry(tickers, end_date)
    returns = historical_data.pct_change().dropna()

    historical_dates = historical_data.index
    initial_date_dt = pd.to_datetime(initial_investment_date)
    initial_date = historical_dates[historical_dates >= initial_date_dt][0].strftime("%Y-%m-%d")
    logging.info(f"Using initial_date = {initial_date}, end_date = {end_date}")

    weekly_prices = historical_data.resample('W-FRI').last()
    weekly_returns = weekly_prices.pct_change().dropna()

    num_assets = len(tickers)
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = np.array([1 / num_assets] * num_assets)
    base_constraint = {'type': 'eq', 'fun': constraint_sum}
    avg_volume_data = get_avg_daily_volume(tickers, end_date)
    if initial_date in historical_data.index:
        initial_prices_data = historical_data.loc[initial_date].values
    else:
        initial_prices_data = historical_data.iloc[0].values
    liquidity_constr = {'type': 'ineq', 'fun': lambda w: liquidity_constraint(w, avg_volume_data, initial_prices_data, initial_investment, max_turnover=0.2)}
    constraints = [base_constraint, liquidity_constr]

    cov_matrix = get_ewm_covariance(returns)
    
    bl_returns = get_black_litterman_returns(cov_matrix, returns)
    
    optimized = minimize(objective_function, initial_guess, args=(returns, cov_matrix, risk_free_rate),
                         method='SLSQP', bounds=bounds, constraints=constraints)
    if not optimized.success:
        logging.warning("Initial optimization failed; using equal weights.")
        optimal_weights = np.ones(num_assets) / num_assets
    else:
        optimal_weights = optimized['x'] / np.sum(optimized['x'])
    optimal_stats = calculate_portfolio_stats(optimal_weights, returns, cov_matrix, risk_free_rate)
    optimal_sortino = calculate_sortino_ratio(optimal_weights, returns, risk_free_rate)

    effective_investment = initial_investment * (1 - transaction_cost_rate)
    if initial_date in historical_data.index:
        initial_prices = historical_data.loc[initial_date]
    else:
        initial_prices = historical_data.iloc[0]
    if end_date in historical_data.index:
        end_prices = historical_data.loc[end_date]
    else:
        end_prices = historical_data.iloc[-1]
    
    shares = {}
    for ticker, weight in zip(tickers, optimal_weights):
        allocation = weight * effective_investment
        share_price = initial_prices[ticker]
        shares[ticker] = allocation / share_price

    initial_portfolio = {}
    end_portfolio = {}
    for ticker in tickers:
        initial_portfolio[ticker] = {
            'Shares': shares[ticker],
            'Price': initial_prices[ticker],
            'Value': shares[ticker] * initial_prices[ticker],
            'Weight': optimal_weights[tickers.index(ticker)] * 100
        }
        end_portfolio[ticker] = {
            'Shares': shares[ticker],
            'Price': end_prices[ticker],
            'Value': shares[ticker] * end_prices[ticker]
        }
    
    initial_df = pd.DataFrame.from_dict(initial_portfolio, orient='index')
    initial_df['Weight'] = initial_df['Weight'].round(2)
    initial_df = initial_df.sort_values('Value', ascending=False)
    
    initial_value = sum(item['Value'] for item in initial_portfolio.values())
    end_value = sum(item['Value'] for item in end_portfolio.values())
    portfolio_return = (end_value - initial_value) / initial_value
    days_invested = (datetime.strptime(end_date, "%Y-%m-%d") - datetime.strptime(initial_date, "%Y-%m-%d")).days
    annualized_return = portfolio_return * (365 / days_invested)
    
    portfolio_value_over_time = (historical_data.loc[initial_date:end_date] * pd.Series(shares)).sum(axis=1)
    max_drawdown = calculate_max_drawdown(portfolio_value_over_time)
    
    weekly_perf = weekly_returns.copy()
    weekly_perf['Portfolio Return (%)'] = weekly_returns.dot(optimal_weights) * 100
    logging.info("Weekly Portfolio Performance:")
    for date, row in weekly_perf.iterrows():
        logging.info(f"{date.date()}: {row['Portfolio Return (%)']:.2f}%")
    
    drawdown_series = (portfolio_value_over_time - portfolio_value_over_time.cummax()) / portfolio_value_over_time.cummax()
    plot_additional_charts(weekly_returns, portfolio_value_over_time, drawdown_series)
    
    def generate_efficient_frontier(returns, cov_matrix, rf, num_portfolios=1000):
        num_assets = len(returns.columns)
        results = np.zeros((3, num_portfolios))
        weights_record = []
        for i in range(num_portfolios):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            weights_record.append(weights)
            stats = calculate_portfolio_stats(weights, returns, cov_matrix, rf)
            results[0, i] = float(stats['std_dev'])
            results[1, i] = float(stats['return'])
            results[2, i] = float(stats['sharpe'])
        return results, weights_record
    
    results_frontier, weights_record = generate_efficient_frontier(returns, cov_matrix, risk_free_rate)
    plt.figure(figsize=(10, 6))
    plt.scatter(results_frontier[0, :], results_frontier[1, :], c=results_frontier[2, :],
                cmap='viridis', marker='o', s=10, alpha=0.3)
    plt.colorbar(label='Sharpe Ratio')
    plt.scatter(optimal_stats['std_dev'], optimal_stats['return'], c='red', marker='*', s=300, label='Optimal Portfolio')
    max_sharpe_idx = np.argmax(results_frontier[2, :])
    max_sharpe_std = float(results_frontier[0, max_sharpe_idx])
    max_sharpe_return = float(results_frontier[1, max_sharpe_idx])
    cml_x = np.linspace(0, max_sharpe_std * 2, 100)
    cml_y = rf_scalar + (max_sharpe_return - rf_scalar) / max_sharpe_std * cml_x
    plt.plot(cml_x, cml_y, 'r--', label='Capital Market Line')
    plt.title('Efficient Frontier')
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.legend()
    plt.grid(True)
    plt.savefig("efficient_frontier.png")
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_value_over_time.index, portfolio_value_over_time.values, label='Portfolio Value')
    plt.title('Portfolio Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig("portfolio_equity_curve.png")
    plt.show()
    
    stock_metrics = {}
    for ticker in tickers:
        ticker_returns = returns[ticker]
        annual_return = ticker_returns.mean() * 252
        std_dev = ticker_returns.std() * np.sqrt(252)
        ticker_beta = calculate_betas(returns[ticker].to_frame())[0] if isinstance(returns, pd.DataFrame) else np.nan
        sharpe = (annual_return - float(risk_free_rate)) / std_dev if std_dev != 0 else np.nan
        stock_metrics[ticker] = {
            'Avg Annual Return (%)': annual_return * 100,
            'Beta': ticker_beta,
            'Std Dev (%)': std_dev * 100,
            'Sharpe': sharpe,
            'Avg Daily Volume (Mill)': get_avg_daily_volume([ticker], end_date)[0]
        }
    metrics_df = pd.DataFrame.from_dict(stock_metrics, orient='index')
    metrics_df = metrics_df.sort_values('Sharpe', ascending=False)
    
    logging.info("Optimal Portfolio Allocation (Maximum Sharpe Ratio):")
    for ticker, weight in sorted(zip(tickers, optimal_weights), key=lambda x: x[1], reverse=True):
        logging.info(f"{ticker}: {weight * 100:.2f}%")
    
    logging.info("Portfolio Statistics:")
    logging.info(f"Expected Annual Return: {optimal_stats['return'] * 100:.2f}%")
    logging.info(f"Expected Annual Volatility: {optimal_stats['std_dev'] * 100:.2f}%")
    logging.info(f"Expected Sharpe Ratio: {float(optimal_stats['sharpe']):.2f}")
    logging.info(f"Expected Sortino Ratio: {optimal_sortino:.2f}")
    logging.info(f"Portfolio Beta: {optimal_stats['beta']:.2f}")
    
    logging.info(f"Initial Portfolio ({initial_date}):")
    logging.info(initial_df[['Shares', 'Price', 'Value', 'Weight']])
    logging.info(f"Total Portfolio Value: ${initial_value:.2f}")
    
    logging.info(f"End Portfolio ({end_date}):")
    end_df = pd.DataFrame.from_dict(end_portfolio, orient='index')
    end_df = end_df.sort_values('Value', ascending=False)
    logging.info(end_df)
    logging.info(f"Total Portfolio Value: ${end_value:.2f}")
    logging.info(f"Portfolio Return: {portfolio_return * 100:.2f}%")
    logging.info(f"Annualized Return: {annualized_return * 100:.2f}%")
    logging.info(f"Maximum Drawdown: {max_drawdown * 100:.2f}%")
    
    logging.info("Stock Metrics (5-Year Historical Data):")
    logging.info(metrics_df)
    
    pareto_results = pareto_frontier_optimization(returns, cov_matrix, risk_free_rate)
    if pareto_results is not None:
        logging.info("Pareto Frontier Solutions (weights):")
        logging.info(pareto_results.X)
    
    wf_results = walk_forward_optimization(returns, n_splits=5, initial_guess=initial_guess, bounds=bounds, constraints=constraints, rf=risk_free_rate)
    logging.info("Walk–Forward Optimization Results:")
    for idx, res in enumerate(wf_results):
        logging.info(f"Split {idx+1}: Optimal Weights = {res.x}")
    
    crisis_stats = crisis_simulation(optimal_weights, returns)
    if crisis_stats is not None:
        logging.info("Portfolio Stress Test (Crisis Simulation) Statistics:")
        logging.info(crisis_stats)
    
    bayes_opt_weights = bayesian_optimization(returns, cov_matrix, optimal_weights, max_turnover=0.3, tickers=tickers, rf=risk_free_rate)
    logging.info("Bayesian Optimization Weights (with turnover control):")
    logging.info(bayes_opt_weights)
    
    mc_results = monte_carlo_validation(optimal_weights, n_simulations=1000, returns=returns, cov_matrix=cov_matrix, rf=risk_free_rate)
    logging.info("Monte Carlo Robustness Testing Sharpe Ratio Percentiles (5th, 50th, 95th):")
    logging.info(mc_results)
    
    # fr_decomp = factor_risk_decomposition(optimal_weights, returns)
    # if fr_decomp is not None:
    #     logging.info("Factor Risk Decomposition:")
    #     logging.info(fr_decomp)
    
    regimes = detect_market_regimes(returns)
    current_regime = regimes[-1] if len(regimes) > 0 else 0
    regime_opt_result = regime_aware_optimization(returns, current_regime, initial_guess, bounds, constraints, risk_free_rate)
    logging.info("Regime Aware Optimization Optimal Weights:")
    logging.info(regime_opt_result.x)
    
    simulated_portfolio = simulate_portfolio(initial_investment, transaction_cost_rate, returns, historical_data,
                                               initial_guess, bounds, constraints, risk_free_rate)
    logging.info("Simulated Portfolio Value Over Time (Weekly Rebalancing with Transaction Costs):")
    logging.info(simulated_portfolio)
    
    # -----------------------------
    # Benchmark Comparisons
    # -----------------------------
    benchmark_tickers = {"S&P500": "^GSPC", "DowJones": "^DJI", "Nasdaq": "^IXIC"}
    benchmark_data = {}
    benchmark_cum_returns = {}
    for name, ticker in benchmark_tickers.items():
        bench_prices = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)['Adj Close']
        bench_returns = bench_prices.pct_change().dropna()
        cum_return = (bench_returns + 1).cumprod() - 1
        benchmark_data[name] = bench_prices
        benchmark_cum_returns[name] = cum_return
        bench_prices.to_csv(f"{name}_prices.csv")
        cum_return.to_csv(f"{name}_cumulative_returns.csv")
    
    portfolio_cum_return = (portfolio_value_over_time / portfolio_value_over_time.iloc[0]) - 1
    comparison_df = pd.DataFrame({"Portfolio": portfolio_cum_return})
    for name, cum_ret in benchmark_cum_returns.items():
        comparison_df[name] = cum_ret.reindex(comparison_df.index).ffill()
    plt.figure(figsize=(12, 8))
    for col in comparison_df.columns:
        plt.plot(comparison_df.index, comparison_df[col], label=col)
    plt.title("Cumulative Returns Comparison: Portfolio vs Benchmarks")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.savefig("cumulative_returns_comparison.png")
    plt.show()
    comparison_df.to_csv("benchmark_comparison.csv", index=True)
    
    # -----------------------------
    # Save CSV Outputs
    # -----------------------------
    initial_df.to_csv('initial_portfolio.csv', index=True)
    end_df.to_csv('end_portfolio.csv', index=True)
    metrics_df.to_csv('stock_metrics.csv', index=True)
    pd.DataFrame(results_frontier.T, columns=['Volatility', 'Return', 'Sharpe']).to_csv('efficient_frontier.csv', index=False)
    weekly_perf.to_csv('weekly_returns.csv', index=True)
    portfolio_value_over_time.to_csv('portfolio_equity_curve.csv', index=True)
    simulated_portfolio.to_csv('simulated_portfolio.csv', index=True)
    comparison_df.to_csv("benchmark_comparison.csv", index=True)

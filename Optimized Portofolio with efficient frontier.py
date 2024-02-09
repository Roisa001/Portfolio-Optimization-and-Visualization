#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 23:01:19 2024

@author: robertisaksen
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Data paths for each asset
data_paths = [
    "~/Downloads/Download Data - STOCK_US_XNYS_PG.csv",
    "~/Downloads/Download Data - STOCK_US_XNAS_AAPL.csv",
    "~/Downloads/Download Data - STOCK_US_XNAS_MSFT.csv",
    "~/Downloads/Download Data - STOCK_US_XNYS_KO.csv",
    "~/Downloads/Download Data - STOCK_US_XNYS_NKE.csv",
    "~/Downloads/Download Data - STOCK_US_XNYS_BA.csv",
    "~/Downloads/Download Data - STOCK_US_XNYS_JPM.csv",
    "~/Downloads/Download Data - STOCK_US_XNYS_GS.csv"
]

# Load data for each asset
data_list = []
for path in data_paths:
    data = pd.read_csv(path)
    data_list.append(data)

# Extract 'Close' prices for each asset
close_prices = pd.concat([data['Close'] for data in data_list], axis=1)
close_prices.columns = [f"Asset_{i+1}" for i in range(len(data_list))]

# Calculate returns
returns = close_prices.pct_change().dropna()

# Calculate covariance matrix
covariance_matrix = returns.cov()

# Calculate correlation matrix
correlation_matrix = returns.corr()

# Display covariance matrix
print("Covariance Matrix:")
print(covariance_matrix)

# Display correlation matrix
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Plot correlation matrix
plt.figure(figsize=(10, 8))
img = plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none', vmin=-1, vmax=1)
plt.colorbar(img, ticks=np.linspace(-1, 1, num=41))  # Adjust ticks for quicker color change
plt.title('Correlation Matrix')
plt.xticks(np.arange(len(correlation_matrix)), correlation_matrix.columns, rotation=90)
plt.yticks(np.arange(len(correlation_matrix)), correlation_matrix.columns)
plt.show()


# Number of assets
num_assets = len(data_list)

# Calculate expected returns
expected_returns = returns.mean()

# Function to calculate portfolio returns and volatility
def calculate_portfolio_performance(weights, expected_returns, covariance_matrix):
    portfolio_return = np.sum(expected_returns * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
    return portfolio_return, portfolio_volatility

# Generate random portfolio weights
def generate_random_weights(num_portfolios, num_assets):
    random_weights = np.random.random((num_portfolios, num_assets))
    random_weights /= np.sum(random_weights, axis=1)[:, np.newaxis]
    return random_weights

# Number of portfolios to simulate
num_portfolios = 10000

# Generate random portfolio weights
random_weights = generate_random_weights(num_portfolios, num_assets)

# Calculate portfolio returns and volatilities for random weights
portfolio_returns = []
portfolio_volatilities = []
for weights in random_weights:
    portfolio_return, portfolio_volatility = calculate_portfolio_performance(weights, expected_returns, covariance_matrix)
    portfolio_returns.append(portfolio_return)
    portfolio_volatilities.append(portfolio_volatility)

# Convert lists to numpy arrays
portfolio_returns = np.array(portfolio_returns)
portfolio_volatilities = np.array(portfolio_volatilities)

# Plot the efficient frontier
plt.figure(figsize=(10, 6))
plt.scatter(portfolio_volatilities, portfolio_returns, c=portfolio_returns / portfolio_volatilities, marker='o', cmap='viridis')
plt.title('Efficient Frontier')
plt.xlabel('Volatility')
plt.ylabel('Expected Return')
plt.colorbar(label='Sharpe Ratio')
plt.grid(True)
plt.show()

# Function to calculate negative Sharpe ratio (for minimization)
def neg_sharpe_ratio(weights, expected_returns, covariance_matrix):
    portfolio_return, portfolio_volatility = calculate_portfolio_performance(weights, expected_returns, covariance_matrix)
    return -portfolio_return / portfolio_volatility

# Function for portfolio optimization
def optimize_portfolio(expected_returns, covariance_matrix):
    num_assets = len(expected_returns)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    initial_guess = num_assets * [1. / num_assets,]
    optimized_portfolio = minimize(neg_sharpe_ratio, initial_guess, args=(expected_returns, covariance_matrix,),
                                   method='SLSQP', bounds=bounds, constraints=constraints)
    return optimized_portfolio

# Perform optimization
optimized_portfolio = optimize_portfolio(expected_returns, covariance_matrix)

# Define asset names corresponding to data paths
asset_names = [
    "STOCK_US_XNYS_PG",
    "STOCK_US_XNAS_AAPL",
    "STOCK_US_XNAS_MSFT",
    "STOCK_US_XNYS_KO",
    "STOCK_US_XNYS_NKE",
    "STOCK_US_XNYS_BA",
    "STOCK_US_XNYS_JPM",
    "STOCK_US_XNYS_GS",
    "STOCK_US_XNAS_TSLA"
]

# Display optimized portfolio allocation
print("\nOptimized Portfolio Allocation:")
for i, weight in enumerate(optimized_portfolio.x):
    print(f"{asset_names[i]} Weight: {weight:.2f}")

# Display optimized portfolio expected return and volatility
optimal_return, optimal_volatility = calculate_portfolio_performance(optimized_portfolio.x, expected_returns, covariance_matrix)
print(f"\nOptimized Portfolio Expected Return: {optimal_return:.4f}")
print(f"Optimized Portfolio Volatility: {optimal_volatility:.4f}")

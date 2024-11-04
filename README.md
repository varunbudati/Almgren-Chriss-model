# Optimal Trading Execution System with Reinforcement Learning
This project implements an enhanced version of the Almgren-Chriss optimal trading model using reinforcement learning, based on the paper "A reinforcement learning extension to the Almgren-Chriss framework for optimal trade execution" by Hendricks and Wilcox.
Overview
The system provides a web-based interface for optimal trade execution, combining traditional analytical methods with machine learning to improve trading performance. It features:

## Implementation of the Almgren-Chriss model
Reinforcement learning enhancement
Real-time market data integration
Order book visualization
Performance analytics
Backtesting capabilities

## Features

### Market Data Analysis

Real-time data fetching
Price history visualization
Market statistics


### Trading Strategies

Base Almgren-Chriss implementation
RL-enhanced execution
Comparative analysis


### Visualization

Order book depth
Trading trajectories
Performance metrics


### Analytics

Implementation shortfall calculation
Cost analysis
Performance comparison

![chrome_KN8axHwwfF](https://github.com/user-attachments/assets/e3a4f02a-7f65-45a4-84a3-c0f8ab3f7c76)
## Requirements

Python 3.8+
streamlit
yfinance
pandas
numpy
matplotlib
seaborn
scipy

## Model Parameters

### σ (Volatility): Market price volatility
### η (Temporary Impact): Immediate price impact parameter
### λ (Risk Aversion): Trade-off between execution cost and risk
### T (Time Horizon): Trading period length
### X (Trade Size): Total volume to execute

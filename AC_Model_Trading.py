import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import yfinance as yf  # for market data

class AlmgrenChrissModel:
    def __init__(self, sigma, eta, lambda_param, T, N):
        self.sigma = sigma  # volatility
        self.eta = eta      # temporary impact
        self.lambda_param = lambda_param  # risk aversion
        self.T = T          # time horizon
        self.N = N          # number of periods
        self.tau = T/N      # time step size
        
    def calculate_trajectory(self, X):
        # Implementation of AC trajectory calculation
        kappa = self.calculate_kappa()
        times = np.linspace(0, self.T, self.N+1)
        
        # Calculate remaining shares at each time point
        trajectory = np.array([
            (np.sinh(kappa * (self.T - t)) / np.sinh(kappa * self.T)) * X 
            for t in times
        ])
        
        # Calculate trades at each time point
        trades = -np.diff(trajectory)
        return trajectory, trades
    
    def calculate_kappa(self):
        # Calculate kappa parameter
        kappa_tilde = np.sqrt(self.lambda_param * self.sigma**2 / self.eta)
        return kappa_tilde

class QLearningAgent: #Re-enforcement learning agent - Better than Dynamic Programming
    def __init__(self, states, actions, alpha=0.1, gamma=1.0):
        self.Q = defaultdict(float)
        self.states = states
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
    
    def update(self, state, action, reward, next_state):
        # Q-learning update rule
        best_next_value = max([self.Q[(next_state, a)] for a in self.actions])
        self.Q[(state, action)] += self.alpha * (
            reward + self.gamma * best_next_value - self.Q[(state, action)]
        )
    
    def get_best_action(self, state):
        # Get action with highest Q-value
        return max(self.actions, key=lambda a: self.Q[(state, a)])

def main():
    st.title("Optimal Trading Execution System")
    
    # Sidebar inputs
    st.sidebar.header("Model Parameters")
    sigma = st.sidebar.slider("Volatility (σ)", 0.0, 1.0, 0.2)
    eta = st.sidebar.slider("Temporary Impact (η)", 0.0, 1.0, 0.1)
    lambda_param = st.sidebar.slider("Risk Aversion (λ)", 0.0, 1.0, 0.1)
    
    # Trading parameters
    T = st.sidebar.selectbox("Time Horizon (minutes)", [20, 40, 60])
    N = T // 5  # 5-minute intervals
    X = st.sidebar.number_input("Total Shares to Trade", 1000, 1000000, 100000)
    
    # Initialize models
    ac_model = AlmgrenChrissModel(sigma, eta, lambda_param, T/60, N)
    
    # Calculate AC trajectory
    trajectory, trades = ac_model.calculate_trajectory(X)
    
    # Display results
    st.header("Trading Trajectory")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    times = np.linspace(0, T, N+1)
    ax.plot(times, trajectory, 'b-', label='Remaining Shares')
    ax.bar(times[:-1], trades, width=T/N/2, alpha=0.3, label='Trade Size')
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Shares')
    ax.legend()
    st.pyplot(fig)
    
    # Display trade list
    st.header("Trade List")
    trade_df = pd.DataFrame({
        'Time': [f"{t:.1f} min" for t in times[:-1]],
        'Shares to Trade': trades.astype(int)
    })
    st.dataframe(trade_df)
    
    # RL Enhancement section
    if st.checkbox("Enable RL Enhancement"):
        st.subheader("RL-Enhanced Trading")
        # Here you would implement the RL modification of the AC trajectory
        # Based on market conditions (spread, volume states)
        st.info("RL enhancement would modify these trades based on real-time market conditions")

if __name__ == "__main__":
    main()
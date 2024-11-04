import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from collections import defaultdict
import datetime
from scipy.stats import norm
import seaborn as sns


class AlmgrenChrissModel:
    def __init__(self, sigma, eta, lambda_param, T, N):
        self.sigma = sigma  # volatility
        self.eta = eta      # temporary impact
        self.lambda_param = lambda_param  # risk aversion
        self.T = T          # time horizon
        self.N = N          # number of periods
        self.tau = T/N      # time step size
        
    def calculate_trajectory(self, X):
        """Calculate the optimal trading trajectory"""
        # Calculate kappa
        kappa = self.calculate_kappa()
        
        # Create time points
        times = np.linspace(0, self.T, self.N+1)
        
        # Calculate remaining shares at each time point
        trajectory = np.array([
            (np.sinh(kappa * (self.T - t)) / np.sinh(kappa * self.T)) * X 
            for t in times
        ])
        
        # Calculate trades at each time point (negative diff for selling)
        trades = -np.diff(trajectory)
        
        # Ensure trades sum to total volume (handle numerical errors)
        trades = trades * (X / trades.sum())
        
        return trajectory, trades
    
    def calculate_kappa(self):
        """Calculate the kappa parameter"""
        # Avoid division by zero
        eta = max(self.eta, 1e-8)
        
        # Calculate kappa
        kappa_tilde = np.sqrt(self.lambda_param * (self.sigma**2) / eta)
        kappa = (1/self.tau) * np.arccosh(1 + (self.tau**2 * kappa_tilde**2)/2)
        
        return kappa

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

class MarketData:
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.order_book = None
        
    def fetch_data(self):
        # Fetch data from yfinance
        self.data = yf.download(self.symbol, self.start_date, self.end_date, interval='5m')
        return self.data
    
    def generate_synthetic_order_book(self, price, timestamp):
        """Generate synthetic order book since real L2 data isn't freely available, so I can't really calculate accurate order book or get good data"""
        spread = price * 0.001  # 0.1% spread
        levels = 5
        
        asks = []
        bids = []
        
        # Generate ask levels
        for i in range(levels):
            ask_price = price + spread * (i + 1)
            ask_volume = np.random.poisson(1000 * (1 / (i + 1)))
            asks.append({'price': ask_price, 'volume': ask_volume})
            
        # Generate bid levels
        for i in range(levels):
            bid_price = price - spread * (i + 1)
            bid_volume = np.random.poisson(1000 * (1 / (i + 1)))
            bids.append({'price': bid_price, 'volume': bid_volume})
            
        return {
            'timestamp': timestamp,
            'asks': asks,
            'bids': bids
        }

class OrderBook:
    def __init__(self):
        self.asks = []
        self.bids = []
        
    def update(self, order_book_data):
        self.asks = order_book_data['asks']
        self.bids = order_book_data['bids']
        
    def get_vwap(self, side, volume):
        """Calculate VWAP for given volume"""
        levels = self.asks if side == 'buy' else self.bids
        remaining_volume = volume
        total_cost = 0
        
        for level in levels:
            trade_volume = min(remaining_volume, level['volume'])
            total_cost += trade_volume * level['price']
            remaining_volume -= trade_volume
            
            if remaining_volume <= 0:
                break
                
        return total_cost / volume if volume > 0 else 0
class BacktestEngine:
    def __init__(self, market_data, ac_model, rl_agent):
        self.market_data = market_data
        self.ac_model = ac_model
        self.rl_agent = rl_agent
        self.order_book = OrderBook()
        
    def run_backtest(self, initial_shares, start_time, end_time):
        # Get AC trajectory first
        _, ac_trades = self.ac_model.calculate_trajectory(initial_shares)
        
        # Get trading periods from market data
        trading_periods = self.market_data.data[start_time:end_time].index
        
        # Determine number of periods to use
        n_periods = min(len(ac_trades), len(trading_periods))
        
        # Truncate both sequences to same length
        trading_periods = trading_periods[:n_periods]
        ac_trades = ac_trades[:n_periods]
        
        # Initialize results
        results = {
            'timestamp': [],
            'shares_traded': [],
            'price': [],
            'cost': [],
            'implementation_shortfall': [],
            'remaining_shares': []
        }
        
        # Get AC trajectory
        _, ac_trades = self.ac_model.calculate_trajectory(initial_shares)
        
        # Initialize tracking variables
        remaining_shares = initial_shares
        cumulative_cost = 0
        reference_price = self.market_data.data.loc[start_time, 'Close']
        
        # Get relevant time periods from market data
        trading_periods = self.market_data.data[start_time:end_time].index
        
        # Ensure we have the right number of periods
        n_periods = min(len(ac_trades), len(trading_periods))
        
        for i in range(n_periods):
            timestamp = trading_periods[i]
            ac_trade = ac_trades[i]
            
            # Get current market state
            current_price = self.market_data.data.loc[timestamp, 'Close']
            current_state = self.get_state(timestamp, remaining_shares)
            
            # Update order book
            self.order_book.update(
                self.market_data.generate_synthetic_order_book(
                    current_price, timestamp
                )
            )
            
            # Get RL action and modify AC trade
            action = self.rl_agent.get_best_action(current_state)
            modified_trade = ac_trade * action
            
            # Ensure we don't trade more than remaining shares
            modified_trade = min(modified_trade, remaining_shares)
            
            # Execute trade
            execution_price = self.order_book.get_vwap('sell', modified_trade)
            trade_cost = modified_trade * execution_price
            
            # Update tracking variables
            remaining_shares -= modified_trade
            cumulative_cost += trade_cost
            
            # Calculate implementation shortfall for this trade
            shortfall = (execution_price - reference_price) * modified_trade / (reference_price * initial_shares)
            
            # Record results
            results['timestamp'].append(timestamp)
            results['shares_traded'].append(modified_trade)
            results['price'].append(execution_price)
            results['cost'].append(trade_cost)
            results['implementation_shortfall'].append(shortfall)
            results['remaining_shares'].append(remaining_shares)
            
        # Handle any remaining shares in the last period
        if remaining_shares > 0:
            final_timestamp = trading_periods[-1]
            final_price = self.order_book.get_vwap('sell', remaining_shares)
            final_cost = remaining_shares * final_price
            final_shortfall = (final_price - reference_price) * remaining_shares / (reference_price * initial_shares)
            
            results['timestamp'].append(final_timestamp)
            results['shares_traded'].append(remaining_shares)
            results['price'].append(final_price)
            results['cost'].append(final_cost)
            results['implementation_shortfall'].append(final_shortfall)
            results['remaining_shares'].append(0)
        
        # Convert results to DataFrame
        df_results = pd.DataFrame(results)
        df_results.set_index('timestamp', inplace=True)
        
        return df_results
    
    def get_state(self, timestamp, remaining_shares):
        """Convert market conditions to state space"""
        data = self.market_data.data.loc[timestamp]
        
        # Calculate spread and volume states
        spread = (data['High'] - data['Low']) / data['Close']
        volume = data['Volume']
        
        # Discretize state space
        spread_state = min(int(np.floor(spread * 10)), 9)  # 0-9 states
        volume_state = min(int(np.floor(np.log(volume) / np.log(10))), 9)  # 0-9 states
        shares_state = min(int(np.floor(remaining_shares / 1000)), 9)  # 0-9 states
        
        return (spread_state, volume_state, shares_state)

class PerformanceMetrics:
    @staticmethod
    def calculate_metrics(backtest_results, benchmark_price):
        metrics = {}
        
        # Implementation Shortfall
        metrics['implementation_shortfall'] = (
            (backtest_results['price'] * backtest_results['shares_traded']).sum() -
            benchmark_price * backtest_results['shares_traded'].sum()
        ) / (benchmark_price * backtest_results['shares_traded'].sum())
        
        # VWAP Slippage
        vwap = (backtest_results['price'] * backtest_results['shares_traded']).sum() / backtest_results['shares_traded'].sum()
        metrics['vwap_slippage'] = (vwap - benchmark_price) / benchmark_price
        
        # Volatility of execution prices
        metrics['price_volatility'] = backtest_results['price'].std() / backtest_results['price'].mean()
        
        return metrics
def main():
    st.title("Advanced Trading Execution System")
    
    # Sidebar - Parameters
    st.sidebar.header("Trading Parameters")
    symbol = st.sidebar.text_input("Stock Symbol", "AAPL")
    start_date = st.sidebar.date_input("Start Date", datetime.date.today() - datetime.timedelta(days=30))
    end_date = st.sidebar.date_input("End Date", datetime.date.today())
    
    # Model parameters
    st.sidebar.header("Model Parameters")
    sigma = st.sidebar.slider("Volatility (σ)", 0.0, 1.0, 0.2)
    eta = st.sidebar.slider("Temporary Impact (η)", 0.0, 1.0, 0.1)
    lambda_param = st.sidebar.slider("Risk Aversion (λ)", 0.0, 1.0, 0.1)
    
    # Trading parameters
    T = st.sidebar.selectbox("Time Horizon (minutes)", [20, 40, 60])
    N = T // 5  # 5-minute intervals
    X = st.sidebar.number_input("Total Shares to Trade", 1000, 1000000, 100000)

    # Initialize components
    market_data = MarketData(symbol, start_date, end_date)
    
    # Fetch data first
    try:
        with st.spinner("Fetching market data..."):
            data = market_data.fetch_data()
            
        if data.empty:
            st.error(f"No data available for {symbol} in the specified date range.")
            return
            
        # Initialize models after we have data
        ac_model = AlmgrenChrissModel(sigma, eta, lambda_param, T/60, N)
        rl_agent = QLearningAgent(states=None, actions=np.linspace(0.5, 1.5, 5))
        backtest_engine = BacktestEngine(market_data, ac_model, rl_agent)
        
        # Display market data overview
        st.header("Market Data Overview")
        fig_overview, ax_overview = plt.subplots(figsize=(10, 6))
        ax_overview.plot(data.index, data['Close'], label='Close Price')
        ax_overview.set_xlabel('Date')
        ax_overview.set_ylabel('Price')
        ax_overview.set_title(f'{symbol} Price History')
        ax_overview.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig_overview)
        
        # Add price statistics
        price_stats = {
            'Current Price': f"${data['Close'].iloc[-1]:.2f}",
            'Average Price': f"${data['Close'].mean():.2f}",
            'Price Range': f"${data['Close'].min():.2f} - ${data['Close'].max():.2f}",
            'Volatility': f"{data['Close'].std()/data['Close'].mean()*100:.2f}%"
        }
        st.write("### Price Statistics")
        stats_df = pd.DataFrame(list(price_stats.items()), columns=['Metric', 'Value'])
        st.table(stats_df)
        
        if st.button("Run Trading Analysis"):
            with st.spinner("Running analysis..."):
                # Order Book Visualization
                st.header("Order Book Visualization")
                sample_ob = market_data.generate_synthetic_order_book(
                    data['Close'].iloc[-1],
                    data.index[-1]
                )
                
                fig_ob, ax_ob = plt.subplots(figsize=(10, 6))
                ask_prices = [level['price'] for level in sample_ob['asks']]
                ask_volumes = [level['volume'] for level in sample_ob['asks']]
                bid_prices = [level['price'] for level in sample_ob['bids']]
                bid_volumes = [level['volume'] for level in sample_ob['bids']]
                
                ax_ob.barh(range(len(ask_prices)), ask_volumes, alpha=0.3, color='red', label='Ask')
                ax_ob.barh(range(len(bid_prices)), [-v for v in bid_volumes], alpha=0.3, color='green', label='Bid')
                ax_ob.set_yticks(range(len(ask_prices)))
                ax_ob.set_yticklabels([f"${p:.2f}" for p in ask_prices])
                ax_ob.set_title("Current Order Book State")
                ax_ob.set_xlabel("Volume")
                ax_ob.set_ylabel("Price")
                ax_ob.legend()
                plt.tight_layout()
                st.pyplot(fig_ob)
                
                # Trading trajectory analysis
                st.header("Trading Strategies Comparison")
                
                # Run backtest
                backtest_results = backtest_engine.run_backtest(
                    X, data.index[0], data.index[min(len(data)-1, N-1)]
                )
                
                # Get AC trajectory with matching length
                _, ac_trades = ac_model.calculate_trajectory(X)
                
                # Ensure AC trades match the backtest length
                min_length = min(len(ac_trades), len(backtest_results))
                ac_trades = ac_trades[:min_length]
                backtest_results = backtest_results.iloc[:min_length]
                
                # Plot trading trajectories
                fig_traj, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
                
                # Plot trade sizes
                ax1.bar(range(len(backtest_results)), backtest_results['shares_traded'], 
                       alpha=0.5, label='RL Trades', color='blue')
                ax1.bar(range(len(ac_trades)), ac_trades, 
                       alpha=0.5, label='AC Trades', color='orange')
                ax1.set_title("Individual Trade Sizes")
                ax1.set_xlabel("Time Period")
                ax1.set_ylabel("Shares")
                ax1.legend()
                
                # Plot cumulative trades
                ax2.plot(range(len(backtest_results)), 
                        backtest_results['shares_traded'].cumsum(),
                        label='RL Cumulative', linewidth=2)
                ax2.plot(range(len(ac_trades)), 
                        np.cumsum(ac_trades),
                        label='AC Cumulative', linewidth=2)
                ax2.set_title("Cumulative Trades")
                ax2.set_xlabel("Time Period")
                ax2.set_ylabel("Cumulative Shares")
                ax2.legend()
                
                plt.tight_layout()
                st.pyplot(fig_traj)

                # Performance Metrics
                # Replace the performance metrics section in main() with this:

                # Performance comparison
                st.header("Performance Metrics")

                try:
                    # Calculate metrics for both strategies
                    ac_costs = ac_trades * backtest_results['price'].values
                    initial_price = data['Close'].iloc[0]
                    
                    # Calculate implementation shortfall
                    ac_shortfall = (ac_costs.sum() - (initial_price * sum(ac_trades))) / (initial_price * X)
                    rl_shortfall = backtest_results['implementation_shortfall'].sum()
                    
                    # Calculate other metrics
                    rl_avg_price = (backtest_results['price'] * backtest_results['shares_traded']).sum() / backtest_results['shares_traded'].sum()
                    ac_avg_price = ac_costs.sum() / ac_trades.sum()
                    price_vol = backtest_results['price'].std() / backtest_results['price'].mean()
                    rl_total_shares = int(backtest_results['shares_traded'].sum())
                    ac_total_shares = int(ac_trades.sum())
                    
                    # Create metrics list with manually formatted strings
                    metrics_list = [
                        {"Metric": "Implementation Shortfall (RL)", "Value": f"{rl_shortfall:.4%}"},
                        {"Metric": "Implementation Shortfall (AC)", "Value": f"{ac_shortfall:.4%}"},
                        {"Metric": "Average Price (RL)", "Value": f"${rl_avg_price:.2f}"},
                        {"Metric": "Average Price (AC)", "Value": f"${ac_avg_price:.2f}"},
                        {"Metric": "Price Volatility", "Value": f"{price_vol:.4f}"},
                        {"Metric": "Total Shares Traded (RL)", "Value": f"{rl_total_shares:,}"},
                        {"Metric": "Total Shares Traded (AC)", "Value": f"{ac_total_shares:,}"}
                    ]
                    
                    if ac_shortfall != 0:
                        improvement = (ac_shortfall - rl_shortfall)/ac_shortfall
                        metrics_list.append({
                            "Metric": "Improvement",
                            "Value": f"{improvement:.2%}"
                        })
                    
                    # Convert to DataFrame
                    metrics_df = pd.DataFrame(metrics_list)
                    st.table(metrics_df)

                    # Prepare download data
                    download_data = {
                        'Time': backtest_results.index,
                        'RL_Trades': [f"{x:.2f}" for x in backtest_results['shares_traded']],
                        'AC_Trades': [f"{x:.2f}" for x in ac_trades],
                        'Price': [f"{x:.2f}" for x in backtest_results['price']],
                        'RL_Cost': [f"{x:.2f}" for x in backtest_results['cost']],
                        'AC_Cost': [f"{x:.2f}" for x in ac_costs],
                        'RL_Cumulative': [f"{x:.2f}" for x in backtest_results['shares_traded'].cumsum()],
                        'AC_Cumulative': [f"{x:.2f}" for x in np.cumsum(ac_trades)]
                    }
                    
                    combined_results = pd.DataFrame(download_data)
                    
                    st.download_button(
                        label="Download Trading Results",
                        data=combined_results.to_csv(index=True),
                        file_name=f"trading_results_{symbol}_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.error("Please check your inputs and try again.")
                    
                    # Debug information
                    st.write("Debug Information:")
                    st.write("Shapes:")
                    st.write(f"AC trades shape: {len(ac_trades)}")
                    st.write(f"Backtest results shape: {len(backtest_results)}")
                    st.write("\nData Types:")
                    st.write(backtest_results.dtypes)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check your inputs and try again.")

if __name__ == "__main__":
    main()
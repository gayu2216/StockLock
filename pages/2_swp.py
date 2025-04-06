import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai

st.set_page_config(page_title="DCA Investment Calculator", layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #FFDBED, #ffffff);
        padding: 40px;  # Increased padding
    }
    .analysis-box {
        background-color: #f8f9fa;
        border-left: 4px solid #4682B4;
        padding: 15px;
        margin-bottom: 20px;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Initialize session state variables if they don't exist
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None
if 'annual_returns' not in st.session_state:
    st.session_state.annual_returns = None
if 'filtered_stocks' not in st.session_state:
    st.session_state.filtered_stocks = None
if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'ai_analysis' not in st.session_state:
    st.session_state.ai_analysis = None
if 'params' not in st.session_state:
    st.session_state.params = {
        "initial_investment": 10000.0,
        "monthly_contribution": 500.0,
        "time_years": 10,
        "expected_rate": 0.08,
        "max_stocks": 50
    }

# Initialize Gemini API with hardcoded key - UPDATED
def initialize_gemini():
    try:
        # Hard-coded API key - replace with your own in production
        GEMINI_API_KEY = "AIzaSyA2EP7PqgtNGbbt96OLMbTolNAplzKjK7k"
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Using the latest available model
        model = genai.GenerativeModel('gemini-1.5-pro')
        return model
    except Exception as e:
        st.error(f"Error initializing Gemini API: {e}")
        return None

# Function to get AI analysis of investment - ENHANCED
def get_investment_analysis(ticker, results, params):
    model = initialize_gemini()
    
    if not model:
        return generate_fallback_analysis(ticker, results, params)
    
    # Enhanced prompt for better analysis
    prompt = f"""
    Analyze this Dollar-Cost Averaging (DCA) investment in {ticker}:
    - Initial investment: ${params['initial_investment']}
    - Monthly contribution: ${params['monthly_contribution']}
    - Time period: {params['time_years']} years
    - Total invested: ${results['total_invested']:.2f}
    - Final value: ${results['final_value']:.2f}
    - Profit: ${results['profit']:.2f} ({results['roi_percent']:.2f}%)
    - Current price: ${results['current_price']:.2f}
    
    Provide thorough investment advice and highlight potential hidden risks.
    Format your response in two clearly labeled sections:
    1. Investment suggestion (2-3 sentences with specific insights for this stock)
    2. Hidden risks (2-3 sentences about potential concerns)
    
    Make your analysis specific to {ticker} rather than generic advice.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        # Log the error but show fallback analysis
        print(f"Error getting AI analysis: {e}")
        return generate_fallback_analysis(ticker, results, params)

# Enhanced fallback analysis function with more detailed stock-specific insights
def generate_fallback_analysis(ticker, results, params):
    roi = results['roi_percent']
    years = params['time_years']
    
    # More nuanced logic-based analysis
    if roi > 150:
        suggestion = f"**Investment suggestion:** {ticker} has demonstrated exceptional historical returns of {roi:.2f}% over {years} years, suggesting it could be a strong candidate for your DCA strategy. Consider allocating a portion of your portfolio to this stock while maintaining diversification with other assets to balance risk."
    elif roi > 100:
        suggestion = f"**Investment suggestion:** {ticker}'s historical performance shows strong returns of {roi:.2f}% over {years} years, making it potentially suitable for your DCA investment approach. The consistent growth pattern suggests this could be a good long-term holding, though market conditions may change."
    elif roi > 50:
        suggestion = f"**Investment suggestion:** {ticker} has shown moderate historical returns of {roi:.2f}% over {years} years. While not exceptional, this performance could complement a diversified portfolio. Consider allocating funds to this stock as part of a broader investment strategy rather than as a core holding."
    else:
        suggestion = f"**Investment suggestion:** {ticker}'s historical returns of {roi:.2f}% over {years} years are relatively modest. This stock might be better suited as a small position in a well-diversified portfolio. Consider evaluating the company's fundamentals and future growth prospects before committing significant capital."
    
    # More specific risk analysis
    risk_factors = [
        f"Past performance of {ticker} doesn't guarantee future results, especially given changing market conditions and economic uncertainties.",
        f"{ticker} may face industry-specific challenges, competitive pressures, or regulatory changes that could impact future performance.",
        f"Dollar-cost averaging can help mitigate short-term volatility, but won't protect against long-term underperformance if {ticker}'s fundamentals deteriorate.",
        f"Consider how {ticker} fits within your overall portfolio allocation and risk tolerance before investing."
    ]
    
    # Select 2 random risk factors for variety
    import random
    selected_risks = random.sample(risk_factors, 2)
    risks = f"**Hidden risks:** {' '.join(selected_risks)}"
    
    return f"{suggestion}\n\n{risks}"

# App title and description
st.title("Dollar-Cost Averaging (DCA) Investment Calculator")
st.write("""
This app helps you simulate dollar-cost averaging investment strategies using S&P 500 stocks.
Enter your investment parameters, and we'll show you potential returns based on historical performance.
""")

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def get_sp500_tickers():
    """Fetch current S&P 500 tickers from Wikipedia."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the table with S&P 500 companies
        table = soup.find('table', {'id': 'constituents'})
        
        if not table:
            # Alternative method if table structure changes
            tables = soup.find_all('table', {'class': 'wikitable'})
            for t in tables:
                if 'Symbol' in t.text:
                    table = t
                    break
        
        if not table:
            return get_backup_sp500_tickers()
        
        # Extract tickers from the first column
        tickers = []
        for row in table.find_all('tr')[1:]:  # Skip header row
            cells = row.find_all('td')
            if cells:
                ticker = cells[0].text.strip()
                # Remove any extra whitespace or newlines
                ticker = ticker.replace('\n', '')
                tickers.append(ticker)
        
        return tickers
    
    except Exception as e:
        st.error(f"Error retrieving S&P 500 tickers: {e}")
        return get_backup_sp500_tickers()

def get_backup_sp500_tickers():
    """Provide a backup list of major S&P 500 components if web scraping fails."""
    st.warning("Using backup list of major S&P 500 components.")
    # This is a subset of S&P 500 companies, focusing on larger components
    return [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "JPM", 
        "V", "JNJ", "WMT", "PG", "MA", "UNH", "HD", "DIS", "BAC", "VZ",
        "CSCO", "XOM", "INTC", "CVX", "ABT", "PFE", "TMO", "COST", "AVGO",
        "ACN", "DHR", "MRK", "ADBE", "LLY", "CRM", "NKE", "NEE", "T", "AMD",
        "TXN", "LIN", "PYPL", "PM", "UPS", "LOW", "MDT", "QCOM", "SBUX"
    ]

@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_stock_data(tickers, period="5y", max_stocks=50):
    """Fetch historical stock data for multiple tickers, limited to max_stocks."""
    stock_data = {}
    
    # Limit number of stocks to analyze
    if len(tickers) > max_stocks:
        tickers = tickers[:max_stocks]
    
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    for i, ticker in enumerate(tickers):
        progress_text.text(f"Fetching data for {ticker} ({i+1}/{len(tickers)})")
        progress_bar.progress((i + 1) / len(tickers))
        
        try:
            # Yahoo Finance sometimes requires a .replace() for special symbols
            safe_ticker = ticker.replace('.', '-')
            stock = yf.Ticker(safe_ticker)
            hist = stock.history(period=period)
            if not hist.empty and len(hist) > 30:  # Ensure we have meaningful data
                stock_data[ticker] = hist
        except Exception:
            pass
    
    progress_bar.empty()
    progress_text.empty()
    return stock_data

def calculate_historical_returns(stock_data):
    """Calculate historical annual returns for each stock."""
    annual_returns = {}
    
    for ticker, data in stock_data.items():
        if len(data) > 250:  # Ensure enough data (approximately 1 year of trading days)
            try:
                # Calculate annual returns based on daily close prices
                first_price = data['Close'].iloc[0]
                last_price = data['Close'].iloc[-1]
                days = len(data)
                annual_return = (last_price / first_price) ** (252 / days) - 1
                annual_returns[ticker] = annual_return * 100  # Convert to percentage
            except Exception as e:
                st.warning(f"Couldn't calculate returns for {ticker}: {e}")
    
    return annual_returns

def filter_stocks_by_rate(annual_returns, min_rate):
    """Filter stocks with returns greater than or equal to the specified rate."""
    return {ticker: rate for ticker, rate in annual_returns.items() if rate >= min_rate}

def simulate_dca(stock_data, ticker, params):
    """Simulate DCA investment for a specific stock."""
    # Extract parameters
    initial = params["initial_investment"]
    monthly = params["monthly_contribution"]
    years = params["time_years"]
    
    # Get historical data
    hist_data = stock_data[ticker].copy()
    
    # Ensure we have the 'Close' column
    if 'Close' not in hist_data.columns:
        st.error(f"Missing 'Close' data for {ticker}")
        return None
    
    # For simulation, we'll use the most recent data
    # If we don't have enough historical data, we'll use what we have and cycle through it
    monthly_prices = hist_data['Close'].resample('M').last().dropna()
    
    if monthly_prices.empty:
        st.error(f"Not enough monthly data for {ticker}")
        return None
    
    # Calculate how many months we need to simulate
    total_months = years * 12
    
    # Create a dataframe to store monthly values
    simulation_df = pd.DataFrame(index=range(total_months))
    
    # If we don't have enough historical data, we'll cycle through what we have
    if len(monthly_prices) < total_months:
        # Create a synthetic price series by cycling through the available data
        cycles = total_months // len(monthly_prices) + 1
        monthly_prices_array = np.tile(monthly_prices.values, cycles)
        simulation_df['price'] = monthly_prices_array[:total_months]
    else:
        # Use the most recent months
        simulation_df['price'] = monthly_prices.values[-total_months:]
    
    # Initialize simulation variables
    simulation_df.loc[0, 'investment'] = initial
    simulation_df.loc[0, 'shares'] = initial / simulation_df.loc[0, 'price']
    simulation_df.loc[0, 'value'] = initial
    
    # Simulate monthly investments
    for i in range(1, total_months):
        # Calculate new shares purchased this month
        if i == 0:
            simulation_df.loc[i, 'investment'] = initial
        else:
            simulation_df.loc[i, 'investment'] = simulation_df.loc[i-1, 'investment'] + monthly
        
        # Calculate cumulative shares
        if i == 0:
            simulation_df.loc[i, 'shares'] = initial / simulation_df.loc[i, 'price']
        else:
            new_shares = monthly / simulation_df.loc[i, 'price']
            simulation_df.loc[i, 'shares'] = simulation_df.loc[i-1, 'shares'] + new_shares
        
        # Calculate current portfolio value
        simulation_df.loc[i, 'value'] = simulation_df.loc[i, 'shares'] * simulation_df.loc[i, 'price']
    
    # Calculate current stock price
    current_price = hist_data['Close'].iloc[-1]
    
    # Prepare final results
    result = {
        "ticker": ticker,
        "final_value": simulation_df['value'].iloc[-1],
        "total_invested": simulation_df['investment'].iloc[-1],
        "profit": simulation_df['value'].iloc[-1] - simulation_df['investment'].iloc[-1],
        "roi_percent": (simulation_df['value'].iloc[-1] / simulation_df['investment'].iloc[-1] - 1) * 100,
        "values_over_time": simulation_df['value'].tolist(),
        "invested_over_time": simulation_df['investment'].tolist(),
        "timepoints": list(range(total_months)),
        "current_price": current_price,
        "simulation_df": simulation_df
    }
    
    return result

def plot_dca_results(results, ticker):
    """Create plotly chart for DCA simulation results."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=results["timepoints"],
        y=results["values_over_time"],
        mode='lines',
        name='Portfolio Value',
        line=dict(width=3, color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=results["timepoints"],
        y=results["invested_over_time"],
        mode='lines',
        name='Total Invested',
        line=dict(width=3, dash='dash', color='green')
    ))
    
    fig.update_layout(
        title=f"DCA Investment in {ticker} Over Time",
        xaxis_title="Months",
        yaxis_title="Value ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def run_analysis():
    """Run the entire analysis based on current parameters"""
    # Get S&P 500 tickers
    with st.spinner("Fetching S&P 500 components..."):
        sp500_tickers = get_sp500_tickers()
    
    if not sp500_tickers:
        st.error("Error: Could not retrieve S&P 500 tickers.")
        return
    
    with st.spinner(f"Analyzing stock data for {st.session_state.params['max_stocks']} S&P 500 companies..."):
        # Get stock data
        stock_data = get_stock_data(sp500_tickers, max_stocks=st.session_state.params['max_stocks'])
        st.session_state.stock_data = stock_data
        
        if not stock_data:
            st.error("Error: No valid stock data retrieved.")
            return
        
        # Calculate historical returns
        annual_returns = calculate_historical_returns(stock_data)
        st.session_state.annual_returns = annual_returns
        
        # Filter stocks based on user's expected rate
        filtered_stocks = filter_stocks_by_rate(annual_returns, st.session_state.params["expected_rate"] * 100)
        st.session_state.filtered_stocks = filtered_stocks

def main():
    # Sidebar for user inputs
    st.sidebar.header("Investment Parameters")
    
    # Input parameters with callbacks to update session state
    def update_initial_investment():
        st.session_state.params["initial_investment"] = st.session_state.initial_investment_input
        # Reset results so they are recalculated
        if 'results' in st.session_state:
            del st.session_state.results
        if 'ai_analysis' in st.session_state:
            del st.session_state.ai_analysis
    
    def update_monthly_contribution():
        st.session_state.params["monthly_contribution"] = st.session_state.monthly_contribution_input
        # Reset results so they are recalculated
        if 'results' in st.session_state:
            del st.session_state.results
        if 'ai_analysis' in st.session_state:
            del st.session_state.ai_analysis
    
    def update_time_years():
        st.session_state.params["time_years"] = st.session_state.time_years_input
        # Reset results so they are recalculated
        if 'results' in st.session_state:
            del st.session_state.results
        if 'ai_analysis' in st.session_state:
            del st.session_state.ai_analysis
    
    def update_expected_rate():
        st.session_state.params["expected_rate"] = st.session_state.expected_rate_input / 100  # Convert to decimal
        # Reset filtered stocks and results
        if 'filtered_stocks' in st.session_state:
            st.session_state.filtered_stocks = filter_stocks_by_rate(
                st.session_state.annual_returns, 
                st.session_state.params["expected_rate"] * 100
            )
        if 'results' in st.session_state:
            del st.session_state.results
        if 'ai_analysis' in st.session_state:
            del st.session_state.ai_analysis
    
    def update_max_stocks():
        st.session_state.params["max_stocks"] = st.session_state.max_stocks_input
        # Since we're changing which stocks to analyze, we should clear the analysis
        for key in ['stock_data', 'annual_returns', 'filtered_stocks', 'selected_ticker', 'results', 'ai_analysis']:
            if key in st.session_state:
                del st.session_state[key]
    
    st.sidebar.number_input(
        "Initial Investment ($)", 
        min_value=100.0,
        max_value=1000000.0,
        value=st.session_state.params["initial_investment"],
        step=100.0,
        key="initial_investment_input",
        on_change=update_initial_investment
    )
    
    st.sidebar.number_input(
        "Monthly Contribution ($)", 
        min_value=0.0,
        max_value=100000.0,
        value=st.session_state.params["monthly_contribution"],
        step=50.0,
        key="monthly_contribution_input",
        on_change=update_monthly_contribution
    )
    
    st.sidebar.slider(
        "Investment Time Horizon (Years)",
        min_value=1,
        max_value=40,
        value=st.session_state.params["time_years"],
        key="time_years_input",
        on_change=update_time_years
    )
    
    st.sidebar.slider(
        "Expected Annual Return Rate (%)",
        min_value=0.0,
        max_value=30.0,
        value=st.session_state.params["expected_rate"] * 100,  # Convert from decimal
        step=0.5,
        key="expected_rate_input",
        on_change=update_expected_rate
    )
    
    st.sidebar.slider(
        "Maximum Stocks to Analyze",
        min_value=10,
        max_value=500,
        value=st.session_state.params["max_stocks"],
        key="max_stocks_input",
        on_change=update_max_stocks
    )
    
    # Add auto-update toggle - default is True
    auto_update = st.sidebar.checkbox("Auto-update analysis", value=True)
    
    # Manual update button as fallback
    if not auto_update and st.sidebar.button("Run Analysis"):
        run_analysis()
    
    # Auto-run analysis if enabled or if we don't have any analysis results yet
    if (auto_update or st.session_state.stock_data is None) and not st.sidebar.button("Skip Analysis"):
        run_analysis()
    
    # Display info about analyzed stocks
    if st.session_state.stock_data:
        st.sidebar.info(f"Analyzing top {st.session_state.params['max_stocks']} of {len(get_sp500_tickers())} S&P 500 companies")
    
    # Display results in main area if we have data
    if st.session_state.stock_data and st.session_state.annual_returns:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Top Performing Stocks")
            
            if not st.session_state.filtered_stocks:
                st.warning(f"No stocks found with returns >= {st.session_state.params['expected_rate'] * 100:.2f}%")
                st.write("Top performing analyzed stocks:")
                sorted_returns = sorted(st.session_state.annual_returns.items(), key=lambda x: x[1], reverse=True)
                
                # Create a DataFrame for better display
                top_stocks_df = pd.DataFrame(
                    sorted_returns[:20],  # Show top 20
                    columns=['Ticker', 'Annual Return (%)']
                )
                top_stocks_df['Annual Return (%)'] = top_stocks_df['Annual Return (%)'].round(2)
                st.dataframe(top_stocks_df, use_container_width=True)
                
                # Use the top performer for simulation
                if sorted_returns:
                    selected_ticker = sorted_returns[0][0]
                else:
                    st.error("No stocks available for simulation.")
                    return
            else:
                st.write(f"Stocks with returns >= {st.session_state.params['expected_rate'] * 100:.2f}%:")
                sorted_filtered = sorted(st.session_state.filtered_stocks.items(), key=lambda x: x[1], reverse=True)
                
                # Create a DataFrame for better display
                filtered_df = pd.DataFrame(
                    sorted_filtered,
                    columns=['Ticker', 'Annual Return (%)']
                )
                filtered_df['Annual Return (%)'] = filtered_df['Annual Return (%)'].round(2)
                st.dataframe(filtered_df, use_container_width=True)
                
                # Let user select a stock from the filtered list
                def update_selected_ticker():
                    st.session_state.selected_ticker = st.session_state.ticker_select
                    # Reset results so they are recalculated
                    if 'results' in st.session_state:
                        del st.session_state.results
                    if 'ai_analysis' in st.session_state:
                        del st.session_state.ai_analysis
                
                selected_ticker = st.selectbox(
                    "Select a stock for DCA simulation:",
                    options=list(st.session_state.filtered_stocks.keys()),
                    index=0,
                    key="ticker_select",
                    on_change=update_selected_ticker
                )
                if 'selected_ticker' not in st.session_state or st.session_state.selected_ticker is None:
                    st.session_state.selected_ticker = selected_ticker
        
            # Run DCA simulation with clear error handling
            selected_ticker = st.session_state.selected_ticker if st.session_state.selected_ticker else selected_ticker
            if selected_ticker in st.session_state.stock_data:
                if 'results' not in st.session_state or st.session_state.results is None:
                    with st.spinner(f"Running DCA simulation for {selected_ticker}..."):
                        try:
                            results = simulate_dca(st.session_state.stock_data, selected_ticker, st.session_state.params)
                            st.session_state.results = results
                            
                            if results is None:
                                st.error(f"DCA simulation failed for {selected_ticker}.")
                                return
                        except Exception as e:
                            st.error(f"Error in DCA simulation: {e}")
                            import traceback
                            st.code(traceback.format_exc())
                            return
            else:
                st.error(f"No data available for {selected_ticker}")
                return
        
        # Display simulation results
        if 'results' in st.session_state and st.session_state.results:
            results = st.session_state.results
            selected_ticker = st.session_state.selected_ticker
            
            with col2:
                st.subheader(f"DCA Simulation for {selected_ticker}")
                
                # Create metrics for key results
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric("Total Invested", f"${results['total_invested']:.2f}")
                
                with metric_col2:
                    st.metric("Final Value", f"${results['final_value']:.2f}")
                
                with metric_col3:
                    st.metric(
                        "Profit", 
                        f"${results['profit']:.2f}", 
                        f"{results['roi_percent']:.2f}%"
                    )
                
                # Additional info
                st.write(f"Current {selected_ticker} price: **${results['current_price']:.2f}**")
                
                # Plot DCA simulation results
                st.plotly_chart(plot_dca_results(results, selected_ticker), use_container_width=True)
                
                # Show a table with yearly progression
                st.subheader("Year by Year Progression")
                
                # Create dataframe for yearly data
                yearly_data = []
                for year in range(1, st.session_state.params['time_years'] + 1):
                    month_idx = year * 12 - 1  # End of each year
                    if month_idx < len(results['invested_over_time']):
                        invested = results['invested_over_time'][month_idx]
                        value = results['values_over_time'][month_idx]
                        profit = value - invested
                        roi = (value / invested - 1) * 100 if invested > 0 else 0
                        
                        yearly_data.append({
                            'Year': year,
                            'Total Invested': invested,
                            'Portfolio Value': value,
                            'Profit/Loss': profit,
                            'ROI (%)': roi
                        })
                
                yearly_df = pd.DataFrame(yearly_data)
                
                # Format as currency and percentages
                for col in ['Total Invested', 'Portfolio Value', 'Profit/Loss']:
                    yearly_df[col] = yearly_df[col].map('${:,.2f}'.format)
                yearly_df['ROI (%)'] = yearly_df['ROI (%)'].map('{:,.2f}%'.format)
                
                st.dataframe(yearly_df, use_container_width=True)
                
                # Show monthly data in an expandable section
                with st.expander("Show Monthly Data"):
                    monthly_df = results['simulation_df'].copy()
                    monthly_df['Month'] = monthly_df.index + 1
                    monthly_df = monthly_df[['Month', 'price', 'shares', 'investment', 'value']]
                    monthly_df.columns = ['Month', 'Stock Price', 'Shares Owned', 'Total Invested', 'Portfolio Value']
                    
                    # Format columns
                    monthly_df['Stock Price'] = monthly_df['Stock Price'].map('${:,.2f}'.format)
                    monthly_df['Shares Owned'] = monthly_df['Shares Owned'].map('{:,.4f}'.format)
                    monthly_df['Total Invested'] = monthly_df['Total Invested'].map('${:,.2f}'.format)
                    monthly_df['Portfolio Value'] = monthly_df['Portfolio Value'].map('${:,.2f}'.format)
                    
                    st.dataframe(monthly_df, use_container_width=True)
            
            # ENHANCED: AI Analysis with styling at the bottom of the page
            st.write("---")  # Add a separator
            
            # Create a visually distinct section for AI analysis
            ai_container = st.container()
            with ai_container:
                st.subheader("🧠 AI Investment Analysis")
                
                # Get AI analysis if not already in session state
                if 'ai_analysis' not in st.session_state or st.session_state.ai_analysis is None:
                    with st.spinner("Getting AI investment analysis..."):
                        # Use the hardcoded Gemini API analysis
                        ai_analysis = get_investment_analysis(selected_ticker, results, st.session_state.params)
                        st.session_state.ai_analysis = ai_analysis
                
                # Display AI analysis in a styled box
                st.markdown(f"""
                {st.session_state.ai_analysis}
                """, unsafe_allow_html=True)
                
                # Add a disclaimer for educational purposes
                st.caption("This analysis is provided for educational purposes only and does not constitute financial advice.")
    else:
        st.info("Adjust parameters in the sidebar and analysis will run automatically.")
        
if __name__ == "__main__":
    main()
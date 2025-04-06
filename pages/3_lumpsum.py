import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai
import os
import json


def get_sp500_tickers():
    """Fetch current S&P 500 tickers from Wikipedia."""
    st.info("Fetching S&P 500 components...")
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the table with S&P 500 companies
        table = soup.find('table', {'id': 'constituents'})
        
        if not table:
            st.warning("Could not find S&P 500 table, using backup method.")
            # Alternative method if table structure changes
            tables = soup.find_all('table', {'class': 'wikitable'})
            for t in tables:
                if 'Symbol' in t.text:
                    table = t
                    break
        
        if not table:
            st.error("Could not retrieve S&P 500 components.")
            return get_backup_sp500_tickers()
        
        tickers = []
        for row in table.find_all('tr')[1:]:  # Skip header row
            cells = row.find_all('td')
            if cells:
                ticker = cells[0].text.strip()
                ticker = ticker.replace('\n', '')
                tickers.append(ticker)
        
        st.success(f"Successfully retrieved {len(tickers)} S&P 500 companies.")
        return tickers
    
    except Exception as e:
        st.error(f"Error retrieving S&P 500 tickers: {e}")
        return get_backup_sp500_tickers()

def get_backup_sp500_tickers():
    """Provide a backup list of major S&P 500 components if web scraping fails."""
    st.info("Using backup list of major S&P 500 components.")
    return [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "JPM", 
        "V", "JNJ", "WMT", "PG", "MA", "UNH", "HD", "DIS", "BAC", "VZ",
        "CSCO", "XOM", "INTC", "CVX", "ABT", "PFE", "TMO", "COST", "AVGO",
        "ACN", "DHR", "MRK", "ADBE", "LLY", "CRM", "NKE", "NEE", "T", "AMD",
        "TXN", "LIN", "PYPL", "PM", "UPS", "LOW", "MDT", "QCOM", "SBUX"
    ]

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def get_stock_data(tickers, period="5y", max_stocks=50):
    """Fetch historical stock data for multiple tickers, limited to max_stocks."""
    stock_data = {}
    
    # Limit number of stocks to analyze
    if len(tickers) > max_stocks:
        st.info(f"Limiting analysis to {max_stocks} stocks from the S&P 500")
        tickers = tickers[:max_stocks]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"Fetching data for {len(tickers)} stocks...")
    
    for i, ticker in enumerate(tickers):
        try:
            safe_ticker = ticker.replace('.', '-')
            stock = yf.Ticker(safe_ticker)
            hist = stock.history(period=period)
            if not hist.empty and len(hist) > 250:  # Ensure we have meaningful data (at least 1 year)
                stock_data[ticker] = hist
                status_text.text(f"✓ Processed {i+1}/{len(tickers)}: {ticker}")
            else:
                status_text.text(f"✗ {ticker} - Insufficient data available")
        except Exception as e:
            status_text.text(f"✗ {ticker} - Error: {e}")
        
        progress_bar.progress((i + 1) / len(tickers))
    
    progress_bar.empty()
    status_text.empty()
    st.success(f"Successfully retrieved data for {len(stock_data)} stocks.")
    
    return stock_data

def calculate_historical_returns(stock_data):
    """Calculate historical annual returns for each stock."""
    annual_returns = {}
    
    for ticker, data in stock_data.items():
        if len(data) > 250:  # Ensure enough data (approximately 1 year of trading days)
            # Calculate annual returns based on daily close prices
            annual_return = (data['Close'].iloc[-1] / data['Close'].iloc[0]) ** (252 / len(data)) - 1
            annual_returns[ticker] = annual_return * 100  # Convert to percentage
    
    return annual_returns

def filter_stocks_by_rate(annual_returns, min_rate):
    """Filter stocks with returns greater than or equal to the specified rate."""
    return {ticker: rate for ticker, rate in annual_returns.items() if rate >= min_rate}

def simulate_lump_sum(stock_data, ticker, params):
    """Simulate lump sum investment for a specific stock."""
    initial_amount = params["investment_amount"]
    years = params["time_years"]
    
    # Get historical data
    hist_data = stock_data[ticker]
    
    # Get starting price and calculate shares bought
    starting_price = hist_data['Close'].iloc[0]
    shares_bought = initial_amount / starting_price
    
    # Calculate monthly values for visualization
    monthly_prices = hist_data['Close'].resample('M').last()
    
    # Calculate how many months we need to simulate
    total_months = years * 12
    
    # If we don't have enough historical data, we'll cycle through what we have
    if len(monthly_prices) < total_months:
        # Create a synthetic price series by cycling through the available data
        cycles = total_months // len(monthly_prices) + 1
        extended_prices = pd.Series(np.tile(monthly_prices.values, cycles)[:total_months])
    else:
        # Use the most recent months
        extended_prices = monthly_prices[-total_months:].reset_index(drop=True)
    
    # Calculate value over time
    value_over_time = [shares_bought * price for price in extended_prices]
    
    # Current price and total value
    current_price = hist_data['Close'].iloc[-1]
    final_value = shares_bought * extended_prices.iloc[-1]
    
    return {
        "initial_investment": initial_amount,
        "shares_bought": shares_bought,
        "final_value": final_value,
        "profit": final_value - initial_amount,
        "roi_percent": ((final_value / initial_amount) - 1) * 100,
        "values_over_time": value_over_time,
        "timepoints": list(range(total_months)),
        "current_price": current_price
    }

def plot_top_performers(stock_data, annual_returns, params, top_n=5):
    """Plot lump sum investment results for top performing stocks using Plotly."""
    # Get top performers
    sorted_returns = sorted(annual_returns.items(), key=lambda x: x[1], reverse=True)
    top_performers = sorted_returns[:top_n]
    
    # Create plotly figure with subplots
    fig = make_subplots(rows=top_n, cols=1, subplot_titles=[f"{ticker}: {rate:.2f}% Annual Return" for ticker, rate in top_performers])
    
    results_data = []
    
    for i, (ticker, rate) in enumerate(top_performers):
        # Simulate lump sum
        results = simulate_lump_sum(stock_data, ticker, params)
        results_data.append({"ticker": ticker, "results": results, "rate": rate})
        
        # Add investment value line
        fig.add_trace(
            go.Scatter(
                x=results["timepoints"], 
                y=results["values_over_time"],
                mode='lines',
                name=f"{ticker} Value",
                line=dict(width=2),
                showlegend=False
            ),
            row=i+1, col=1
        )
        
        # Add initial investment line
        fig.add_trace(
            go.Scatter(
                x=[min(results["timepoints"]), max(results["timepoints"])],
                y=[params["investment_amount"], params["investment_amount"]],
                mode='lines',
                name="Initial Investment",
                line=dict(color='red', dash='dash', width=1.5),
                showlegend=False
            ),
            row=i+1, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=180 * top_n,
        title_text=f"Lump Sum Investment of ${params['investment_amount']:,.2f} Over {params['time_years']} Years",
        showlegend=False
    )
    
    # Update y-axes labels
    for i in range(1, top_n+1):
        fig.update_yaxes(title_text="Value ($)", row=i, col=1)
    
    # Update the bottom x-axis label
    fig.update_xaxes(title_text="Months", row=top_n, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    return results_data

def initialize_gemini():
    """Initialize the Gemini LLM API."""
    # Hard-coding API key (not recommended for production, use environment variables)
    API_KEY = "AIzaSyA2EP7PqgtNGbbt96OLMbTolNAplzKjK7k"  # Replace with actual API key in production
    
    # Configure the Gemini API
    genai.configure(api_key=API_KEY)
    
    # Set up the model
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 2048,
    }
    
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    
    model = genai.GenerativeModel(
        model_name="gemini-pro",
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    
    return model

def get_investment_insights(results_data, params):
    """Generate investment insights and risk assessment using Gemini LLM."""
    try:
        model = initialize_gemini()
        
        # Prepare prompt with analysis data
        top_stocks = [f"{data['ticker']} (Annual Return: {data['rate']:.2f}%, Final Value: ${data['results']['final_value']:.2f}, ROI: {data['results']['roi_percent']:.2f}%)" 
                      for data in results_data]
        
        top_stocks_str = "\n".join([f"- {item}" for item in top_stocks])
        
        prompt = f"""
        As a financial advisor, provide investment insights and risk assessment based on the following data:
        
        Investment Parameters:
        - Initial Investment: ${params['investment_amount']:,.2f}
        - Time Horizon: {params['time_years']} years
        - Expected Annual Return: {params['expected_rate']*100:.2f}%
        
        Top Performing Stocks:
        {top_stocks_str}
        
        Please provide:
        1. A concise summary of the investment performance analysis
        2. Investment suggestions based on this data
        3. Three critical hidden risks that investors should consider
        4. A balanced perspective on the limitations of this historical analysis
        
        Format your response in markdown with clear sections.
        """
        
        # Generate response
        response = model.generate_content(prompt)
        
        # Return the formatted response
        return response.text
    
    except Exception as e:
        return f"""
        ## Investment Insights
        
        Sorry, we couldn't generate personalized insights at this time. Here are some general considerations:
        
        - Past performance is not indicative of future results
        - Consider diversifying your investments
        - Consult with a financial advisor before making investment decisions
        
        Error details: {str(e)}
        """

def main():
    # Set page config
    st.set_page_config(
        page_title="Lump Sum Investment Analyzer",
        page_icon="💰",
        layout="wide"
    )

    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #FFDBED, #ffffff);
            padding: 40px;  /* Increased padding */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # App title and description
    st.title("💰 Lump Sum Investment Analyzer")
    st.markdown("""
    This application analyzes how a lump sum investment would perform across S&P 500 stocks.
    Enter your investment parameters below to get started.
    """)
    
    # Sidebar for user inputs
    st.sidebar.header("Investment Parameters")
    
    investment_amount = st.sidebar.number_input(
        "Investment Amount ($)",
        min_value=1000.0,
        max_value=1000000.0,
        value=10000.0,
        step=1000.0
    )
    
    time_years = st.sidebar.slider(
        "Investment Time Horizon (Years)",
        min_value=1,
        max_value=30,
        value=10
    )
    
    expected_rate = st.sidebar.slider(
        "Expected Annual Return Rate (%)",
        min_value=1.0,
        max_value=25.0,
        value=7.0,
        step=0.5
    )
    
    max_stocks = st.sidebar.slider(
        "Max Stocks to Analyze",
        min_value=10,
        max_value=100,
        value=50
    )
    
    # Create params dictionary
    params = {
        "investment_amount": investment_amount,
        "time_years": time_years,
        "expected_rate": expected_rate / 100,  # Convert to decimal
    }
    
    # Analysis section
    st.header("Analysis")
    
    # Run button
    if st.button("Run Analysis"):
        with st.spinner("Fetching S&P 500 tickers..."):
            # Get S&P 500 tickers
            sp500_tickers = get_sp500_tickers()
            
        if not sp500_tickers:
            st.error("Could not retrieve S&P 500 tickers. Please try again later.")
            return
        
        with st.spinner(f"Fetching stock data (this may take a few minutes)..."):
            # Get stock data
            stock_data = get_stock_data(sp500_tickers, max_stocks=max_stocks)
            
        if not stock_data:
            st.error("No valid stock data retrieved. Please try again later.")
            return
        
        # Calculate historical returns
        with st.spinner("Calculating returns..."):
            annual_returns = calculate_historical_returns(stock_data)
            
            # Filter stocks based on user's expected rate
            filtered_stocks = filter_stocks_by_rate(annual_returns, params["expected_rate"] * 100)
            
            # Create tabs for results
            tab1, tab2, tab3, tab4 = st.tabs(["Top Performers", "Detailed Results", "Expected Rate Comparison", "AI Insights"])
            
            with tab1:
                st.subheader("Top Performing Stocks")
                
                if not filtered_stocks:
                    st.warning(f"No stocks found with returns >= {params['expected_rate'] * 100:.2f}%")
                else:
                    st.success(f"Found {len(filtered_stocks)} S&P 500 stocks with returns >= {params['expected_rate'] * 100:.2f}%")
                
                # Plot top 5 performers and get results
                results_data = plot_top_performers(stock_data, annual_returns, params)
            
            with tab2:
                st.subheader("Detailed Investment Results")
                
                # Convert results to DataFrame for display
                results_df = pd.DataFrame([
                    {
                        "Ticker": data["ticker"],
                        "Annual Return (%)": round(data["rate"], 2),
                        "Initial Investment ($)": params["investment_amount"],
                        "Shares Bought": round(data["results"]["shares_bought"], 2),
                        "Final Value ($)": round(data["results"]["final_value"], 2),
                        "Profit ($)": round(data["results"]["profit"], 2),
                        "ROI (%)": round(data["results"]["roi_percent"], 2)
                    } for data in results_data
                ])
                
                st.dataframe(results_df, use_container_width=True)
                
                # Download button for results
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="investment_results.csv",
                    mime="text/csv",
                )
            
            with tab3:
                st.subheader("Comparison with Expected Rate")
                
                # Calculate ideal investment
                ideal_final = params["investment_amount"] * (1 + params["expected_rate"])**params["time_years"]
                ideal_profit = ideal_final - params["investment_amount"]
                ideal_roi = ((ideal_final / params["investment_amount"]) - 1) * 100
                
                # Create two columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Expected Annual Return", f"{params['expected_rate']*100:.2f}%")
                    st.metric("Initial Investment", f"${params['investment_amount']:,.2f}")
                    st.metric("Ideal Final Value", f"${ideal_final:,.2f}")
                
                with col2:
                    st.metric("Ideal Profit", f"${ideal_profit:,.2f}")
                    st.metric("Ideal ROI", f"{ideal_roi:.2f}%")
                    st.metric("Investment Period", f"{params['time_years']} years")
                
                # Create a simple chart to visualize growth
                years = list(range(params['time_years'] + 1))
                values = [params["investment_amount"] * (1 + params["expected_rate"])**year for year in years]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=years, 
                    y=values, 
                    mode='lines+markers',
                    name='Expected Growth',
                    line=dict(color='green', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=[0, params['time_years']],
                    y=[params["investment_amount"], params["investment_amount"]],
                    mode='lines',
                    name='Initial Investment',
                    line=dict(color='red', dash='dash')
                ))
                
                fig.update_layout(
                    title=f"Expected Growth at {params['expected_rate']*100:.2f}% Annual Return",
                    xaxis_title="Years",
                    yaxis_title="Value ($)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                st.subheader("AI Investment Insights")
                
                with st.spinner("Generating investment insights..."):
                    # Get AI-generated insights
                    insights = get_investment_insights(results_data, params)
                    
                    # Display insights
                    st.markdown(insights)
                    
                    # Add disclaimer
                    st.warning("**Disclaimer:** These AI-generated insights are for informational purposes only and should not be considered as financial advice. Always consult with a qualified financial advisor before making investment decisions.")
    else:
        # Show instructions when app loads
        st.info("""
        Click "Run Analysis" to start the analysis. The application will:
        1. Fetch the current S&P 500 ticker symbols
        2. Download historical data for up to 50 stocks
        3. Calculate historical returns and simulate your investment
        4. Show you the top performers and detailed results
        5. Provide AI-powered investment insights and risk assessment
        """)
    
    # Footer
    st.markdown("---")
    
    # AI-powered risk considerations section
    st.header("🔮 Investment Prediction Considerations")
    
    # Create two columns for pros and cons
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Potential Benefits")
        st.markdown("""
        - **Historical Performance:** Analysis based on actual market data from top performing stocks
        - **Compound Returns:** Visualize the power of compounding over your selected time horizon
        - **Customizable Parameters:** Adjust investment amount, time horizon, and expected returns
        - **Diversification Insights:** Compare performance across different S&P 500 companies
        - **AI-Enhanced Suggestions:** Receive context-aware investment considerations
        """)
    
    with col2:
        st.subheader("Hidden Risks")
        st.markdown("""
        - **Past ≠ Future:** Historical performance is not indicative of future results
        - **Market Volatility:** Stock markets can experience significant drops and extended bear markets
        - **Cyclical Patterns:** Economic cycles may affect returns during your investment period
        - **Company-Specific Risk:** Individual stocks can underperform despite index strength
        - **Inflation Impact:** Real returns may be lower than nominal returns shown
        - **Tax Considerations:** Analysis does not account for capital gains taxes or other fees
        """)
    
    # Final disclaimer
    st.markdown("---")
    st.markdown("""
    **Important Disclaimer:** This application provides investment simulations based on historical data and 
    should not be considered financial advice. The AI-generated insights are for educational purposes only. 
    Past performance doesn't guarantee future results. Always consult with a qualified financial advisor 
    before making investment decisions.
    """)

if __name__ == "__main__":
    main()
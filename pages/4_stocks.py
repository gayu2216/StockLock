import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import google.generativeai as genai


# Set page configuration
st.set_page_config(
   page_title="Stock Price Predictor",
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


# Initialize Gemini API
genai.configure(api_key="AIzaSyA2EP7PqgtNGbbt96OLMbTolNAplzKjK7k")  # Replace with actual API key in production

# Create Gemini model instance
model = genai.GenerativeModel('gemini-1.5-pro')


# Title and description
st.title("Advanced Stock Price Prediction App")
st.markdown("""
This app predicts future stock prices using LSTM neural networks and Gemini AI, along with analysis of top S&P 500 companies.
""")


# Define tabs
tab1, tab2, tab3 = st.tabs(["Stock Prediction", "S&P 500 Analysis", "AI Insights"])


with tab1:
   # Sidebar for inputs
   st.sidebar.header("User Input Parameters")


   # Stock selection
   stock_symbol = st.sidebar.text_input("Stock Symbol", "AAPL")


   # Timeframe selection
   prediction_days = st.sidebar.slider("Days for prediction pattern", 30, 120, 60)
   future_days = st.sidebar.slider("Days to predict into future", 7, 90, 30)


   # Training period selection
   training_years = st.sidebar.slider("Years of training data", 1, 10, 5)


   def predict_future_prices(model, scaler, data, stock_symbol, prediction_days=60, future_days=30):
       try:
           # Debug information
           st.write(f"Starting prediction for {stock_symbol} with {prediction_days} prediction days and {future_days} future days")
          
           # Validate inputs
           if model is None:
               st.error("Model is not initialized")
               return None
              
           if scaler is None:
               st.error("Scaler is not initialized")
               return None
              
           if data is None or data.empty or 'Close' not in data.columns:
               st.error(f"Invalid training data provided: {type(data)} {'empty' if data is not None and data.empty else ''}")
               return None
          
           # Download recent data for validation
           st.info(f"Downloading recent data for {stock_symbol}...")
           test_data = yf.download(stock_symbol, period="1y")
          
           if test_data.empty:
               st.error(f"Could not download data for {stock_symbol}. Please check the symbol.")
               return None
              
           st.success(f"Downloaded {len(test_data)} days of recent data")
          
           # Get the actual close prices
           actual_prices = test_data['Close'].values
          
           # Display data shapes for debugging
           st.write(f"Training data shape: {data.shape}")
           st.write(f"Test data shape: {test_data.shape}")
          
           # --- PREPARE DATA ---
           # Combine datasets carefully
           st.info("Preparing data for prediction...")
          
           # Make sure indices don't overlap
           combined_data = pd.concat([data['Close'], test_data['Close']], axis=0)
           combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
          
           # Verify we have enough data
           total_required_days = prediction_days + future_days
           if len(combined_data) < total_required_days:
               st.error(f"Not enough data. Need at least {total_required_days} days, but only have {len(combined_data)} days.")
               return None
              
           # Get the required portion of the dataset
           model_inputs = combined_data[-len(test_data) - prediction_days:].values
           model_inputs = model_inputs.reshape(-1, 1)
           model_inputs = scaler.transform(model_inputs)
          
           # --- VALIDATION PREDICTION ---
           st.info("Creating validation prediction...")
          
           # Create the test dataset
           x_test = []
          
           for i in range(prediction_days, len(model_inputs)):
               x_test.append(model_inputs[i - prediction_days:i, 0])
              
           x_test = np.array(x_test)
          
           # Reshape for LSTM [samples, time steps, features]
           x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
          
           # Predict
           st.info(f"Making validation predictions. Input shape: {x_test.shape}")
           try:
               predicted_prices = model.predict(x_test)
           except Exception as e:
               st.error(f"Error during model prediction: {str(e)}")
               return None
              
           # Convert predictions back to price scale
           predicted_prices = scaler.inverse_transform(predicted_prices)
          
           # Create correct timeframe for validation prediction
           validation_dates = test_data.index[-len(predicted_prices):]
          
           # --- DISPLAY VALIDATION CHART ---
           # Create figure
           fig1, ax1 = plt.subplots(figsize=(12, 6))
          
           # Plot actual vs predicted
           ax1.plot(validation_dates,
                   actual_prices[-len(predicted_prices):],
                   color='black', linewidth=2, label=f"Actual {stock_symbol} Price")
           ax1.plot(validation_dates,
                   predicted_prices,
                   color='blue', linewidth=2, label=f"Predicted {stock_symbol} Price")
          
           ax1.set_title(f"{stock_symbol} Share Price - Model Validation", fontsize=16)
           ax1.set_xlabel("Date", fontsize=12)
           ax1.set_ylabel(f"Price (USD)", fontsize=12)
           ax1.grid(True, alpha=0.3)
           ax1.legend(fontsize=12)
           fig1.tight_layout()
          
           # Format dates on x-axis
           ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
           plt.xticks(rotation=45)
          
           # Display validation figure
           st.pyplot(fig1)
          
           # --- FUTURE PREDICTION ---
           st.info("Making future predictions...")
          
           # Get the last sequence of known data points
           last_sequence = model_inputs[-prediction_days:]
           current_batch = last_sequence.reshape((1, prediction_days, 1))
          
           # Current price (last known price)
           current_price = float(actual_prices[-1])
           st.write(f"Current price: ${current_price:.2f}")
          
           # List to hold predictions
           future_predictions = []
          
           # Make day-by-day predictions
           for i in range(future_days):
               # Make a prediction
               future_price = model.predict(current_batch)
              
               # Store the prediction
               future_predictions.append(future_price[0, 0])
              
               # Update the sequence - FIX: Reshape the new prediction to match dimensions
               # Create a properly shaped new value with 3 dimensions to match current_batch
               new_value = np.array([[[future_price[0, 0]]]])
              
               # Remove first timestep and add new prediction at the end
               current_batch = np.concatenate((current_batch[:, 1:, :], new_value), axis=1)
          
           # Convert predictions to numpy array and reshape
           future_pred_array = np.array(future_predictions).reshape(-1, 1)
          
           # Convert back to original price scale
           future_pred_prices = scaler.inverse_transform(future_pred_array).flatten()
          
           # Create future dates
           last_date = test_data.index[-1]
           future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)
          
           # Create DataFrame for future predictions
           future_df = pd.DataFrame({
               'Date': future_dates,
               'Predicted_Price': future_pred_prices
           })
           future_df.set_index('Date', inplace=True)
          
           # Calculate changes
           final_price = float(future_df['Predicted_Price'].iloc[-1])
           price_change = final_price - current_price
           percent_change = (price_change / current_price) * 100
          
           st.write(f"Final predicted price: ${final_price:.2f}")
           st.write(f"Predicted change: ${price_change:.2f} ({percent_change:.2f}%)")
          
           # --- DISPLAY FORECAST CHART ---
           # Create figure
           fig2, ax2 = plt.subplots(figsize=(12, 6))
          
           # Plot recent actual data (last 30 days or less)
           days_to_show = min(30, len(actual_prices))
           recent_dates = test_data.index[-days_to_show:]
          
           # Plot actual data with dates
           ax2.plot(recent_dates,
                   actual_prices[-days_to_show:],
                   color='black', linewidth=2, label="Recent Actual Price")
          
           # Plot future predictions with dates
           combined_dates = pd.DatetimeIndex(list(recent_dates[-1:]) + list(future_dates))
           combined_prices = np.append(actual_prices[-1:], future_pred_prices)
          
           ax2.plot(combined_dates,
                   combined_prices,
                   color='red', linewidth=2, label="Future Predicted Price")
          
           # Add markers
           ax2.scatter(recent_dates[-1], current_price, color='green', s=100,
                     label=f"Current: ${current_price:.2f}")
           ax2.scatter(future_dates[-1], final_price, color='purple', s=100,
                     label=f"Final: ${final_price:.2f}")
          
           # Add annotation
           change_color = 'green' if price_change > 0 else 'red'
           change_text = f"Predicted Change: ${price_change:.2f} ({percent_change:.2f}%)"
           plt.figtext(0.5, 0.01, change_text, ha='center', color=change_color,
                      fontsize=14, bbox=dict(facecolor='white', alpha=0.8))
          
           ax2.set_title(f"{stock_symbol} {future_days}-Day Price Forecast", fontsize=16)
           ax2.set_xlabel("Date", fontsize=12)
           ax2.set_ylabel("Price (USD)", fontsize=12)
           ax2.grid(True, alpha=0.3)
           ax2.legend(fontsize=12, loc='best')
          
           # Format dates on x-axis
           ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
           plt.xticks(rotation=45)
          
           fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
          
           # Display forecast figure
           st.pyplot(fig2)
          
           # --- CREATE FORECAST TABLE ---
           # Create table
           future_table = pd.DataFrame({
               'Day': range(1, future_days + 1),
               'Date': future_dates,
               'Predicted Price': future_pred_prices,
               'Change from Today': future_pred_prices - current_price,
               'Percent Change': ((future_pred_prices - current_price) / current_price) * 100
           })
          
           # Format for display
           display_table = pd.DataFrame({
               'Day': future_table['Day'],
               'Date': future_table['Date'].dt.strftime('%Y-%m-%d'),
               'Predicted Price': [f"${price:.2f}" for price in future_table['Predicted Price']],
               'Change': [f"${change:+.2f}" for change in future_table['Change from Today']],
               'Percent Change': [f"{pct:+.2f}%" for pct in future_table['Percent Change']]
           })
          
           # Display table
           st.dataframe(display_table, use_container_width=True)
          
           # Get Gemini prediction insights
           try:
               st.session_state['gemini_prediction_data'] = {
                   'symbol': stock_symbol,
                   'current_price': current_price,
                   'final_price': final_price,
                   'percent_change': percent_change,
                   'future_days': future_days,
                   'historical_data': data['Close'].tail(30).tolist()
               }
           except Exception as e:
               st.warning(f"Could not prepare Gemini data: {str(e)}")
          
           # Success message
           st.success(f"Successfully predicted {future_days} days ahead for {stock_symbol}")
          
           return future_df
          
       except Exception as e:
           st.error(f"Error in prediction function: {str(e)}")
           import traceback
           st.code(traceback.format_exc())
           return None


   # Function to create and train LSTM model
   def train_model(data, prediction_days):
       # Prepare the data
       scaler = MinMaxScaler(feature_range=(0, 1))
       scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
      
       # Create training dataset
       x_train = []
       y_train = []
      
       for i in range(prediction_days, len(scaled_data)):
           x_train.append(scaled_data[i - prediction_days:i, 0])
           y_train.append(scaled_data[i, 0])
          
       x_train, y_train = np.array(x_train), np.array(y_train)
       x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
      
       # Build the model
       model = Sequential()
      
       # Adding LSTM layers
       model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
       model.add(Dropout(0.2))
       model.add(LSTM(units=50, return_sequences=False))
       model.add(Dropout(0.2))
       model.add(Dense(units=1))
      
       # Compile the model
       model.compile(optimizer='adam', loss='mean_squared_error')
      
       # Status update
       st.info("Model compiled, beginning training...")
      
       # Train the model
       with st.spinner("Training model..."):
           model.fit(x_train, y_train, epochs=25, batch_size=32, verbose=0)
      
       st.success("Model training complete!")
      
       return model, scaler


   # Stock prediction section
   st.header("Stock Price Prediction")


   # Show a spinner while downloading data
   if st.button("Download Historical Data"):
       with st.spinner(f"Downloading historical data for {stock_symbol}..."):
           # Download data
           start_date = pd.Timestamp.now() - pd.DateOffset(years=training_years)
           training_data = yf.download(stock_symbol, start=start_date)
          
       if training_data.empty:
           st.error(f"Could not download data for {stock_symbol}. Please check the symbol.")
       else:
           st.success(f"Downloaded {len(training_data)} days of historical data")
          
           # Show historical data
           st.subheader("Historical Data")
           st.dataframe(training_data.tail(), use_container_width=True)
          
           # Plot historical price
           fig_hist = plt.figure(figsize=(12, 6))
           plt.plot(training_data['Close'])
           plt.title(f"{stock_symbol} Historical Close Price")
           plt.xlabel("Date")
           plt.ylabel("Price (USD)")
           plt.grid(True, alpha=0.3)
           st.pyplot(fig_hist)
          
           # Store data in session state
           st.session_state['training_data'] = training_data
           st.session_state['data_loaded'] = True


   # Button to start prediction
   if st.button("Train Model and Predict"):
       if 'data_loaded' not in st.session_state or not st.session_state['data_loaded']:
           st.warning("Please download historical data first")
       else:
           # Train model
           st.subheader("Model Training")
           model, scaler = train_model(st.session_state['training_data'], prediction_days)
          
           # Make predictions
           st.subheader("Price Predictions")
           prediction_results = predict_future_prices(
               model,
               scaler,
               st.session_state['training_data'],
               stock_symbol,
               prediction_days,
               future_days
           )
          
           if prediction_results is not None:
               st.balloons()


with tab2:
   st.header("S&P 500 Top Companies Analysis")
  
   # Function to get top 5 S&P 500 companies by market cap
   @st.cache_data(ttl=3600)  # Cache for 1 hour
   def get_top_sp500_companies():
       try:
           st.info("Fetching S&P 500 companies data...")
          
           # Download S&P 500 tickers list
           sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
           symbols = sp500['Symbol'].tolist()
          
           # Get top companies by market cap
           market_caps = {}
           company_names = {}
          
           # Limit to first 20 for faster processing
           sample_symbols = symbols[:30]
          
           for symbol in sample_symbols:
               try:
                   ticker = yf.Ticker(symbol)
                   info = ticker.info
                  
                   if 'marketCap' in info and info['marketCap'] is not None:
                       market_caps[symbol] = info['marketCap']
                       company_names[symbol] = info.get('longName', symbol)
               except Exception as e:
                   continue
          
           # Get top 5 by market cap
           top_symbols = sorted(market_caps.keys(), key=lambda x: market_caps[x], reverse=True)[:5]
          
           # Create result dataframe
           top_companies = pd.DataFrame({
               'Symbol': top_symbols,
               'Company': [company_names[symbol] for symbol in top_symbols],
               'Market Cap (Billions)': [market_caps[symbol] / 1e9 for symbol in top_symbols]
           })
          
           return top_companies, top_symbols
      
       except Exception as e:
           st.error(f"Error fetching S&P 500 data: {str(e)}")
           return pd.DataFrame(), []
  
   # Get top companies
   with st.spinner("Analyzing top S&P 500 companies..."):
       top_companies, top_symbols = get_top_sp500_companies()
  
   if not top_companies.empty:
       # Display top companies
       st.subheader("Top 5 S&P 500 Companies by Market Cap")
       st.dataframe(top_companies, use_container_width=True)
      
       # Display market cap chart
       fig_market_cap = px.bar(
           top_companies,
           x='Symbol',
           y='Market Cap (Billions)',
           title="Market Capitalization (Billions USD)",
           text_auto='.2f',
           color='Market Cap (Billions)',
           height=500
       )
       st.plotly_chart(fig_market_cap, use_container_width=True)
      
       # Performance comparison
       st.subheader("Performance Comparison")
      
       # Date range selection
       compare_period = st.radio(
           "Select comparison timeframe:",
           ["1 Month", "3 Months", "6 Months", "1 Year", "5 Years"],
           horizontal=True
       )
      
       period_map = {
           "1 Month": "1mo",
           "3 Months": "3mo",
           "6 Months": "6mo",
           "1 Year": "1y",
           "5 Years": "5y"
       }
      
       # Get performance data
       with st.spinner(f"Fetching {compare_period} performance data..."):
           performance_data = pd.DataFrame()
          
           for symbol in top_symbols:
               try:
                   stock_data = yf.download(symbol, period=period_map[compare_period])
                   if not stock_data.empty:
                       # Normalize to first day = 100%
                       normalized = stock_data['Close'] / stock_data['Close'].iloc[0] * 100
                       performance_data[symbol] = normalized
               except Exception as e:
                   st.warning(f"Could not get data for {symbol}: {str(e)}")
          
       if not performance_data.empty:
           # Plot performance comparison
           fig_performance = px.line(
               performance_data,
               title=f"Stock Performance ({compare_period})",
               labels={"value": "Price (Normalized %)", "variable": "Company"},
               height=600
           )
          
           # Add reference line at 100%
           fig_performance.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.7)
          
           # Improve layout
           fig_performance.update_layout(
               legend_title_text='',
               xaxis_title="Date",
               yaxis_title="Performance (%)",
               hovermode="x unified"
           )
          
           st.plotly_chart(fig_performance, use_container_width=True)
          
           # Financial metrics comparison
           st.subheader("Financial Metrics Comparison")
          
           # Get key metrics
           metrics_data = []
          
           for symbol in top_symbols:
               try:
                   ticker = yf.Ticker(symbol)
                   info = ticker.info
                  
                   metrics_data.append({
                       'Symbol': symbol,
                       'Company': info.get('longName', symbol),
                       'P/E Ratio': info.get('trailingPE', None),
                       'Dividend Yield (%)': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else None,
                       'EPS': info.get('trailingEps', None),
                       '52 Week High': info.get('fiftyTwoWeekHigh', None),
                       '52 Week Low': info.get('fiftyTwoWeekLow', None),
                       'Analyst Rating': info.get('recommendationKey', 'N/A')
                   })
               except Exception as e:
                   continue
          
           metrics_df = pd.DataFrame(metrics_data)
          
           # Display metrics table
           if not metrics_df.empty:
               st.dataframe(metrics_df, use_container_width=True)
              
               # Financial metrics visualization
               col1, col2 = st.columns(2)
              
               with col1:
                   # P/E Ratio Chart
                   if not metrics_df['P/E Ratio'].isna().all():
                       pe_fig = px.bar(
                           metrics_df,
                           x='Symbol',
                           y='P/E Ratio',
                           title="P/E Ratio Comparison",
                           color='Symbol',
                           text_auto='.2f'
                       )
                       st.plotly_chart(pe_fig, use_container_width=True)
              
               with col2:
                   # Dividend Yield Chart
                   if not metrics_df['Dividend Yield (%)'].isna().all():
                       div_fig = px.bar(
                           metrics_df,
                           x='Symbol',
                           y='Dividend Yield (%)',
                           title="Dividend Yield Comparison (%)",
                           color='Symbol',
                           text_auto='.2f'
                       )
                       st.plotly_chart(div_fig, use_container_width=True)
   else:
       st.error("Unable to retrieve S&P 500 companies data. Please try again later.")


with tab3:
    st.header("AI Insights & Risk Analysis")
    
    # Function to get Gemini AI insights
    def get_gemini_insights(stock_symbol, data=None):
        try:
            # Basic stock info
            ticker_info = yf.Ticker(stock_symbol).info
            company_name = ticker_info.get('longName', stock_symbol)
            sector = ticker_info.get('sector', 'Unknown')
            industry = ticker_info.get('industry', 'Unknown')
            
            # Construct the prompt
            prompt = f"""
            As a financial analyst, provide insights on {company_name} ({stock_symbol}) in the {industry} industry, {sector} sector.
            
            1. Provide a brief investment summary and outlook (3-4 sentences)
            2. List 3 key potential growth catalysts
            3. List 3 key risks investors should be aware of
            4. Rate the stock's risk level (Low, Medium, High) and explain why
            
            Format your response in clear sections with headers.
            """
            
            # If we have prediction data, add it to the prompt
            if 'gemini_prediction_data' in st.session_state:
                pred_data = st.session_state['gemini_prediction_data']
                if pred_data['symbol'] == stock_symbol:
                    prompt += f"""
                    
                    Additional information: Our LSTM model predicts the price will change from ${pred_data['current_price']:.2f} to ${pred_data['final_price']:.2f} 
                    in the next {pred_data['future_days']} days (a {pred_data['percent_change']:.2f}% change).
                    Consider this prediction in your analysis.
                    """
            
            # Get the response from Gemini
            response = model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"Error generating AI insights: {str(e)}"
    
    # Button to get AI analysis
    stock_to_analyze = st.text_input("Enter stock symbol for AI analysis:", value="AAPL")
    
    if st.button("Generate AI Analysis"):
        with st.spinner("Generating AI insights..."):
            insights = get_gemini_insights(stock_to_analyze)
            
        st.markdown("### Gemini AI Analysis")
        st.markdown(insights)
        
        # Display investment recommendations
        st.markdown("### Investment Considerations")
        
        # Risk disclosure
        with st.expander("⚠️ Important Risk Disclosures", expanded=True):
            st.warning("""
            The predictions and analyses provided by this application are for informational purposes only and should not be considered as financial advice. 
            Stock markets are inherently volatile and unpredictable. All investments involve risk, including the possible loss of principal.
            
            - Machine learning models like LSTM have limitations and cannot account for all market factors
            - AI-generated insights are based on available data and may not consider recent events
            - Past performance is not indicative of future results
            - Always consult with a qualified financial advisor before making investment decisions
            """)
        
        # VaR calculation
        st.subheader("Value at Risk (VaR) Analysis")
        try:
            # Get historical data
            hist_data = yf.download(stock_to_analyze, period="1y")
            
            if not hist_data.empty:
                # Calculate daily returns
                returns = hist_data['Close'].pct_change().dropna()
                
                # Calculate VaR
                confidence_level = 0.95
                var_95 = np.percentile(returns, 5)
                
                # Display VaR
                st.write(f"95% Value at Risk (1-day): {var_95:.2%}")
                st.write(f"This means there is a 5% chance of losing more than {abs(var_95):.2%} in a single day based on historical data.")
                
                # Plot returns distribution
                fig_var = px.histogram(returns, nbins=50, title="Daily Returns Distribution")
                fig_var.add_vline(x=var_95, line_dash="dash", line_color="red")
                fig_var.add_annotation(x=var_95, y=0, text="95% VaR", showarrow=True, arrowhead=1)
                st.plotly_chart(fig_var, use_container_width=True)
        except Exception as e:
            st.error(f"Error calculating VaR: {str(e)}")
    
    # Disclaimer at bottom of page
    st.markdown("---")
    st.markdown("""
    **Disclaimer:** This application combines traditional LSTM neural network models with Google's Gemini AI to provide comprehensive stock analysis. 
    The predictions represent a synthesis of multiple analytical approaches but should be used alongside fundamental research and professional advice. 
    Remember that all investments carry risk, and technology-based predictions have inherent limitations in capturing market complexity and black swan events.
    """)
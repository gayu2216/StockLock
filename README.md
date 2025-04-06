# StockLock
The Sherlock of your finance. Get personalized AI powered finance advice.

## Inspiration
Stock Lock bridges the gap between people and traditional stock market analysis with AI-powered, real-time investment insights. By leveraging real-time market data from Yahoo Finance and Gemini, along with cutting-edge AI models like RAG (Retrieval-Augmented Generation) and LSTM (Long Short-Term Memory), we aimed to create a system that doesn't just predict stock prices but helps users understand market trends and make data-driven investment decisions.

## What it does
Our Stock Lock is designed to provide real-time stock market insights and predictions. It pulls live data from trusted sources like the Yahoo Finance API and Gemini API, offering up-to-date information on stock prices and market trends. By using a combination of advanced AI models, including a RAG model and LSTM networks, the application analyzes historical stock data and performs time series analysis to predict future stock prices based on user input.
Users can interact with the tool by entering a stock ticker, and the system will provide predictions on future stock prices. This allows users to make more informed decisions about their investments. Additionally, our model can generate insights about the overall market and specific stocks, creating a more personalized and data-driven approach to investing.

## How we built it
The core backend of our application is based on an LSTM model through TensorFlow. The data fed into the model is collected through an API call from Yahoo Finance to get real-time data for increased accuracy. We used other Python libraries such as LangChain, Matplotlib, Numpy, and Pandas in order to develop the RAG model and preprocess the data for the neural network.
The frontend was developed using Streamlit.

## Challenges we ran into
The training of our LSTM model was challenging due to the complex structure of the neural network. Retrieving and training the data in the context of finance was also a challenge. It was our first time working with a RAG model, and learning vector database concepts was challenging.

## Accomplishments that we're proud of
Using Yahoo Finance API calls for retrieving the data and enabling us to provide insights with accurate and real-time dynamic data.
We were able to successfully implement the LSTM model, taking time series analysis into consideration to make our predictions.
We were able to train our LLM with specific context in order to avoid hallucinations.

## What we learned
We learned the importance of data preprocessing and how it hugely impacts the accuracy of the machine learning model through our trial-and-error process. We also learned how to add context to the RAG model to get specific information from the LLM.

## What's next for Stock Lock
Stock Lock will soon be able to integrate as an extension to your stock portfolio, providing real-time pop-up suggestions while you are investing. It will also be scaled into a subscription-based stock advising application.


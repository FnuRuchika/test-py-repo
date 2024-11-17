import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
import requests
from sklearn.metrics import mean_squared_error
from datetime import datetime
import json
import os

# Initialize logging directory
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Helper function to log actions
def log_action(action, details):
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
        "details": details,
    }
    log_file = os.path.join(LOG_DIR, "data_flow_logs.json")
    with open(log_file, "a") as file:
        file.write(json.dumps(log_entry) + "\n")

@st.cache_data
def load_stock_data(symbol, interval):
    alpha_api_key = os.getenv("ALPHA_API_KEY", "5OM8ZCNGL507NBDE")
    alpha_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&apikey={alpha_api_key}"
    
    try:
        response = requests.get(alpha_url)
        response.raise_for_status()  # Raise HTTPError for bad responses
        data = response.json()

        # Debugging: Print the raw API response
        st.write("API Response:", data)

        # Check if the response contains the expected data
        time_series_key = f"Time Series ({interval})"
        if time_series_key not in data:
            raise ValueError("Invalid response structure or rate limit exceeded")

        # Parse the data into a DataFrame
        time_series_data = data[time_series_key]
        df = pd.DataFrame.from_dict(time_series_data, orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        numeric_columns = ["1. open", "2. high", "3. low", "4. close", "5. volume"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching stock data: {e}")
        return pd.DataFrame()
    except ValueError as ve:
        st.error(f"Data processing error: {ve}")
        return pd.DataFrame()

# Title and description
st.title("Riskalytics: Risk Scoring Meets Analytics")
st.write("Analyze stock risks with enhanced data observability and business insights.")

# User option to choose data source
data_source = st.radio(
    "Select Data Source",
    options=["Fetch Stock Data", "Upload Your Data"],
    index=0,
)

# Initialize stock_df as an empty DataFrame
stock_df = pd.DataFrame()

# Fetch stock data if the user selects "Fetch Stock Data"
if data_source == "Fetch Stock Data":
    stock_symbols = ["IBM", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NFLX", "NVDA", "AMD"]
    symbol = st.selectbox("Select Stock Symbol", options=stock_symbols, index=0)
    interval = st.selectbox("Select Interval", options=["1min", "5min", "15min", "30min", "60min"], index=1)

    stock_df = load_stock_data(symbol, interval)

# Allow user to upload their own data if "Upload Your Data" is selected
elif data_source == "Upload Your Data":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        stock_df = pd.read_csv(uploaded_file)
        log_action("Data Upload", f"User uploaded a custom dataset: {uploaded_file.name}")

        # Check if required columns are present
        required_columns = ["4. close"]
        if not all(col in stock_df.columns for col in required_columns):
            st.error(f"The uploaded file must contain the following columns: {', '.join(required_columns)}")
            stock_df = pd.DataFrame()  # Reset stock_df to avoid errors
        else:
            # Convert "date" column to datetime if it exists
            if "date" in stock_df.columns:
                stock_df["date"] = pd.to_datetime(stock_df["date"])
                stock_df.set_index("date", inplace=True)

if not stock_df.empty:
    # Display uploaded or fetched stock data
    st.write("### Stock Data", stock_df.head())

    # Moving Average Window Slider
    moving_avg_window = st.slider("Select Moving Average Window (days)", min_value=1, max_value=30, value=5)

    # Calculate Moving Average and Volatility
    stock_df["4. close"] = pd.to_numeric(stock_df["4. close"], errors="coerce")
    stock_df["moving_avg"] = stock_df["4. close"].rolling(window=moving_avg_window).mean()
    stock_df["volatility"] = stock_df["4. close"].rolling(window=moving_avg_window).std()

    # Log data access
    log_action("Data Access", f"Calculated moving average and volatility for window {moving_avg_window} days")

    # Filter by Date Range
    st.write("### Filter by Date Range")
    start_date = st.date_input("Start Date", value=stock_df.index.min().date())
    end_date = st.date_input("End Date", value=stock_df.index.max().date())
    stock_df = stock_df[(stock_df.index >= pd.to_datetime(start_date)) & (stock_df.index <= pd.to_datetime(end_date))]

    # Validate filtered data
    if not stock_df.empty:
        st.write("### Filtered Stock Data", stock_df.head())

        # Plot Stock Price and Moving Average
        st.write("### Stock Price and Moving Average")
        fig, ax = plt.subplots()
        ax.plot(stock_df.index, stock_df["4. close"], label="Closing Price", linewidth=2)
        ax.plot(stock_df.index, stock_df["moving_avg"], label=f"{moving_avg_window}-Day Moving Avg", linestyle="--")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.title("Stock Price and Moving Average")
        plt.grid()
        plt.legend()
        st.pyplot(fig)

        # Plot Volatility
        st.write("### Volatility Over Time")
        fig, ax = plt.subplots()
        ax.plot(stock_df.index, stock_df["volatility"], label="Volatility", color="orange", linewidth=2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.title("Volatility Over Time")
        plt.grid()
        plt.legend()
        st.pyplot(fig)

        # Load Model and Predict Risk Score
        try:
            model = joblib.load("model.pkl")
            scaler = joblib.load("scaler.pkl")  # Load the scaler for normalization
            X = stock_df[["4. close", "moving_avg"]].dropna()
            X_scaled = scaler.transform(X)  # Scale input features

            if not X.empty:
                risk_score = model.predict(X_scaled)[-1]  # Predict using the last row
                st.write(f"Predicted Risk Score: {risk_score}")
                log_action("Data Usage", f"Predicted risk score: {risk_score}")
            else:
                st.warning("Not enough data for prediction with the selected filters.")
        except FileNotFoundError as e:
            st.error(f"Model or scaler file not found: {e}")

    else:
        st.warning("No data available after applying filters. Please adjust your filters.")

else:
    st.warning("Please upload or fetch data to proceed.")

# Display Observability Dashboard
if st.checkbox("Show Observability Dashboard"):
    st.write("### Observability Dashboard")
    try:
        # Read and parse the log file
        with open(os.path.join(LOG_DIR, "data_flow_logs.json"), "r") as file:
            logs = [json.loads(line) for line in file]
        
        # Convert logs to a DataFrame
        logs_df = pd.DataFrame(logs)
        st.write("### Full Logs", logs_df)

        # Add filtering functionality
        if st.checkbox("Filter Logs by Action"):
            action_filter = st.selectbox("Select Action", options=logs_df["action"].unique())
            filtered_logs = logs_df[logs_df["action"] == action_filter]
            st.write("### Filtered Logs", filtered_logs)
    except FileNotFoundError:
        st.warning("No logs available yet.")

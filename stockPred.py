# stockPred.py (Modified for Due Diligence integration)

import os
import warnings
import gradio as gr
import pandas as pd
import numpy as np
import re
import requests
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from datetime import datetime, timedelta
import pickle
from io import StringIO
# from redditScraper import analyze # Commented out as redditScraper.py is not provided and causes import error
import json

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Download NLTK resources silently
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# --- Text Preprocessing and Sentiment Model Loading (Global) ---
def preprocess_text(text):
    """
    Cleans and preprocesses text for sentiment analysis.
    Removes URLs, special characters, converts to lowercase, and standardizes whitespace.
    """
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

lgbm_model = None
vectorizer = None
try:
    # Attempt to load pre-trained LightGBM model and TF-IDF vectorizer
    with open('lgbm_model (3).pkl', 'rb') as f:
        lgbm_model = pickle.load(f)
    with open('tfidf_vectorizer (3).pkl', 'rb') as l:
        vectorizer = pickle.load(l)
except FileNotFoundError:
    print("Error: 'lgbm_model (3).pkl' or 'tfidf_vectorizer (3).pkl' not found. Please ensure model files are in the correct directory.")
    # The application will likely fail if these are missing, but error handling is in generate_forecast_plot

# --- Global Sentiment Data Processing (to create adjusted_daily) ---
adjusted_daily = pd.DataFrame()
try:
    if lgbm_model and vectorizer: # Only proceed if models loaded successfully
        # Load the initial dataset (assuming 'df.csv' contains raw text data with 'date_only' and 'text' columns)
        df = pd.read_csv('df.csv')
        df['date'] = pd.to_datetime(df['date_only'])
        df = df.dropna(subset=['text', 'date_only']).copy() # Drop rows with missing text or date
        
        # Preprocess texts and transform them into features using the loaded vectorizer
        texts = df['text'].apply(preprocess_text)
        X_posts = vectorizer.transform(texts)
        
        # Predict sentiment using the loaded LightGBM model
        df['sentiment'] = lgbm_model.predict(X_posts)

        # Aggregate daily sentiment
        daily_sentiment = df.groupby('date_only').agg(
            avg_sentiment=('sentiment', 'mean'),
        ).reset_index()
        daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['date_only'])
        daily_sentiment.set_index('Date', inplace=True)

        # Prepare 'adjusted_daily' DataFrame for merging with stock data
        adjusted_daily = daily_sentiment.copy()
        adjusted_daily = adjusted_daily.reset_index() # Reset index for merging
        adjusted_daily['IN HOUSE SENTIMENT'] = adjusted_daily['avg_sentiment']
        adjusted_daily['Day_of_Week'] = adjusted_daily['Date'].dt.day_name()
        adjusted_daily.drop(columns=['avg_sentiment', 'date_only'], inplace=True) # Drop temporary columns
except FileNotFoundError:
    print("Error: 'df.csv' not found. Sentiment data cannot be loaded.")
except Exception as e:
    print(f"Error during initial sentiment data processing: {e}")


# --- Global Scaler Initialization ---
scaler = StandardScaler()


# --- Stock Data Fetching Function (Alpha Vantage) ---
def get_data_alpha_vantage(ticker, api_key):
    """
    Fetches daily stock data from Alpha Vantage and renames columns.
    Args:
        ticker (str): Stock ticker symbol.
        api_key (str): Alpha Vantage API key.
    Returns:
        pandas.DataFrame: DataFrame containing daily stock data, or empty DataFrame if error.
    """
    url = (
      f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY'
      f'&symbol={ticker}&outputsize=full&apikey={api_key}&datatype=csv'
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        
        # Check if the response is JSON (indicating an API error or info message)
        try:
            payload = r.json()
        except json.JSONDecodeError:
            payload = None

        if isinstance(payload, dict):
            # API returned JSON instead of CSV (e.g., error message, limit reached)
            msg = payload.get("Error Message") or payload.get("Note") or payload.get("Information")
            print(f"Alpha Vantage API response for {ticker}: {msg}")
            return pd.DataFrame()

        # Otherwise, parse as CSV
        df = pd.read_csv(StringIO(r.text), parse_dates=['timestamp'])
        df = df.set_index('timestamp').sort_index()

        # Rename columns (e.g., “1. open” → “open”) for easier access
        rename_map = {
            c: (c.split('.',1)[1].strip().lower() if '.' in c else c.lower())
            for c in df.columns
        }
        return df.rename(columns=rename_map)

    except requests.RequestException as e:
        print(f"Network/API error for {ticker}: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Processing error for {ticker}: {e}")
        return pd.DataFrame()


# --- Data Processing Function ---
def process_data(company_ticker, data):
    """
    Processes financial data, calculates previous close, and applies smoothing.
    Args:
        company_ticker (str): The stock ticker (for context, not directly used in processing data).
        data (pandas.DataFrame): Merged stock and sentiment data.
    Returns:
        pandas.DataFrame: Processed DataFrame.
    Raises:
        ValueError: If 'close' column is missing or data becomes empty after cleaning.
    """
    # Ensure index is datetime and sorted
    if not pd.api.types.is_datetime64_any_dtype(data.index):
        data.index = pd.to_datetime(data.index)
    data = data.sort_index()

    if 'close' not in data.columns:
        raise ValueError("The 'close' column is missing after merging in process_data. Check Alpha Vantage data columns.")

    # Calculate previous day's close price
    data['Prev_Close'] = data['close'].shift(1)
    data.dropna(inplace=True) # Drop rows with NaN (first row after shift will have NaN)

    # Drop specific columns if they exist (e.g., from merging)
    columns_to_drop = []
    if 'Day_of_Week' in data.columns:
        columns_to_drop.append('Day_of_Week')
    if 'date_only' in data.columns:
        columns_to_drop.append('date_only')

    if columns_to_drop:
        data = data.drop(columns=columns_to_drop)

    # Apply Gaussian smoothing to the close prices
    from scipy.ndimage import gaussian_filter1d
    sigma = 3 # Smoothing strength
    data['Smoothed_Close'] = gaussian_filter1d(data['close'], sigma=sigma)
    
    if data.empty:
        raise ValueError("Data became empty after processing. Check for sufficient data and NaNs.")
    return data


# --- Model Training Function ---
def train_final_model(data):
    """
    Trains an XGBoost model using the provided data.
    Args:
        data (pandas.DataFrame): Processed financial data for training.
    Returns:
        tuple: A tuple containing the best trained XGBoost model and evaluation metrics (rmse, mae, r2).
    Raises:
        ValueError: If no data is available for training.
    """
    data = data.sort_index()
    features = ["open", "high", "low", "volume", "IN HOUSE SENTIMENT"]
    target = "close"
    data.dropna(inplace=True) # Ensure no NaNs in features/target before training
    X = data[features]
    y = data[target]

    if X.empty:
        raise ValueError("No data available for training after dropping NaNs. Check data import and processing.")

    # Scale features
    X_scaled = scaler.fit_transform(X)
    
    # Initialize XGBoost Regressor and define parameters for GridSearchCV
    xgb_model = XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.05, max_depth=5)
    grid_params = {"n_estimators": [100, 200], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 5, 7]}
    
    # Perform Grid Search Cross-Validation to find best model parameters
    grid_search = GridSearchCV(xgb_model, grid_params, scoring="neg_root_mean_squared_error", verbose=0, n_jobs=-1)
    grid_search.fit(X_scaled, y)
    best_xgb = grid_search.best_estimator_
    
    # Evaluate the best model
    y_pred = best_xgb.predict(X_scaled)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    metrics = (rmse, mae, r2)
    return best_xgb, metrics


# --- Feature Prediction Function ---
def predict_features(data):
    """
    Predicts future values for input features (open, high, low, volume, sentiment)
    using individual XGBoost models for each feature.
    Args:
        data (pandas.DataFrame): Historical data for feature prediction.
    Returns:
        pandas.DataFrame: DataFrame containing predicted features for the next trading day.
    Raises:
        ValueError: If not enough data is available for lagged feature creation or model training.
    """
    last_date = data.index[-1]
    time = data.copy()
    numeric_cols = time.select_dtypes(include=['int64', 'float64']).columns
    # Identify features to predict (excluding target and derived columns)
    features_to_predict = [col for col in numeric_cols if col not in ['close', 'Prev_Close', 'Smoothed_Close']]
    
    time['Date'] = pd.to_datetime(time.index)
    time.set_index('Date', inplace=True)

    predictions = {}
    
    # Helper function to create lagged features for a given feature
    def create_lagged_features(data_frame, feature, lags=5):
        for lag in range(1, lags + 1):
            data_frame[f'{feature}_lag_{lag}'] = data_frame[feature].shift(lag)
        return data_frame

    # Predict each feature independently
    for feature in features_to_predict:
        df_lagged = create_lagged_features(time, feature, lags=5)
        df_lagged.dropna(inplace=True)

        if df_lagged.empty or len(df_lagged) < 2:
            # If not enough data for lagging/training, fill with last known value as a fallback
            last_known_val = time[feature].iloc[-1] if not time[feature].empty else 0
            predictions[feature] = [last_known_val]
            # print(f"Warning: Not enough data to create lagged features or train model for {feature}. Using last known value.")
            continue # Skip to next feature

        X = df_lagged[[f'{feature}_lag_{i}' for i in range(1, 6)]]
        y = df_lagged[feature]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        xgb_model = XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.05, max_depth=5)
        xgb_model.fit(X_train, y_train)

        # Predict the next value using the last available lagged values
        last_known_values = df_lagged[[f'{feature}_lag_{i}' for i in range(1, 6)]].iloc[-1].values
        next_pred = xgb_model.predict([last_known_values])[0]
        predictions[feature] = [next_pred]

    predictions_df = pd.DataFrame(predictions)
    # Calculate the next business day for the prediction
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=1, freq='B')
    predictions_df.index = future_dates
    predictions_df['Date'] = predictions_df.index.normalize()
    predictions_df['Day'] = predictions_df['Date'].dt.strftime('%A')
    
    # Reorder columns to have Date and Day first, then predicted features
    predictions_df = predictions_df[['Date', 'Day'] + features_to_predict]
    return predictions_df


# --- Final Price Prediction Function ---
def final_preds(predictions_df, model):
    """
    Makes final predictions using the trained model on predicted features.
    Args:
        predictions_df (pandas.DataFrame): DataFrame with predicted features for the next day.
        model (XGBRegressor): The trained XGBoost model for price prediction.
    Returns:
        pandas.DataFrame: DataFrame containing the 'Predicted_Close' price.
    Raises:
        ValueError: If predicted features DataFrame is empty.
    """
    features = ["open", "high", "low", "volume", "IN HOUSE SENTIMENT"]

    # Ensure all required features are present in the predictions_df
    for feature in features:
        if feature not in predictions_df.columns:
            predictions_df[feature] = np.nan # This should ideally not happen if predict_features works correctly

    predictions_df_ordered = predictions_df[features]
    predictions_df_ordered.dropna(inplace=True) # Drop any rows with NaN in features

    if predictions_df_ordered.empty:
        raise ValueError("Predicted features DataFrame is empty after dropping NaNs. Cannot make final price prediction.")

    # Scale the predicted features using the same scaler fitted on training data
    X_scaled_predictions = scaler.transform(predictions_df_ordered)
    
    best_xgb = model
    y_pred_new = best_xgb.predict(X_scaled_predictions)
    predictions_df["Predicted_Close"] = y_pred_new
    
    return predictions_df[["Predicted_Close"]]


# --- Test Function (for comparing values) ---
def test(prev_val, new_val):
    """
    Compares a previous scalar value to a new scalar value and determines percentage change.
    Args:
        prev_val (float): Previous value.
        new_val (float): New value.
    Returns:
        float: Percentage change.
    """
    if prev_val == 0: # Avoid division by zero
        return 0.0
    percent_change = ((new_val - prev_val) / prev_val) * 100
    return round(percent_change, 2)


# --- Gradio Wrapper Function ---
def generate_forecast_plot(company_ticker):
    """
    Generates the stock price forecast and markdown output for the Gradio interface.
    Returns markdown string, status message, and the numerical forecast details.
    Args:
        company_ticker (str): The stock ticker symbol.
    Returns:
        tuple: (markdown_output, status_output, percent_change, risk_level, predicted_close, last_actual_close)
               Returns None for numerical values on error.
    """
    fig = None # Initialize fig to None for safe cleanup
    markdown_output = "An unexpected error occurred. Please try again or check the inputs."
    status_output = "❌ Failed"
    percent_change = None
    risk_level = None
    predicted_close = None
    last_actual_close = None

    try:
        if not company_ticker:
            return "Please enter a company ticker.", "❌ Input missing", None, None, None, None

        # Get Alpha Vantage API key from environment variable
        alpha_api_key = os.getenv('alpha_key')
        if not alpha_api_key:
            return "Alpha Vantage API key not found in environment variables. Please set the 'alpha_key' environment variable.", "❌ API Key Missing", None, None, None, None

        # Font properties (not directly used for plotting anymore, but kept for completeness if needed elsewhere)
        # These lines are commented out as they are not critical for functionality and might cause warnings
        # custom_font_properties = {'family': 'sans-serif', 'size': 10}
        # font_path = "AfacadFlux-VariableFont_slnt,wght[1].ttf"
        # try:
        #     temp_custom_font_obj = fm.FontProperties(fname=font_path)
        #     custom_font_properties = temp_custom_font_obj
        # except FileNotFoundError:
        #     pass
        # except Exception as e:
        #     pass

        # Ensure sentiment data and models are loaded
        global adjusted_daily, scaler, lgbm_model, vectorizer, preprocess_text
        if adjusted_daily.empty or lgbm_model is None or vectorizer is None:
            return "Required sentiment data or model files could not be loaded. Please ensure 'df.csv', 'lgbm_model (3).pkl', and 'tfidf_vectorizer (3).pkl' are correctly placed.", "❌ Data/Model Error", None, None, None, None
        
        # 1. Fetch stock data
        stonks = get_data_alpha_vantage(company_ticker, alpha_api_key)
        if stonks.empty:
            return f"Could not fetch data for {company_ticker}. Please check the ticker and API key.", "❌ Data Fetch Error", None, None, None, None

        stonks['Date'] = stonks.index.normalize()
        stonks = stonks.reset_index(drop=True)

        # 2. Merge stock data with sentiment data
        merged_df = pd.merge(stonks, adjusted_daily, on='Date', how='inner')

        if merged_df.empty:
            return f"No common dates found between stock data for {company_ticker} and sentiment data. Please check data ranges and ensure df.csv covers the same period.", "❌ Merge Error", None, None, None, None

        merged_df = merged_df.set_index('Date').sort_index()

        # 3. Process the merged data
        data1 = process_data(company_ticker, merged_df.copy())
        if data1.empty:
            return "The 'process_data' function returned an empty DataFrame. This might indicate insufficient historical data after cleaning or too many NaNs.", "❌ Process Data Error", None, None, None, None

        # 4. Train the model
        model, metrics = train_final_model(data1)

        # 5. Predict future features
        features = predict_features(data1)
        
        # 6. Make final price predictions
        preds = final_preds(features, model)
        predictions_df = preds

        # Get the last actual close date and price
        last_actual_date = data1.index[-1]
        last_actual_close = data1['close'].iloc[-1]

        # Get the predicted date and price
        predicted_date = predictions_df.index[0]
        predicted_close = predictions_df['Predicted_Close'].iloc[0]

        # --- Generate Markdown Output for this module ---
        percent_change = test(last_actual_close, predicted_close)

        markdown_output = f"## {company_ticker} Forecasted Change\n\n"
        markdown_output += f"**Expected Change (Next Trading Day):** {percent_change:.2f}%\n\n"

        # Add recommendation based on percent change
        if percent_change < 0:
            markdown_output += "**Recommendation:** Short or consider buying inverse asset.\n\n"
        elif percent_change > 0: # Using a small positive threshold for "buy" recommendation
            markdown_output += "**Recommendation:** Long or consider buying asset.\n\n"
        else: # For small positive, zero, or very small negative changes
            markdown_output += "**Recommendation:** Hold.\n\n"

        abs_percent_change = abs(percent_change)
        risk_level = ""
        if 0 <= abs_percent_change <= 1:
            risk_level = "Low Risk"
        elif 1 < abs_percent_change <= 4.5:
            risk_level = "Moderate Risk"
        else: # abs_percent_change > 4.5
            risk_level = "High Risk"
        
        markdown_output += f"**Risk Assessment:** {risk_level}\n\n"

        markdown_output += "---\n\n" # Separator

        status_output = "✅ Forecast complete!"

        # Return all relevant data points
        return markdown_output, status_output, percent_change, risk_level, predicted_close, last_actual_close

    except ValueError as e:
        markdown_output = f"Error: {e}"
        status_output = "❌ Value Error"
        return markdown_output, status_output, None, None, None, None
    except Exception as e:
        markdown_output = f"An unexpected error occurred: {e}"
        status_output = "❌ Unexpected Error"
        return markdown_output, status_output, None, None, None, None
    finally:
        if fig is not None:
            plt.close(fig) # Ensure any matplotlib figures are closed to free memory

# --- Gradio Interface Setup (This block is for stand-alone use and not called by dueDiligence.py) ---
with gr.Blocks() as demo:
    gr.Markdown("# Stock Price Forecast App")

    ticker_input = gr.Textbox(label="Company Ticker", placeholder="e.g., PG, MSFT, AAPL")
    
    output_markdown = gr.Markdown()
    status_output = gr.Textbox(label="Status", interactive=False)

    generate_btn = gr.Button("Generate Forecast")

    generate_btn.click(
        fn=generate_forecast_plot,
        inputs=[ticker_input],
        outputs=[output_markdown, status_output]
        # The additional numerical outputs are for internal use by dueDiligence.py
    )

# demo.queue()
# demo.launch() # Commented out so it's not launched when imported by another script

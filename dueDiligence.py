# dueDiligence.py (Corrected for LLM Report Display)

import gradio as gr
import os
import sys
import openai
import pandas as pd
import matplotlib.pyplot as plt

# Ensure imports work by adding current directory to path
sys.path.append(os.path.dirname(__file__))

# Import specific functions/components from other scripts
# These imports assume the files stockPred.py, sentSearch.py, financialHive.py,
# and sectorSent.py are in the same directory.
try:
    # stock_forecast_function now returns: markdown, status, percent_change, risk_level, predicted_close, last_actual_close
    from stockPred import generate_forecast_plot as stock_forecast_function
except ImportError as e:
    print(f"Error importing stockPred: {e}")
    stock_forecast_function = None

try:
    # sentiment_analysis_function now returns: markdown_output_sent, table_df_sent
    from sentSearch import analyze_ticker as sentiment_analysis_function
except ImportError as e:
    print(f"Error importing sentSearch: {e}")
    sentiment_analysis_function = None

try:
    from financialHive import summarize_clusters_wrapper as hive_news_summary_function
except ImportError as e:
    print(f"Error importing financialHive: {e}")
    hive_news_summary_function = None

try:
    # sector_sentiment_function now returns: fig, avg_by_sector_dict
    from sectorSent import fetch_and_plot_avg_sentiment as sector_sentiment_function
except ImportError as e:
    print(f"Error importing sectorSent: {e}")
    sector_sentiment_function = None

# --- OpenAI API Setup ---
# Get OpenAI API key from environment variables
open_api_key = os.getenv('open_api_key')
# Initialize the OpenAI client
openai_client = None
if open_api_key:
    openai_client = openai.OpenAI(api_key=open_api_key)
else:
    print("Warning: OpenAI API key not found. Please set the 'open_api_key' environment variable for LLM functionality.")


def get_llm_report(stock_data, sentiment_data, hive_news, sector_data_str):
    """
    Generates an in-depth due diligence report using an LLM.
    Args:
        stock_data (dict): Dictionary containing stock forecast details.
        sentiment_data (dict): Dictionary containing ticker-specific sentiment analysis.
        hive_news (str): String containing general financial hive news briefings.
        sector_data_str (str): String representation of Dow Jones sector sentiment.
    Returns:
        str: The AI-generated due diligence report.
    """
    if not openai_client:
        return "LLM functionality is not available. OpenAI API key is missing."

    prompt = f"""
    You are an expert financial analyst. Your task is to provide a comprehensive due diligence report for a given stock, based on the provided data from various financial analysis modules. Analyze the data critically and present a balanced view, highlighting key insights, potential risks, and opportunities.

    Here is the compiled data:

    ---
    ### 1. Stock Price Forecast:
    {stock_data.get('markdown_output', 'No stock forecast data available.')}
    Predicted Change: {stock_data.get('percent_change', 'N/A')}%
    Risk Level: {stock_data.get('risk_level', 'N/A')}
    Predicted Close Price (Next Trading Day): ${stock_data.get('predicted_close', 'N/A'):.2f}
    Last Actual Close Price: ${stock_data.get('last_actual_close', 'N/A'):.2f}

    ---
    ### 2. Ticker-Specific Sentiment Analysis:
    {sentiment_data.get('markdown_output', 'No ticker-specific sentiment data available.')}
    Summary Table of Articles (Top 5 entries, if available):
    ```
    {sentiment_data.get('table_summary', 'N/A')}
    ```

    ---
    ### 3. General Financial Hive News Briefings (Top Market Sentiment):
    {hive_news if hive_news else 'No general financial hive news briefings available.'}

    ---
    ### 4. Dow Jones Sector Sentiment:
    {sector_data_str if sector_data_str else 'No sector sentiment data available.'}

    ---

    **Instructions for the Due Diligence Report:**
    1.  **Overall Summary:** Start with a concise executive summary of the stock's current standing and outlook based on the combined data.
    2.  **Market Sentiment Analysis:**
        * Elaborate on the ticker-specific sentiment: Is it positive, negative, or mixed? What are the main drivers of this sentiment?
        * Connect it to broader market trends from the "Financial Hive News Briefings" and how they might influence the specific stock or sector.
        * Discuss the sector sentiment: How does the stock's sector compare to others? Is this a tailwind or a headwind for the stock?
    3.  **Stock Performance & Outlook:**
        * Analyze the forecasted price change and risk assessment. What does this imply for short-term trading or investment?
        * Discuss any underlying factors from the sentiment data that might support or contradict the forecast.
    4.  **Key Considerations/Risks:** Identify potential risks or negative factors based on the data (e.g., negative sentiment trends, high risk assessment, sector-specific headwinds).
    5.  **Opportunities/Strengths:** Identify potential opportunities or positive factors (e.g., strong positive sentiment, favorable sector trends).
    6.  **Recommendation:** Provide a concluding recommendation (e.g., "Buy," "Hold," "Sell," or "Monitor Closely") with a brief justification based *only* on the provided data.
    7.  **Format:** Use clear headings and bullet points where appropriate for readability. Maintain a professional and objective tone.
    """

    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4o", # Using gpt-4o for comprehensive analysis. Can be changed to "gemini-2.0-flash"
            messages=[
                {"role": "system", "content": "You are a highly skilled financial analyst. Generate a comprehensive due diligence report."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7, # Adjust creativity; 0.7 is a good balance for analytical tasks.
            max_tokens=1500 # Ensure sufficient length for an in-depth report.
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error generating LLM report: {e}"


def perform_due_diligence(ticker, progress=gr.Progress()):
    """
    Performs due diligence by gathering data from various modules
    and generating an LLM-based report.
    Args:
        ticker (str): The stock ticker symbol.
        progress (gr.Progress): Gradio progress tracker.
    Returns:
        str: The final AI-generated due diligence report.
    """
    if not ticker or not ticker.strip():
        yield "Please enter a valid ticker symbol." # Yield for immediate display
        return

    ticker = ticker.strip().upper()
    
    # Initialize a placeholder for the overall report and track progress
    current_status_message = "### Generating Due Diligence Report...\n\n"
    yield current_status_message # Yield initial message to show progress immediately
    
    # --- 1. Stock Price Forecast ---
    stock_data = {}
    progress(0.2, desc=f"Fetching stock forecast for {ticker}...")
    try:
        if stock_forecast_function:
            # stock_forecast_function returns: markdown, status, percent_change, risk_level, predicted_close, last_actual_close
            markdown_output, status_output, percent_change, risk_level, predicted_close, last_actual_close = stock_forecast_function(ticker)
            stock_data = {
                'markdown_output': markdown_output,
                'status_output': status_output,
                'percent_change': percent_change,
                'risk_level': risk_level,
                'predicted_close': predicted_close,
                'last_actual_close': last_actual_close
            }
            if status_output.startswith("❌"):
                current_status_message += f"**Stock Forecast Status:** {status_output}\n*{markdown_output}*\n\n"
                yield current_status_message # Update with error
                return # Exit if critical error
            else:
                current_status_message += f"**Stock Forecast Status:** {status_output}\n\n"
        else:
            stock_data = {'error': "Stock prediction module not loaded."}
            current_status_message += "**Stock Forecast Error:** Module not loaded.\n\n"
        yield current_status_message
    except Exception as e:
        stock_data = {'error': f"Error getting stock forecast: {e}"}
        current_status_message += f"**Stock Forecast Error:** {e}\n\n"
        yield current_status_message # Update with error
        return # Exit on error


    # --- 2. Ticker-Specific Sentiment Analysis ---
    sentiment_data = {}
    progress(0.4, desc=f"Analyzing ticker-specific sentiment for {ticker}...")
    try:
        if sentiment_analysis_function:
            # analyze_ticker returns: markdown_output_sent, table_df_sent
            markdown_output_sent, table_df_sent = sentiment_analysis_function(ticker, fetch_content=False)
            sentiment_data = {
                'markdown_output': markdown_output_sent,
                'table_summary': table_df_sent.to_markdown(index=False) if table_df_sent is not None else "No sentiment articles found."
            }
            current_status_message += "**Ticker Sentiment Status:** Completed\n\n"
        else:
            sentiment_data = {'error': "Sentiment analysis module not loaded."}
            current_status_message += "**Ticker Sentiment Error:** Module not loaded.\n\n"
        yield current_status_message
    except Exception as e:
        sentiment_data = {'error': f"Error getting ticker sentiment: {e}"}
        current_status_message += f"**Ticker Sentiment Error:** {e}\n\n"
        yield current_status_message


    # --- 3. General Financial Hive News Briefings ---
    hive_news_summary = ""
    progress(0.6, desc="Generating general financial hive news briefings...")
    try:
        if hive_news_summary_function:
            hive_news_summary = hive_news_summary_function()
            current_status_message += "**Financial Hive News Status:** Completed\n\n"
        else:
            hive_news_summary = "Financial Hive news module not loaded."
            current_status_message += "**Financial Hive News Error:** Module not loaded.\n\n"
        yield current_status_message
    except Exception as e:
        hive_news_summary = f"Error generating financial hive news: {e}"
        current_status_message += f"**Financial Hive News Error:** {e}\n\n"
        yield current_status_message


    # --- 4. Dow Jones Sector Sentiment ---
    sector_sentiment_str = ""
    progress(0.8, desc="Fetching Dow Jones sector sentiment...")
    try:
        if sector_sentiment_function:
            # fetch_and_plot_avg_sentiment returns fig, avg_by_sector_dict
            _, avg_by_sector_dict = sector_sentiment_function()
            sector_sentiment_str = "\n".join([f"- {sector}: {sentiment:.2f}%" for sector, sentiment in avg_by_sector_dict.items()])
            current_status_message += "**Sector Sentiment Status:** Completed\n\n"
        else:
            sector_sentiment_str = "Sector sentiment module not loaded."
            current_status_message += "**Sector Sentiment Error:** Module not loaded.\n\n"
        yield current_status_message
    except Exception as e:
        sector_sentiment_str = f"Error fetching sector sentiment: {e}"
        current_status_message += f"**Sector Sentiment Error:** {e}\n\n"
        yield current_status_message


    # --- 5. Generate LLM Report ---
    progress(0.9, desc="Compiling final report with LLM...")
    llm_report = get_llm_report(
        stock_data,
        sentiment_data,
        hive_news_summary,
        sector_sentiment_str
    )
    current_status_message += "**LLM Report Generation Status:** Completed\n\n"
    # Instead of returning, yield the final report as well
    yield llm_report # THIS IS THE KEY CHANGE

    progress(1.0, desc="Due diligence complete!")
    # No return statement here, as the final report is already yielded


# --- Gradio Interface Setup ---
with gr.Blocks(title="Comprehensive Due Diligence Report", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 📊 Comprehensive Stock Due Diligence Report")
    gr.Markdown(
        "Enter a stock ticker symbol to generate an in-depth due diligence report. "
        "This report combines stock price forecasts, ticker-specific sentiment, "
        "general market news, and sector sentiment into an AI-powered analysis."
    )

    with gr.Row():
        ticker_input = gr.Textbox(
            label="📈 Enter Stock Ticker Symbol",
            placeholder="e.g., AAPL, MSFT, TSLA",
            value="AAPL" # Default value for testing
        )
        generate_report_btn = gr.Button(
            "🚀 Generate Due Diligence Report",
            variant="primary",
            size="lg"
        )

    # The progress bar is automatically managed by Gradio when show_progress=True is used.
    # We just need a placeholder for the output.
    output_report = gr.Markdown(label="📝 Due Diligence Report")

    # The click event for the button that triggers the due diligence process
    generate_report_btn.click(
        fn=perform_due_diligence,
        inputs=[ticker_input],
        outputs=[output_report],
        show_progress="full" # Show a detailed progress bar
    )

    gr.Examples(
        examples=[
            ["AAPL"],
            ["MSFT"],
            ["NVDA"],
            ["GOOGL"]
        ],
        inputs=[ticker_input],
        label="💡 Try these popular tickers"
    )

    gr.Markdown("""
    ---
    **Disclaimer:** This report is generated by an AI based on available data and should not be considered financial advice.
    Always conduct your own research and consult with a financial professional before making investment decisions.
    """)

# To launch the Gradio app:
# app.queue()
# app.launch()

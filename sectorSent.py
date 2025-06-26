# sectorSent.py (Modified for Due Diligence integration)

import gradio as gr
import requests
from bs4 import BeautifulSoup
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np # Import numpy for np.linspace

def fetch_and_plot_avg_sentiment():
    """
    Fetches live sentiment data for Dow Jones sectors, plots it, and returns the data.
    Returns:
        tuple: A tuple containing the matplotlib figure and a dictionary of average sentiments by sector.
    """
    # 1. Fetch the heatmap page from the external API
    url = "https://heatmap-web.onrender.com/view-dow-heatmap"
    resp = requests.get(url)
    resp.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

    # 2. Extract Plotly trace JSON from the HTML content
    soup = BeautifulSoup(resp.text, "html.parser")
    # Search for the script tag containing 'Plotly.newPlot' and extract its content
    script = soup.find("script", string=re.compile(r"Plotly\.newPlot", re.DOTALL))
    
    if not script:
        raise ValueError("Could not find Plotly.newPlot script in the fetched HTML. The structure of the heatmap page might have changed or it's not fully loaded.")
    
    script_content = script.string
    # Corrected regex: Removed extra backslashes before ( and [
    match = re.search(r"Plotly\.newPlot\([^,]+,\s*(\[\{.*?\}\])", script_content, re.DOTALL)
    
    if not match:
        raise ValueError("Could not extract Plotly trace data from the script content. The Plotly data format might have changed.")

    # Parse the extracted JSON string into a Python object, taking the first trace
    trace = json.loads(match.group(1))[0]

    # 3. Build DataFrame of labels, parents, and sentiment from customdata
    labels = trace["labels"]
    parents = trace["parents"]
    customdata = trace.get("customdata", []) # Get customdata, default to empty list if not present

    # Create a mapping from label (ticker) to sentiment score
    senti_map = {item[0]: float(item[1]) for item in customdata}

    # Create a pandas DataFrame from labels and parents
    df = pd.DataFrame({
        "label": labels,
        "parent": parents
    })
    # Map sentiment scores to the labels
    df["sentiment"] = df["label"].map(senti_map)
    # Extract the top-level sector from the 'parent' path (e.g., "Dow/Tech" -> "Tech")
    # Handle cases where 'parent' might not contain '/'
    df["sector"] = df["parent"].apply(lambda x: x.split("/")[1] if isinstance(x, str) and "/" in x else "")

    # 4. Compute average sentiment by top-level sector
    # Filter out entries where parent is empty (which typically represent the overall "Dow" group or unclassified items)
    tickers_df = df[df["parent"] != ""].dropna(subset=["sentiment", "sector"])
    
    if tickers_df.empty:
        # If no valid ticker data, return an empty plot and dict
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No sector sentiment data available.", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title("Dow Jones Sector Sentiment")
        plt.axis('off') # Hide axes for empty plot
        return fig, {}

    # Group by sector and calculate the mean sentiment for each
    avg_by_sector = tickers_df.groupby("sector")["sentiment"].mean()

    # 5. Plot bar chart using Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6)) # Added figsize for better chart proportions
    sectors = avg_by_sector.index.tolist()
    values = avg_by_sector.values
    # Create a bar chart with a color gradient for better visual appeal
    ax.bar(sectors, values, color=plt.cm.Paired(np.linspace(0, 1, len(sectors))))
    ax.set_xticklabels(sectors, rotation=45, ha="right") # Rotate x-axis labels for readability
    ax.set_ylabel("Average Sentiment (%s)" % ("%")) # Changed to correctly display percentage symbol
    ax.set_title("Average Dow Jones Sentiment by Sector")

    # Dynamic margins for y-axis to ensure bars are clearly visible
    min_val, max_val = values.min(), values.max()
    margin = (max_val - min_val) * 0.05
    ax.set_ylim(min_val - margin, max_val + margin)

    plt.tight_layout() # Adjust plot to ensure everything fits without overlapping

    # Return both the matplotlib figure (for Gradio's gr.Plot component)
    # and the average sentiment data as a dictionary (for LLM consumption).
    return fig, avg_by_sector.to_dict()

# Gradio interface for stand-alone use (not directly called by dueDiligence.py)
demo = gr.Interface(
    fn=fetch_and_plot_avg_sentiment,
    inputs=[],
    outputs=[
        gr.Plot(label="Sector Sentiment Bar Chart"),
        gr.Json(label="Raw Sector Sentiment Data", visible=False) # Data is visible=False by default, used internally
    ],
    title="Dow Jones Sector Sentiment",
    description="Fetches live sentiment data and displays the average sentiment per Dow Jones sector."
)

# if __name__ == "__main__":
#     demo.launch() # Commented out so it's not launched when imported by another script


import gradio as gr
import os
import sys

# Ensure imports work by adding current directory to path
sys.path.append(os.path.dirname(__file__))

# Import your demos (or fallback to error‐messages)
try:
    from stockPred import demo as stock_predictor_demo_block
except Exception as e:
    # Capture the error message here
    error_message_stock = f"Error loading Stock Predictor: {e}"
    stock_predictor_demo_block = gr.Interface(lambda: error_message_stock, [], "text")
    print(error_message_stock) # Add print for debugging

try:
    from financialHive import demo as financial_hive_demo_block
except Exception as e:
    # Capture the error message here
    error_message_hive = f"Error loading Financial Hive: {e}"
    financial_hive_demo_block = gr.Interface(lambda: error_message_hive, [], "text")
    print(error_message_hive) # Add print for debugging

try:
    from sectorSent import demo as sector_sent_demo_block
except Exception as e:
    # Capture the error message here
    error_message_sector = f"Error loading Sector Sentiment: {e}"
    sector_sent_demo_block = gr.Interface(lambda: error_message_sector, [], "text")
    print(error_message_sector) # Add print for debugging

try:
    from sentSearch import demo as sent_search_block
except Exception as e:
    # Capture the error message here
    error_message_sent = f"Error loading Sentiment Search: {e}"
    sent_search_demo_block = gr.Interface(lambda: error_message_sector, [], "text")
    print(error_message_sent) # Add print for debugging

# --- New Import for Due Diligence ---
try:
    from dueDiligence import app as due_diligence_demo_block # Assuming the Gradio app in dueDiligence.py is named 'app'
except Exception as e:
    error_message_due_diligence = f"Error loading Due Diligence: {e}"
    due_diligence_demo_block = gr.Interface(lambda: error_message_due_diligence, [], "text")
    print(error_message_due_diligence) # Add print for debugging

# --- New Import for Reddit Analysis ---
try:
    from redditScraper import create_gradio_interface
    reddit_analysis_demo_block = create_gradio_interface()
except Exception as e:
    error_message_reddit = f"Error loading Reddit Analysis: {e}"
    reddit_analysis_demo_block = gr.Interface(lambda: error_message_reddit, [], "text")
    print(error_message_reddit) # Add print for debugging


with gr.Blocks() as app:
    # 1) Button row
    with gr.Row():
        btn_stock = gr.Button("Stock Predictor")
        btn_hive = gr.Button("Financial Hive")
        btn_sector = gr.Button("Sector Sentiment")
        btn_sent=gr.Button("Sentiment Search")
        btn_due_diligence = gr.Button("Due Diligence") # New button
        btn_reddit = gr.Button("Reddit Analysis") # New Reddit Analysis button

    # 2) Containers for each demo
    # Set stock_container to be visible by default
    stock_container = gr.Column(visible=True)
    hive_container = gr.Column(visible=False)
    sector_container = gr.Column(visible=False)
    sent_container = gr.Column(visible=False)
    due_diligence_container = gr.Column(visible=False) # New container, initially hidden
    reddit_container = gr.Column(visible=False) # New Reddit container, initially hidden

    # 3) Inside each container, render the corresponding demo
    with stock_container:
        gr.Markdown("## Stock Predictor")
        stock_predictor_demo_block.render()
    with hive_container:
        gr.Markdown("## Financial Hive")
        financial_hive_demo_block.render()
    with sector_container:
        gr.Markdown("## Sector Sentiment")
        sector_sent_demo_block.render()
    with sent_container:
        gr.Markdown("## Sentiment Search")
        sent_search_block.render()
    with due_diligence_container: # New container for Due Diligence
        gr.Markdown("## Comprehensive Due Diligence Report")
        due_diligence_demo_block.render()
    with reddit_container: # New container for Reddit Analysis
        gr.Markdown("## Reddit Analysis with HuggingFace Upload")
        reddit_analysis_demo_block.render()

    # 4) Wire up buttons - updated to include the new containers
    btn_stock.click(
        lambda: (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)),
        outputs=[stock_container, hive_container, sector_container, sent_container, due_diligence_container, reddit_container]
    )
    btn_hive.click(
        lambda: (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)),
        outputs=[stock_container, hive_container, sector_container, sent_container, due_diligence_container, reddit_container]
    )
    btn_sector.click(
        lambda: (gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)),
        outputs=[stock_container, hive_container, sector_container, sent_container, due_diligence_container, reddit_container]
    )
    btn_sent.click(
        lambda: (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)),
        outputs=[stock_container, hive_container, sector_container, sent_container, due_diligence_container, reddit_container]
    )
    # New button click event for Due Diligence
    btn_due_diligence.click(
        lambda: (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)),
        outputs=[stock_container, hive_container, sector_container, sent_container, due_diligence_container, reddit_container]
    )
    # New button click event for Reddit Analysis
    btn_reddit.click(
        lambda: (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)),
        outputs=[stock_container, hive_container, sector_container, sent_container, due_diligence_container, reddit_container]
    )


if __name__ == "__main__":
    app.launch()
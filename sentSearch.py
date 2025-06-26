import os, re, datetime
import numpy as np, spacy, hdbscan, praw, openai
from collections import Counter
import datetime
import pandas as pd
import joblib
import scipy.sparse as sp
import gradio as gr
import requests # Added for the new functionality
import json # Added for the new functionality
from bs4 import BeautifulSoup # Added for the new functionality


# Load the dataset
posts_df = pd.read_csv('df.csv')

# Convert 'date_only' to datetime objects
posts_df["time"] = pd.to_datetime(posts_df["date_only"])

# Filter for recent posts (currently commented out)
# seven_days_ago = datetime.datetime.utcnow() - datetime.timedelta(days=7)
# posts_df = posts_df[posts_df["time"] >= seven_days_ago]

# Downsample to 2000 rows if there are more
if len(posts_df) > 2000:
    posts_df = posts_df.sample(n=2000, random_state=42)

"""**Preprocess our texts**"""
def simple_preprocess(text):
    """
    Cleans and preprocesses text by lowercasing, removing special characters,
    and standardizing whitespace.
    """
    text = text.lower().strip()
    # Remove anything that is not a lowercase letter, number, or whitespace
    text = re.sub(r'[^a-z0-9\s]+', '', text)
    # Replace multiple whitespaces with a single space
    text = re.sub(r'\s+', ' ', text)
    return text

"""**Embed**"""
# Drop rows where 'text' is missing
posts_df.dropna(subset=['text'], inplace=True)

# Extract texts for processing
texts = posts_df['text'].tolist()

# Load pre-trained classifier and vectorizer
# These models are used to identify "positive" texts and generate embeddings
classifier = joblib.load('AutoClassifier.pkl')
vectorizer = joblib.load("AutoVectorizer.pkl")

# Transform texts into feature vectors using the vectorizer
X = vectorizer.transform(texts)

# Ensure the feature matrix has the expected number of features (columns)
# This handles cases where the current data might result in fewer features than the model expects
expected_features = 5000
if X.shape[1] < expected_features:
    n_missing = expected_features - X.shape[1]
    # Add zero-filled columns to match the expected feature dimension
    X = sp.hstack([X, sp.csr_matrix((X.shape[0], n_missing))])

# Predict sentiment (or a similar classification) using the classifier
predictions = classifier.predict(X)

# Filter for texts classified as "positive" (prediction == 1)
positive_texts = [text for text, pred in zip(texts, predictions) if pred == 1]

# Preprocess the positive texts
processed_texts = [simple_preprocess(t) for t in positive_texts]

# Generate embeddings for the processed positive texts using the vectorizer
# (Original code commented out SentenceTransformer, using vectorizer for embeddings)
embeddings = vectorizer.transform(processed_texts)

"""**Cluster our embeddings**"""
# Initialize HDBSCAN clusterer with specified parameters
# min_cluster_size: minimum number of samples in a cluster
# min_samples: the number of samples in a neighborhood for a point to be considered as a core point
clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=3)
# Fit the clusterer to the embeddings and predict cluster labels
labels = clusterer.fit_predict(embeddings)

# Count occurrences of each label
counts = Counter(labels)

# Filter out clusters smaller than min_cluster_size (10) as noise (-1)
# This reinforces the min_cluster_size
labels = np.array([lbl if counts[lbl] >= 10 else -1 for lbl in labels])

# Organize texts into clusters based on their labels
clusters = {}
for text, lbl in zip(positive_texts, labels):
    clusters.setdefault(lbl, []).append(text)

# --- OpenAI API Setup ---
# Get OpenAI API key from environment variables
open_api_key = os.getenv('open_api_key')
client = openai.OpenAI(api_key=open_api_key)

def generate_summary(cluster_texts):
    """
    Generates a structured financial news summary from a list of cluster texts
    using the OpenAI API.
    """
    prompt = f"""\
You are a financial journalist summarizing how investors and market participants are reacting to recent events, based on real-time public commentary from social media. Your task is to distill these reactions into a clear, structured summary that reflects what people are saying, how they feel about developments, and what concerns or expectations they express.
Use the following exact format:
[Number]. [Concise Headline Title in Title Case]
• [Bullet 1: Describe the financial development or market move being reacted to, including specific tickers, companies, statistics, or economic reports mentioned in posts]
• [Bullet 2: Summarize the broader context or reasoning being discussed — corporate earnings, Fed policy, geopolitical events, regulatory changes, etc.]
• [Bullet 3: Highlight how people are interpreting or connecting this to sector-wide or global trends, especially if they discuss spillover effects or broader themes]
• [Bullet 4: Convey the forward-looking sentiment — what investors are worried about, expecting, positioning for, or still debating]
Tone & Style Guidelines:
This is not an event summary — it is a sentiment digest. Focus on how people are reacting and what themes dominate discussion.
Do not quote individual users. Instead, distill the overall tone and reasoning reflected across many posts.
Use clear, AP-style sentences. Remain neutral and professional. Avoid speculation or emotional language.
Emphasize specific names, tickers (e.g., $TSLA, $SPY), locations, or statistics only if they appear in the posts.
Never mention Reddit, social media, models, data, or analysis methods. Present the insights as a clean synthesis of market commentary.
Examples:
“Investors expressed concern over $AAPL’s 7% revenue growth, with many noting slowing iPhone demand in China despite strong U.S. sales.”
“Discussions reflected renewed inflation anxiety after Powell described price stability as ‘far from achieved,’ prompting debate over further hikes.”
“Many participants pointed to rising oil prices as a bullish driver for $XLE and energy stocks broadly, citing supply risks in the Middle East.”
“Posts showed mixed sentiment ahead of the CPI report, with some expecting a downside surprise while others warned of persistent services inflation.”
Input excerpts:
{" ".join(cluster_texts)}
Generate the summary in the specified format:"""

    completion = client.chat.completions.create(
        model="gpt-4o", # Using gpt-4o for high-quality summaries
        messages=[
            {"role": "system", "content": "You are a senior editor at a major news network."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

def generate_header(cluster_texts):
    """
    Generates a concise financial news headline from a list of cluster texts
    using the OpenAI API.
    """
    prompt = f"""\
You are a financial news editor summarizing investor sentiment and market reactions based on social media commentary. Your task is to write a single professional news headline that reflects how investors and market participants are reacting to a key development, based solely on the tone and content of the provided posts.
Your headline must follow these exact rules:
Write the headline in Title Case.
Use a strong, active verb that reflects the development or sentiment discussed in the posts.
Include specific companies, indexes, institutions, policymakers, sectors, or locations only if they appear in the posts.
Ensure the headline is directly grounded in the actual reactions, events, or discussions in the posts — do not assume or invent facts.
Match the tone and style of major financial outlets like Bloomberg, Reuters, or AP — maintain neutrality, clarity, and conciseness.
Never reference Reddit, clustering, datasets, AI, or any data sources.
Focus on the most financially material development mentioned in the posts.
Reflect the mood and interpretation of public investor commentary — what people are saying, feeling, or debating.
Examples of Good Headlines:
“Tech Stocks Slide as Investors Cite Concerns Over Prolonged Rate Hikes”
“$NVDA Gains After Posts Highlight Strong Demand for AI Chips”
“Investor Discussions Focus on Powell’s Inflation Remarks Ahead of Fed Decision”
“Posts Flag Growing Recession Worries as Jobless Claims Rise Unexpectedly”
Focus on what the public discussion reveals — how investors are framing the issue, what they emphasize, and what tone dominates.
Excerpts: {" ".join(cluster_texts)}
Headline:"""

    completion = client.chat.completions.create(
        model="gpt-4o", # Using gpt-4o for high-quality headlines
        messages=[
            {"role": "system", "content": "You are a newspaper headline writer."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

def naive_count_proper_nouns(texts_list):
    """
    A naive approach to detect words that begin with a capital letter
    and continue with lowercase letters, used as a proxy for identifying proper nouns.
    """
    pattern = re.compile(r'\b[A-Z][a-z]+\b')
    count = 0
    for text in texts_list:
        matches = pattern.findall(text)
        count += len(matches)
    return count

def get_word_frequencies(texts_list, stopwords=None):
    """
    Calculates word frequencies from a list of texts, excluding common stopwords.
    """
    if stopwords is None:
        stopwords = {"the", "and", "this", "that", "with", "from", "for", "was", "were", "are"}
    all_text = " ".join(texts_list).lower()
    words = re.findall(r'\w+', all_text)
    words = [w for w in words if w not in stopwords]
    return Counter(words)

# --- Identify Top Clusters (ignoring noise) ---
# Calculate a 'proper noun' count for each cluster (excluding noise cluster -1)
# This heuristic helps rank clusters by their potential financial relevance
proper_counts = {
    cid: naive_count_proper_nouns(txts)
    for cid, txts in clusters.items()
    if cid != -1
}

def summarize_clusters(proper_counts, clusters):
    """
    Generates news briefings for the top N clusters based on proper noun counts.
    Processes summarization and header generation sequentially for each cluster.
    """
    if not proper_counts:
        return "No valid clusters found."

    # Sort clusters by proper noun count in descending order
    top_clusters_sorted = sorted(proper_counts, key=proper_counts.get, reverse=True)
    
    # Process only the top 3 clusters
    top_clusters_to_summarize = top_clusters_sorted[:3] # <--- Changed to top 3

    output = "### Top News Briefings\n\n"

    for idx, cid in enumerate(top_clusters_to_summarize, 1): # Iterate only through the top 3
        cluster_texts = clusters[cid]
        # Generate header and summary sequentially for each cluster
        header = generate_header(cluster_texts)
        summary = generate_summary(cluster_texts)
        
        output += f"#### {idx}. {header}\n\n"
        output += f"{summary}\n\n"
        output += "---\n\n" # Separator between clusters

    return output

# --- New functions for ticker analysis (from user's copied code) ---
def get_ticker_data(ticker):
    """Send request to ticker article scorer API"""
    base_url = "https://tickerarticlescorer-3.onrender.com"
    try:
        url = f"{base_url}/api/sentiment"
        response = requests.get(url, params={"ticker": ticker})
        if response.status_code == 200:
            try:
                return response.json()
            except json.JSONDecodeError:
                return {"error": "Response is not valid JSON"}
        else:
            return {"error": f"Request failed with status code: {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request error: {str(e)}"}

def get_article_text(url):
    """Fetch and extract text from article URL"""
    try:
        headers_list = [
            {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Referer': 'https://www.google.com/'
            }
        ]
        response = None
        for headers in headers_list:
            try:
                response = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
                if response.status_code == 200:
                    break
            except:
                continue
        if not response or response.status_code != 200:
            return "Failed to fetch article - Site may be blocking requests"
        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Try common article selectors
        article_selectors = [
            'article',
            '[class*="article"]',
            '[class*="content"]',
            'main',
            '.entry-content'
        ]

        article_text = ""
        for selector in article_selectors:
            article_elem = soup.select_one(selector)
            if article_elem:
                text = article_elem.get_text(separator='\n', strip=True)
                if len(text) > 200:
                    article_text = text
                    break
        if not article_text:
            paragraphs = soup.find_all('p')
            article_text = '\n'.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50])

        # Clean up the text
        lines = article_text.split('\n')
        cleaned_lines = [line.strip() for line in lines if line.strip() and len(line.strip()) > 20]

        return '\n'.join(cleaned_lines) # Removed truncation to first 10 lines
    except Exception as e:
        return f"Error fetching article: {str(e)}"

def analyze_ticker(ticker, fetch_content=False):
    """Main function to analyze ticker sentiment and optionally fetch full article content."""
    if not ticker or not ticker.strip():
        return "Please enter a valid ticker symbol", None

    ticker = ticker.strip().upper()

    # Get sentiment data
    result = get_ticker_data(ticker)

    if isinstance(result, dict) and "error" in result:
        return f"Error: {result['error']}", None

    if not result:
        return "No data found for this ticker", None

    # Format output
    output_text = f"# 📈 Sentiment Analysis for {ticker}\n\n"

    # Create summary table data
    table_data = []

    for i, entry in enumerate(result, 1):
        sentiment_score = entry.get('sentiment', 0)
        timestamp = entry.get('timestamp', '')

        # Format timestamp
        try:
            dt = datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00')) # Use datetime.datetime
            formatted_time = dt.strftime('%Y-%m-%d %H:%M')
        except:
            formatted_time = timestamp

        output_text += f"## Entry {i}: {entry['ticker']}\n"
        output_text += f"**Sentiment Score:** {sentiment_score:.2f}%\n"
        output_text += f"**Timestamp:** {formatted_time}\n\n"

        # Extract article links
        article_blurb = entry.get('article_blurb', '')
        links = re.findall(r"<a href='([^']+)'[^>]*>([^<]+)</a>", article_blurb)

        if links:
            output_text += "### 📰 Articles:\n"
            for j, (url, title) in enumerate(links, 1):
                output_text += f"{j}. **{title}**\n"
                output_text += f"   🔗 [Read Article]({url})\n"

                # Add to table data
                table_data.append({
                    "Ticker": entry['ticker'],
                    "Sentiment": f"{sentiment_score:.2f}%",
                    "Article": title[:50] + "..." if len(title) > 50 else title,
                    "URL": url,
                    "Time": formatted_time
                })

                if fetch_content:
                    output_text += f"   📄 **Full Article Content:**\n" # Updated label
                    article_text = get_article_text(url)
                    if "Failed to fetch" in article_text or "Error fetching" in article_text:
                        output_text += f"   ❌ {article_text}\n"
                    else:
                        output_text += f"```\n{article_text}\n```\n" # Output full article in a code block
                output_text += "\n"
        output_text += "---\n\n"

    # Create DataFrame for table view
    df = pd.DataFrame(table_data) if table_data else None

    return output_text, df


def summarize_clusters_wrapper():
    """
    Wrapper function to initiate the cluster summarization process.
    It relies on `proper_counts` and `clusters` being populated by the initial script execution.
    """
    return summarize_clusters(proper_counts, clusters)

# --- Gradio Interface ---
with gr.Blocks(title="📈 Ticker Sentiment Analyzer", theme=gr.themes.Soft()) as demo: # Changed title as this block is for ticker analysis
    gr.Markdown("# 📈 Ticker Sentiment Analyzer") # Changed heading
    gr.Markdown("Get real-time sentiment analysis for stock tickers from financial news articles.") # Updated description

    with gr.Row():
        with gr.Column(scale=2):
            ticker_input = gr.Textbox(
                label="🎯 Enter Ticker Symbol",
                placeholder="e.g., AAPL, TSLA, NVDA",
                value=""
            )
        with gr.Column(scale=1):
            fetch_content = gr.Checkbox(
                label="📄 Try to fetch article content",
                value=False,
                info="May be slow due to site blocking"
            )
        analyze_btn = gr.Button("🔍 Analyze Sentiment", variant="primary", size="lg")

    with gr.Row():
        with gr.Column():
            output_markdown = gr.Markdown(label="📊 Analysis Results")
        with gr.Column():
            output_table = gr.Dataframe(
                label="📋 Summary Table",
                headers=["Ticker", "Sentiment", "Article", "URL", "Time"],
                interactive=False
            )

    # Examples
    gr.Examples(
        examples=[
            ["AAPL", False],
            ["TSLA", False],
            ["NVDA", False],
            ["SPY", False],
            ["MSFT", False]
        ],
        inputs=[ticker_input, fetch_content],
        label="💡 Try these popular tickers"
    )

    # Event handlers
    analyze_btn.click(
        fn=analyze_ticker,
        inputs=[ticker_input, fetch_content],
        outputs=[output_markdown, output_table],
        show_progress=True
    )

    # Allow Enter key to trigger analysis
    ticker_input.submit(
        fn=analyze_ticker,
        inputs=[ticker_input, fetch_content],
        outputs=[output_markdown, output_table],
        show_progress=True
    )

# Footer
with demo:
    gr.Markdown("""
    ---
    💡 **Tips:**
    - Enter any stock ticker symbol (AAPL, TSLA, etc.)
    - Sentiment scores range from 0-100% (higher = more positive)
    - Article fetching may fail due to site restrictions
    - Click article links to read full content
    """)

# To run the Gradio app:
# demo.launch()

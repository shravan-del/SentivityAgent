import os, re, datetime
import numpy as np, spacy, hdbscan, praw, openai
from collections import Counter
import datetime
import pandas as pd
import joblib
import scipy.sparse as sp
import gradio as gr

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

def summarize_clusters_wrapper():
    """
    Wrapper function to initiate the cluster summarization process.
    It relies on `proper_counts` and `clusters` being populated by the initial script execution.
    """
    return summarize_clusters(proper_counts, clusters)

# --- Gradio Interface ---
with gr.Blocks(title="Hive News Headline Generator") as demo:
    gr.Markdown("## Top News Briefings")
    gr.Markdown("Summarizes top Reddit sentiment clusters into professional bullet-point news briefs.")
    
    generate_summary_btn = gr.Button("Generate Latest News Briefs")
    output_markdown = gr.Markdown(label="Summary Output")
    
    generate_summary_btn.click(
        fn=summarize_clusters_wrapper,
        inputs=[],
        outputs=output_markdown,
        show_progress=True # This shows a loading spinner
    )

# To run the Gradio app:
# demo.launch()

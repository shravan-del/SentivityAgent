import os
import gradio as gr
import praw
import pandas as pd
import re
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import defaultdict
import joblib
import scipy.sparse as sp
import requests
import random
import schedule
import time
import os
from datetime import datetime
from huggingface_hub import HfApi, upload_file
import tempfile

author_post_map = {}
last_analysis_scores = []

# Load classifier and vectorizer
try:
    classifier = joblib.load('AutoClassifier.pkl')
    vectorizer = joblib.load("AutoVectorizer.pkl")
    FILTERING_ENABLED = True
except:
    print("Warning: Could not load AutoClassifier.pkl or AutoVectorizer.pkl - filtering disabled")
    FILTERING_ENABLED = False

# Gets 19 relevant overlapping subreddits based on the user provided subreddit for a total of 20
def get_top_overlapping_subreddits(base_subreddit: str, top_n: int = 19) -> list:
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0',
        'Referer': f'https://subredditstats.com/subreddit-user-overlaps/{base_subreddit}',
        'Accept': 'application/json',
    })
    try:
        global_hist_url = f"https://subredditstats.com/api/globalSubredditsIdHist?v={random.random()}"
        global_response = session.get(global_hist_url, timeout=30)
        global_response.raise_for_status()
        global_hist = global_response.json()
        global_total = sum(global_hist.values())
        global_dist = {k: v / global_total for k, v in global_hist.items()}

        subreddit_hist_url = f"https://subredditstats.com/api/subredditNameToSubredditsHist?subredditName={base_subreddit}&v={random.random()}"
        subreddit_response = session.get(subreddit_hist_url, timeout=30)
        subreddit_response.raise_for_status()
        subreddit_hist = subreddit_response.json()
        subreddit_total = sum(subreddit_hist.values())
        subreddit_dist = {k: v / subreddit_total for k, v in subreddit_hist.items()}

        multipliers = {}
        for sid, prob in subreddit_dist.items():
            if sid in global_dist and global_dist[sid] >= 0.0001:
                multipliers[sid] = prob / global_dist[sid]

        if not multipliers:
            return []

        subreddit_ids = list(multipliers.keys())
        names_response = session.post(
            "https://subredditstats.com/api/specificSubredditIdsToNames",
            json={"subredditIds": subreddit_ids},
            headers={"Content-Type": "application/json"}
        )
        names_response.raise_for_status()
        subreddit_names = names_response.json()

        overlaps = []
        for i, (sid, score) in enumerate(multipliers.items()):
            if i < len(subreddit_names):
                overlaps.append((subreddit_names[i], round(score, 3)))

        overlaps.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in overlaps[:top_n]]

    except Exception as e:
        print(f"Overlap scrape error: {e}")
        return []

def generate_hashtags(posts, subreddits):
    import re
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction import text

    # Combine all post titles and content into one text block
    documents = [preprocess_text(p["title"]) + " " + preprocess_text(p["content"]) for p in posts]

    base_stopwords = text.ENGLISH_STOP_WORDS
    custom_stopwords = base_stopwords.union({
        'trump', 'biden', 'like', 'just', 'know', 'think', 'thing', 'things', 'people', 'said', 'also',
        'would', 'could', 'should', 'still', 'even', 'one', 'get', 'going', 'see', 'say', 'make', 'made',
        'want', 'need', 'much', 'many', 'really', 'got', 'look', 'take', 'though', 'well', 'without',
        'every', 'around', 'another', 'others', 'done', 'being', 'next', 'used', 'new'
    })

    stopword_list = list(custom_stopwords)

    # Vectorize with TF-IDF
    vectorizer = TfidfVectorizer(
        stop_words=stopword_list,
        token_pattern=r'\b[a-z]{4,}\b',
        max_df=0.6,
        max_features=100
    )
    tfidf_matrix = vectorizer.fit_transform(documents)

    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).A1
    word_scores = list(zip(feature_names, tfidf_scores))

    top_keywords = sorted(word_scores, key=lambda x: x[1], reverse=True)[:10]
    top_tags = [f"#{word}" for word, _ in top_keywords]

    base_tags = [f"#{s.lower()}" for s in subreddits if s.isalnum()]

    hashtags = list(dict.fromkeys(base_tags + top_tags))

    return hashtags

def get_youtube_comments_for_hashtags(hashtags, max_videos=3, comments_per_video=5):
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        print("Missing YOUTUBE_API_KEY environment variable.")
        return "❌ No YouTube API Key set.", ""

    collected = []
    seen_video_ids = set()

    for tag in hashtags:
        query = tag.lstrip("#")
        search_url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "q": f"#{query}",
            "type": "video",
            "maxResults": max_videos,
            "key": api_key
        }
        r = requests.get(search_url, params=params)
        results = r.json().get("items", [])
        for item in results:

            if item["id"].get("kind") != "youtube#video" or "videoId" not in item["id"]:
                continue

            video_id = item["id"]["videoId"]
            if video_id in seen_video_ids:
                continue
            seen_video_ids.add(video_id)
            title = item["snippet"]["title"]
            published = item["snippet"]["publishedAt"]
            link = f"https://www.youtube.com/watch?v={video_id}"

            # Fetch top comments for this video
            comment_url = "https://www.googleapis.com/youtube/v3/commentThreads"
            comment_params = {
                "part": "snippet",
                "videoId": video_id,
                "maxResults": comments_per_video,
                "textFormat": "plainText",
                "key": api_key
            }
            comment_res = requests.get(comment_url, params=comment_params)
            comments_data = comment_res.json().get("items", [])

            comment_texts = [
                c["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                for c in comments_data
            ]

            collected.append({
                "title": title,
                "url": link,
                "published": published,
                "comments": comment_texts
            })

    if not collected:
        return "No relevant YouTube videos found.", ""

    md = "### YouTube Sample Comments\n"
    for vid in collected:
        md += f"\n**[{vid['title']}]({vid['url']})**  \n📅 Published: {vid['published']}\n"
        for comment in vid["comments"]:
            md += f"- {comment}\n"
    return "", md

# Formatting
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z0-9\s.,!?]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def fetch_reddit_data(subreddit_name, time_period="Past 3 Months", limit=500):
    try:
        reddit = praw.Reddit(
            client_id=os.getenv('client_id'),
            client_secret=os.getenv('client_secret'),
            user_agent='MyAPI/0.0.1',
            check_for_async = False
        )
        period_map = {
            "Past Day": (1, "day"),
            "Past Week": (7, "week"),
            "Past Month": (30, "month"),
            "Past 3 Months": (90, "month"),
            "Past 6 Months": (180, "year"),
            "Past Year": (365, "all")
        }
        days, time_filter = period_map.get(time_period, (30, "month"))
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        cutoff_timestamp = cutoff_date.timestamp()

        posts = []
        for post in reddit.subreddit(subreddit_name).top(time_filter=time_filter, limit=limit):
            if post.created_utc >= cutoff_timestamp:
                posts.append({
                    "post_id": post.id,
                    "title": post.title,
                    "content": post.selftext,
                    "author": post.author.name if post.author else "[deleted]",
                    "created_utc": post.created_utc,
                    "score": post.score,
                    "num_comments": post.num_comments,
                    "upvote_ratio": post.upvote_ratio,
                    "url": post.url,
                    "subreddit": subreddit_name
                })
        if FILTERING_ENABLED:
            try:
                texts = [p["title"] for p in posts]
                X = vectorizer.transform(texts)
                if X.shape[1] < 5000:
                    X = sp.hstack([X, sp.csr_matrix((X.shape[0], 5000 - X.shape[1]))])
                predictions = classifier.predict(X)
                return [p for p, pred in zip(posts, predictions) if pred == 1]
            except:
                return posts
        return posts
    except Exception as e:
        print(f"Reddit fetch error: {e}")
        return []

def group_by_author(posts):
    data = defaultdict(lambda: {
        "posts": [], "total_comments": 0,
        "upvote_ratios": [], "titles": [], "contents": [], "timestamps": []
    })
    for p in posts:
        author = p["author"]
        if author == "[deleted]": continue
        g = data[author]
        g["posts"].append(p)
        g["total_comments"] += p["num_comments"]
        g["upvote_ratios"].append(p["upvote_ratio"])
        g["titles"].append(p["title"])
        g["contents"].append(p["content"])
        g["timestamps"].append(p["created_utc"])
    return data

# Composite score logic, can tweak overtime
def calculate_scores(author_data):
    results = []
    max_comments = max([d["total_comments"] for d in author_data.values()] + [1])
    for author, d in author_data.items():
        text = " ".join(d["titles"] + d["contents"])
        total_words = len(text.split())
        sentiment = 0.5
        toxicity = 1 - sentiment
        neg_ratio = 1 - (sum(d["upvote_ratios"]) / len(d["upvote_ratios"])) if d["upvote_ratios"] else 0
        comment_norm = d["total_comments"] / max_comments
        composite = (toxicity * 0.4) + (comment_norm * 0.3) + (neg_ratio * 0.3)
        results.append({"Author": author, "Sentiment": sentiment, "Toxicity": toxicity,
                        "Composite": composite, "Total Comments": d["total_comments"]})
    return sorted(results, key=lambda x: x["Composite"], reverse=True)[:20]

def plot_top_authors_over_time(top_authors):
    fig, ax = plt.subplots(figsize=(10, 6))
    for author in top_authors:
        posts = sorted(author_post_map.get(author, []), key=lambda x: x["created_utc"])
        if not posts: continue
        times = [datetime.utcfromtimestamp(p["created_utc"]) for p in posts]
        scores = [p["score"] for p in posts]
        ax.plot(times, scores, marker="o", label=author)
    ax.set_title("Post Scores Over Time for Top Authors")
    ax.set_xlabel("Time")
    ax.set_ylabel("Score")
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig

def author_post_dropdown(author):
    posts = sorted(author_post_map.get(author, []), key=lambda x: x["num_comments"], reverse=True)[:5]
    composite = next((a["Composite"] for a in last_analysis_scores if a["Author"] == author), 0.5)
    if not posts:
        return f"No posts found for {author}."
    result = f"### Posts by {author}\n"
    for i, p in enumerate(posts):
        variation = 0.8 + (0.4 * (i / max(len(posts)-1, 1)))
        score = round(min(1.0, max(0.0, composite * variation)), 4)
        ts = datetime.utcfromtimestamp(p["created_utc"]).strftime('%Y-%m-%d %H:%M:%S UTC')
        link = f"https://reddit.com/comments/{p['post_id']}"
        title = p["title"][:80] + ("..." if len(p["title"]) > 80 else "")
        result += f"- **{title}**\n  - Subreddit: r/{p['subreddit']}\n  - [Link]({link})\n  - Composite Score: {score}, Comments: {p['num_comments']}, Time: {ts}\n\n"
    return result

def get_author_score_over_time(author):
    posts = sorted(author_post_map.get(author, []), key=lambda x: x["created_utc"])
    if not posts:
        return None
    times = [datetime.utcfromtimestamp(p["created_utc"]) for p in posts]
    scores = [p["score"] for p in posts]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, scores, marker="o", linestyle="-", color="blue")
    ax.set_title(f"Post Scores Over Time: {author}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Score")
    ax.grid(True)
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig

def upload_dataframe_to_hf(df, repo_id, filename="df.csv"):
    """
    Upload a DataFrame as CSV to Hugging Face Hub repository (or Space)
    
    Args:
        df: pandas DataFrame to upload
        repo_id: HuggingFace repository ID (e.g., "username/repo-name" or "username/space-name")
        filename: name of the CSV file in the repo (default: "df.csv")
    
    Returns:
        str: Success or error message
    """
    try:
        hf_token = os.getenv("hf_token")
        if not hf_token:
            return "❌ Error: HF_TOKEN environment variable not set"
        
        if df.empty:
            return "❌ Error: DataFrame is empty"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            df.to_csv(tmp_file.name, index=False)
            temp_path = tmp_file.name
        
        try:
            api = HfApi()
            api.upload_file(
                path_or_fileobj=temp_path,
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="space",  # <--- Add this line to specify it's a Space
                token=hf_token,
                commit_message=f"Update {filename} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            os.unlink(temp_path)
            
            return f"✅ Successfully uploaded {filename} to {repo_id} (Hugging Face Space)"
            
        except Exception as e:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e
            
    except Exception as e:
        return f"❌ Error uploading to HuggingFace: {str(e)}"

def analyze_and_upload(base_subreddit, time_period, repo_id):
    """
    Run analysis and upload the resulting DataFrame to HuggingFace
    """
    # Run the analysis
    df_result = analyze(base_subreddit, time_period)
    
    if df_result.empty:
        return "❌ Analysis failed - no data to upload", df_result
    
    # Upload to HuggingFace
    upload_message = upload_dataframe_to_hf(df_result, repo_id)
    
    return upload_message, df_result

def analyze(base_subreddit, time_period):
    global author_post_map, last_analysis_scores
    overlapping = get_top_overlapping_subreddits(base_subreddit)
    if not overlapping:
        return pd.DataFrame()

    subreddits_to_check = [base_subreddit] + overlapping
    all_posts = []
    for sub in subreddits_to_check:
        posts = fetch_reddit_data(sub, time_period, limit=500)
        all_posts.extend(posts)

    if not all_posts:
        return pd.DataFrame()

    df_reddit = pd.DataFrame(all_posts)

    # Select and rename columns
    df_reddit = df_reddit[['title', 'created_utc']]
    df_reddit = df_reddit.rename(columns={'title': 'text', 'created_utc': 'date_only'})

    # Convert 'date_only' to datetime and format
    df_reddit['date_only'] = pd.to_datetime(df_reddit['date_only'], unit='s').dt.strftime('%Y-%m-%d')

    return df_reddit

def run_analysis(sub, time_period):
    # Keep the existing analysis logic
    overlapping = get_top_overlapping_subreddits(sub)
    if not overlapping:
        return f"Could not retrieve overlaps for r/{sub}.", [], None

    subreddits_to_check = [sub] + overlapping
    all_posts = []
    for subreddit in subreddits_to_check:
        posts = fetch_reddit_data(subreddit, time_period, limit=500)
        all_posts.extend(posts)

    if not all_posts:
        return "No posts found.", [], None

    for p in all_posts:
        p["post_text"] = preprocess_text(p["title"] + " " + p["content"])

    author_data = group_by_author(all_posts)
    global author_post_map, last_analysis_scores
    author_post_map = {a: d["posts"] for a, d in author_data.items()}
    last_analysis_scores = calculate_scores(author_data)

    df = pd.DataFrame(last_analysis_scores)
    if df.empty:
        return "No valid authors found.", [], None

    top_authors = df["Author"].tolist()[:20]
    summary = f"""
**Base Subreddit:** r/{sub}
**Analyzed Subreddits:** {', '.join(subreddits_to_check)}
**Time Period:** {time_period}
**Filtering:** {'✅' if FILTERING_ENABLED else '❌'}
Total Posts: {len(all_posts)}, Unique Authors: {len(author_data)}
Top Users: {', '.join(top_authors)}
"""
    hashtags = generate_hashtags(all_posts, subreddits_to_check)
    
    first = top_authors[0] if top_authors else None
    posts_md = author_post_dropdown(first) if first else "No author selected."
    graph = get_author_score_over_time(first) if first else None

    updates = [gr.update(visible=False) for _ in range(20)]
    for i, a in enumerate(top_authors[:20]):
        updates[i] = gr.update(
            value=f"### **{a}**\n" + author_post_dropdown(a),
            visible=True
        )

    hashtags_md = "**Suggested Hashtags:**\n\n" + " ".join(hashtags)
    yt_error, yt_comments_md = get_youtube_comments_for_hashtags(hashtags)

    return [
        summary,
        hashtags_md,
        gr.update(choices=top_authors, value=first),
        posts_md,
        graph,
        plot_top_authors_over_time(top_authors),
        yt_comments_md
    ] + updates

# Example Gradio interface setup (you'll need to integrate this with your existing interface)
def create_gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Reddit Analysis with HuggingFace Upload")
        
        with gr.Row():
            subreddit_input = gr.Textbox(label="Subreddit", placeholder="Enter subreddit name")
            time_period = gr.Dropdown(
                choices=["Past Day", "Past Week", "Past Month", "Past 3 Months", "Past 6 Months", "Past Year"],
                value="Past Month",
                label="Time Period"
            )
            repo_id_input = gr.Textbox(label="HuggingFace Repo ID", placeholder="username/repo-name")
        
        with gr.Row():
            analyze_btn = gr.Button("Analyze", variant="primary")
            upload_btn = gr.Button("Analyze & Upload to HF", variant="secondary")
        
        upload_status = gr.Textbox(label="Upload Status", interactive=False)
        data_preview = gr.Dataframe(label="Data Preview")
        
        analyze_btn.click(
            fn=lambda sub, period: analyze(sub, period),
            inputs=[subreddit_input, time_period],
            outputs=[data_preview]
        )
        
        upload_btn.click(
            fn=analyze_and_upload,
            inputs=[subreddit_input, time_period, repo_id_input],
            outputs=[upload_status, data_preview]
        )
    
    return demo

# Uncomment to run the interface
# if __name__ == "__main__":
#     demo = create_gradio_interface()
#     demo.launch()
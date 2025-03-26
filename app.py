import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yaml
import base64
from itertools import islice
from collections import Counter
import re
from datetime import datetime
import os
import time
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO

# Import YouTube search and comment downloader
from youtube_search import YoutubeSearch
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR

# Import LangChain components for AI analysis
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="YouTube Comment Analysis Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS for table-like layout
st.markdown(
    """
<style>
/* Adjust thumbnail size to be consistent */
[data-testid="stImage"] img {
    max-height: 120px;
    width: auto;
    margin: 0 auto;
    display: block;
}

/* Add some spacing between rows */
hr {
    margin: 8px 0;
}

/* Make headings smaller */
h4 {
    margin: 0;
    padding: 0;
    font-size: 16px !important;
}

/* Adjust column widths */
[data-testid="column"] {
    padding: 5px !important;
    display: flex;
    align-items: center;
}

/* Style for sentiment labels */
.positive {
    color: #4CAF50;
    font-weight: bold;
}
.neutral {
    color: #2196F3;
    font-weight: bold;
}
.negative {
    color: #F44336;
    font-weight: bold;
}

/* Heatmap styling */
.heatmap-container {
    margin: 10px 0;
    padding: 10px;
    border-radius: 5px;
}
</style>
""",
    unsafe_allow_html=True,
)


# Load configuration
@st.cache_data
def load_config():
    try:
        with open("config.yaml", "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        # Default configuration if file not found
        return {
            "max_videos": 10,
            "max_comments": 100,
            "wordcloud_max_words": 100,
            "title_max_chars": 60,
            "stopwords": [
                "the",
                "and",
                "is",
                "to",
                "in",
                "of",
                "a",
                "for",
                "that",
                "this",
            ],
        }


config = load_config()


# Function to search YouTube videos
@st.cache_data(ttl=3600)
def search_youtube(keyword, max_results=10):
    try:
        results = YoutubeSearch(keyword, max_results=max_results).to_dict()
        return results
    except Exception as e:
        st.error(f"Error searching YouTube: {e}")
        return []


# Function to fetch comments from a YouTube video
@st.cache_data(ttl=3600)
def fetch_comments(url, max_comments=100):
    try:
        downloader = YoutubeCommentDownloader()
        comments = []
        for comment in islice(
            downloader.get_comments_from_url(url, sort_by=SORT_BY_POPULAR), max_comments
        ):
            comments.append(comment)
        return comments
    except Exception as e:
        st.error(f"Error fetching comments: {e}")
        return []


# Function to convert comments to DataFrame
def comments_to_df(comments, video_id, video_title):
    df = pd.DataFrame(comments)
    df["video_id"] = video_id
    df["video_title"] = video_title
    return df


# Function to create download link
def get_download_link(df, filename="youtube_comments.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href


# Function to create wordcloud
def create_wordcloud(text, max_words=100, stopwords=None):
    # Create default wordcloud with no custom stopwords
    wc = WordCloud(
        width=800,
        height=800,
        background_color="white",
        max_words=max_words,
        contour_width=3,
        contour_color="steelblue",
    )

    # If stopwords are provided, add them to wordcloud's default stopwords
    if stopwords and isinstance(stopwords, list):
        # Get the default stopwords
        default_stopwords = wc.stopwords if wc.stopwords else set()
        # Add our custom stopwords
        for word in stopwords:
            if isinstance(word, str):
                default_stopwords.add(word.lower())
        # Update the wordcloud's stopwords
        wc.stopwords = default_stopwords

    # Generate the wordcloud
    wordcloud = wc.generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")

    return fig


# Function to preprocess text for wordcloud
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters, numbers, and extra whitespace
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Function to determine if a video is a short based on URL suffix
def is_short(url_suffix):
    return "/shorts/" in url_suffix


# Function to truncate text to a specified length
def truncate_text(text, max_length=60):
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


# NEW: Function to analyze comment sentiment using LLM
def analyze_comment_sentiment(comment_text, llm_model="gpt-4o-mini"):
    """Analyze sentiment of a single comment using LLM."""
    llm = ChatOpenAI(temperature=0, model=llm_model)

    prompt = ChatPromptTemplate.from_template(
        """Analyze the sentiment of the following YouTube comment. 
        Provide a sentiment label (positive, negative, or neutral) and a score from -1 (very negative) to 1 (very positive).
        
        Comment: {comment}
        
        Return the result in the following format:
        Sentiment: [SENTIMENT]
        Score: [SCORE]
        Explanation: [BRIEF_EXPLANATION]
        """
    )

    chain = prompt | llm

    try:
        response = chain.invoke({"comment": comment_text})
        response_text = response.content

        sentiment = "neutral"
        score = 0.0
        explanation = ""

        for line in response_text.split("\n"):
            if line.startswith("Sentiment:"):
                sentiment = line.replace("Sentiment:", "").strip().lower()
            elif line.startswith("Score:"):
                try:
                    score = float(line.replace("Score:", "").strip())
                except ValueError:
                    score = 0.0
            elif line.startswith("Explanation:"):
                explanation = line.replace("Explanation:", "").strip()

        return {"sentiment": sentiment, "score": score, "explanation": explanation}
    except Exception as e:
        st.error(f"Error analyzing sentiment: {e}")
        return {"sentiment": "neutral", "score": 0.0, "explanation": f"Error: {str(e)}"}


# NEW: Function to analyze sentiment for a batch of comments
def batch_analyze_comments(comments_df, llm_model="gpt-4o-mini"):
    """Analyze sentiment of multiple comments using LLM."""
    results = []

    # Process comments in batches to avoid API limits
    batch_size = 10
    progress_bar = st.progress(0)

    for i in range(0, len(comments_df), batch_size):
        batch = comments_df.iloc[i : i + batch_size]

        for idx, comment in batch.iterrows():
            comment_text = comment.get("text", "")
            if not comment_text:
                results.append(
                    {
                        "sentiment": "neutral",
                        "score": 0.0,
                        "explanation": "No text content",
                    }
                )
                continue

            result = analyze_comment_sentiment(comment_text, llm_model)
            results.append(result)

        # Update progress
        progress_bar.progress(min(1.0, (i + batch_size) / len(comments_df)))

    progress_bar.empty()
    return results


# NEW: Function to extract key phrases from comments
def extract_key_phrases(comments_df, llm_model="gpt-4o-mini"):
    """Extract key positive and negative phrases from comments."""
    llm = ChatOpenAI(temperature=0, model=llm_model)

    # Combine sample of comments into a single text (limit to avoid token issues)
    sample_size = min(50, len(comments_df))
    sampled_comments = (
        comments_df.sample(sample_size)
        if len(comments_df) > sample_size
        else comments_df
    )
    all_text = " ".join(sampled_comments["text"].fillna("").astype(str).tolist())

    prompt = ChatPromptTemplate.from_template(
        """Analyze the following collection of YouTube comments and extract the most common positive and negative phrases or topics.
        
        Comments: {text}
        
        Return the result in the following format:
        Positive Phrases: [COMMA_SEPARATED_PHRASES]
        Negative Phrases: [COMMA_SEPARATED_PHRASES]
        Key Topics: [COMMA_SEPARATED_TOPICS]
        """
    )

    chain = prompt | llm

    try:
        # Get phrases from LLM
        response = chain.invoke(
            {"text": all_text[:8000]}
        )  # Limit text to avoid token limits
        response_text = response.content

        # Parse response
        positive_phrases = []
        negative_phrases = []
        key_topics = []

        for line in response_text.split("\n"):
            if line.startswith("Positive Phrases:"):
                positive_str = line.replace("Positive Phrases:", "").strip()
                positive_phrases = [p.strip() for p in positive_str.split(",")]
            elif line.startswith("Negative Phrases:"):
                negative_str = line.replace("Negative Phrases:", "").strip()
                negative_phrases = [p.strip() for p in negative_str.split(",")]
            elif line.startswith("Key Topics:"):
                topics_str = line.replace("Key Topics:", "").strip()
                key_topics = [t.strip() for t in topics_str.split(",")]

        return {
            "positive_phrases": positive_phrases,
            "negative_phrases": negative_phrases,
            "key_topics": key_topics,
        }
    except Exception as e:
        st.error(f"Error extracting phrases: {e}")
        return {"positive_phrases": [], "negative_phrases": [], "key_topics": []}


# NEW: Function to generate summary of comment sentiment
def generate_comment_summary(
    comments_df, sentiment_results, phrases, llm_model="gpt-4o-mini"
):
    """Generate a summary of the comments with sentiment analysis."""
    llm = ChatOpenAI(temperature=0, model=llm_model)

    # Calculate overall sentiment statistics
    sentiments = [result["sentiment"] for result in sentiment_results]
    sentiment_counts = Counter(sentiments)
    avg_score = (
        sum(result["score"] for result in sentiment_results) / len(sentiment_results)
        if sentiment_results
        else 0
    )

    prompt = ChatPromptTemplate.from_template(
        """Generate a detailed summary of the following YouTube comments and their sentiment analysis.
        
        Number of Comments: {num_comments}
        Positive Comments: {positive_count}
        Neutral Comments: {neutral_count}
        Negative Comments: {negative_count}
        Average Sentiment Score: {avg_score}
        
        Common Positive Phrases: {positive_phrases}
        Common Negative Phrases: {negative_phrases}
        Key Topics: {key_topics}
        
        Please provide:
        1. An overview of the sentiment distribution
        2. Insights into the main topics discussed
        3. Notable trends or patterns
        4. Any recommendations for the content creator based on the comments
        
        Keep the summary concise but informative.
        """
    )

    chain = prompt | llm

    try:
        # Get summary from LLM
        response = chain.invoke(
            {
                "num_comments": len(comments_df),
                "positive_count": sentiment_counts.get("positive", 0),
                "neutral_count": sentiment_counts.get("neutral", 0),
                "negative_count": sentiment_counts.get("negative", 0),
                "avg_score": round(avg_score, 2),
                "positive_phrases": ", ".join(phrases.get("positive_phrases", [])),
                "negative_phrases": ", ".join(phrases.get("negative_phrases", [])),
                "key_topics": ", ".join(phrases.get("key_topics", [])),
            }
        )

        return response.content
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return "Unable to generate summary due to an error."


# NEW: Function to create phrase heatmap
def create_phrase_heatmap(phrases):
    """Create heatmap visualization for positive and negative phrases."""
    if (
        not phrases
        or not phrases.get("positive_phrases")
        or not phrases.get("negative_phrases")
    ):
        return None

    positive_phrases = phrases.get("positive_phrases", [])[:10]  # Limit to top 10
    negative_phrases = phrases.get("negative_phrases", [])[:10]  # Limit to top 10

    # Create data for the heatmap
    phrases_data = []

    # Calculate sentiment intensity for visualization
    sentiment_intensity = {}
    for phrase in positive_phrases:
        sentiment_intensity[phrase] = (
            0.7 + 0.3 * np.random.random()
        )  # Random value between 0.7 and 1.0

    for phrase in negative_phrases:
        sentiment_intensity[phrase] = (
            -0.7 - 0.3 * np.random.random()
        )  # Random value between -0.7 and -1.0

    # Create a dataframe for the heatmap
    for phrase, intensity in sentiment_intensity.items():
        sentiment_type = "Positive" if intensity > 0 else "Negative"
        phrases_data.append(
            {
                "Phrase": phrase,
                "Sentiment Type": sentiment_type,
                "Intensity": abs(intensity),
            }
        )

    phrases_df = pd.DataFrame(phrases_data)

    # Create the heatmap
    fig = px.density_heatmap(
        phrases_df,
        x="Sentiment Type",
        y="Phrase",
        z="Intensity",
        title="Key Phrases by Sentiment",
        color_continuous_scale=[(0, "lightblue"), (1, "darkblue")],
    )

    return fig


# Main app
def main():
    st.title("YouTube Comment Analysis Dashboard")

    # Initialize session state variables for AI analysis
    if "sentiment_results" not in st.session_state:
        st.session_state.sentiment_results = None
    if "phrases" not in st.session_state:
        st.session_state.phrases = None
    if "summary" not in st.session_state:
        st.session_state.summary = None

    # Sidebar for search and API settings
    with st.sidebar:
        st.header("API Settings")
        openai_api_key = os.getenv("OPENAI_API_KEY", "")
        if not openai_api_key:
            openai_api_key = st.text_input("OpenAI API Key", type="password")
            if openai_api_key:
                os.environ["OPENAI_API_KEY"] = openai_api_key

        st.header("Search Settings")
        search_query = st.text_input("Enter search keyword", "")
        max_videos = st.slider("Maximum number of videos", 5, 30, config["max_videos"])
        max_comments = st.slider(
            "Maximum comments per video", 50, 500, config["max_comments"]
        )
        title_max_chars = st.slider(
            "Title max characters", 30, 100, config.get("title_max_chars", 60)
        )

        search_button = st.button("Search YouTube")

        st.markdown("---")

        # NEW: LLM Model Selection
        st.header("AI Analysis Settings")
        llm_model = st.selectbox(
            "LLM Model", ["gpt-4o-mini", "gpt-4", "gpt-4o"], key="llm_model"
        )

        st.markdown("---")
        st.markdown("### About")
        st.info(
            "This dashboard allows you to search for YouTube videos, analyze their comments, and visualize the results."
        )

    # Initialize session state variables
    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    if "selected_videos" not in st.session_state:
        st.session_state.selected_videos = []
    if "comments_df" not in st.session_state:
        st.session_state.comments_df = pd.DataFrame()
    if "last_search" not in st.session_state:
        st.session_state.last_search = ""

    # Search YouTube when button is clicked
    if search_button and search_query:
        with st.spinner(f"Searching YouTube for '{search_query}'..."):
            st.session_state.search_results = search_youtube(
                search_query, max_results=max_videos
            )
            st.session_state.last_search = search_query

    # Display search results
    if st.session_state.search_results:
        st.subheader(f"Search Results for '{st.session_state.last_search}'")

        selected_video_ids = []

        # Convert search results to a nicer format for display
        formatted_results = []

        for video in st.session_state.search_results:
            video_id = video["id"]
            title = video["title"]
            channel = video["channel"]
            views = video["views"]
            duration = video["duration"]
            video_type = "Short" if is_short(video["url_suffix"]) else "Video"

            # Create URL to the video
            base_url = "https://www.youtube.com"
            full_url = f"{base_url}{video['url_suffix']}"

            # Truncate title if needed
            display_title = truncate_text(title, title_max_chars)

            # Add to formatted results
            formatted_results.append(
                {
                    "id": video_id,
                    "title": display_title,
                    "full_title": title,
                    "thumbnail": video["thumbnails"][0],
                    "channel": channel,
                    "views": views,
                    "duration": duration,
                    "type": video_type,
                    "url": full_url,
                    "original": video,
                }
            )

        # Display videos in a table format
        for i, video in enumerate(formatted_results):
            col1, col2, col3 = st.columns([1, 3, 1])

            # Column 1: Thumbnail
            with col1:
                st.image(video["thumbnail"], use_container_width=True)

            # Column 2: Title and metadata
            with col2:
                st.markdown(f"#### {video['title']}")
                st.caption(
                    f"**Channel:** {video['channel']} | **Views:** {video['views']} | **Duration:** {video['duration']} | **Type:** {video['type']}"
                )

            # Column 3: Actions
            with col3:
                st.markdown(f"[Watch Video]({video['url']})")
                selected = st.checkbox("Analyze", key=f"select_{video['id']}")
                if selected:
                    selected_video_ids.append(video["original"])

            # Add a divider between videos
            st.markdown("---")

        # Process selected videos
        if selected_video_ids:
            st.subheader("Selected Videos for Analysis")

            # Button to fetch comments for selected videos
            if st.button("Fetch Comments for Selected Videos"):
                all_comments = []
                progress_bar = st.progress(0)

                for i, video in enumerate(selected_video_ids):
                    video_id = video["id"]
                    title = video["title"]
                    url_suffix = video["url_suffix"]

                    # Create full URL
                    base_url = "https://www.youtube.com"
                    full_url = f"{base_url}{url_suffix}"

                    with st.spinner(f"Fetching comments for '{title}'..."):
                        comments = fetch_comments(full_url, max_comments=max_comments)
                        if comments:
                            video_df = comments_to_df(comments, video_id, title)
                            all_comments.append(video_df)

                    # Update progress bar
                    progress_bar.progress((i + 1) / len(selected_video_ids))

                if all_comments:
                    # Combine all comments into one DataFrame
                    st.session_state.comments_df = pd.concat(
                        all_comments, ignore_index=True
                    )
                    st.session_state.selected_videos = selected_video_ids
                    # Reset AI analysis results when new comments are fetched
                    st.session_state.sentiment_results = None
                    st.session_state.phrases = None
                    st.session_state.summary = None
                    st.success(
                        f"Successfully fetched comments for {len(selected_video_ids)} videos!"
                    )
                else:
                    st.error("No comments were retrieved. Try different videos.")

            # Display analysis if comments are available
            if not st.session_state.comments_df.empty:
                st.subheader("Comment Analysis")

                # Create tabs for different analyses
                tab1, tab2, tab3, tab4, tab5 = st.tabs(
                    [
                        "Comments Table",
                        "Word Clouds",
                        "AI Sentiment Analysis",
                        "Engagement Metrics",
                        "AI Summary & Insights",
                    ]
                )

                with tab1:
                    # Show comments in expandable containers grouped by video
                    for video in st.session_state.selected_videos:
                        video_id = video["id"]
                        title = video["title"]

                        video_comments = st.session_state.comments_df[
                            st.session_state.comments_df["video_id"] == video_id
                        ]

                        if not video_comments.empty:
                            with st.expander(
                                f"{title} ({len(video_comments)} comments)"
                            ):
                                st.dataframe(
                                    video_comments[
                                        ["author", "text", "time", "votes", "replies"]
                                    ],
                                    use_container_width=True,
                                )

                with tab2:
                    # Generate word clouds for each video
                    for video in st.session_state.selected_videos:
                        video_id = video["id"]
                        title = video["title"]

                        video_comments = st.session_state.comments_df[
                            st.session_state.comments_df["video_id"] == video_id
                        ]

                        if not video_comments.empty:
                            with st.expander(f"Word Cloud for: {title}"):
                                # Combine all comments into a single text
                                all_text = " ".join(video_comments["text"].astype(str))
                                processed_text = preprocess_text(all_text)

                                if processed_text:
                                    wordcloud_fig = create_wordcloud(
                                        processed_text,
                                        max_words=config["wordcloud_max_words"],
                                        stopwords=config["stopwords"],
                                    )
                                    st.pyplot(wordcloud_fig)
                                else:
                                    st.info("Not enough text to generate a word cloud.")

                with tab3:
                    # NEW: Advanced AI-powered sentiment analysis
                    if not openai_api_key:
                        st.warning(
                            "Please enter your OpenAI API key in the sidebar to enable AI sentiment analysis."
                        )
                    else:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(
                                "This tab uses AI to analyze the sentiment of each comment."
                            )
                        with col2:
                            run_analysis = st.button("Run AI Analysis")

                        if (
                            run_analysis
                            or st.session_state.sentiment_results is not None
                        ):
                            if (
                                run_analysis
                                or st.session_state.sentiment_results is None
                            ):
                                with st.spinner(
                                    "Analyzing comment sentiment using AI..."
                                ):
                                    st.session_state.sentiment_results = (
                                        batch_analyze_comments(
                                            st.session_state.comments_df, llm_model
                                        )
                                    )

                                with st.spinner("Extracting key phrases..."):
                                    st.session_state.phrases = extract_key_phrases(
                                        st.session_state.comments_df, llm_model
                                    )

                            # Display sentiment metrics
                            sentiment_counts = Counter(
                                [
                                    r.get("sentiment", "neutral")
                                    for r in st.session_state.sentiment_results
                                ]
                            )
                            pos_count = sentiment_counts.get("positive", 0)
                            neu_count = sentiment_counts.get("neutral", 0)
                            neg_count = sentiment_counts.get("negative", 0)

                            scores = [
                                r.get("score", 0.0)
                                for r in st.session_state.sentiment_results
                            ]
                            avg_score = sum(scores) / len(scores) if scores else 0

                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("😊 Positive", pos_count)
                            with col2:
                                st.metric("😐 Neutral", neu_count)
                            with col3:
                                st.metric("😟 Negative", neg_count)
                            with col4:
                                st.metric("Average Score", f"{avg_score:.2f}")

                            # Create sentiment distribution visualization
                            fig1 = px.pie(
                                values=[pos_count, neu_count, neg_count],
                                names=["Positive", "Neutral", "Negative"],
                                title="Sentiment Distribution",
                                color=["Positive", "Neutral", "Negative"],
                                color_discrete_map={
                                    "Positive": "#4CAF50",
                                    "Neutral": "#2196F3",
                                    "Negative": "#F44336",
                                },
                            )

                            # Create sentiment score histogram
                            sentiment_df = pd.DataFrame(
                                {
                                    "score": scores,
                                    "sentiment": [
                                        r.get("sentiment", "neutral")
                                        for r in st.session_state.sentiment_results
                                    ],
                                }
                            )

                            fig2 = px.histogram(
                                sentiment_df,
                                x="score",
                                color="sentiment",
                                title="Sentiment Score Distribution",
                                color_discrete_map={
                                    "positive": "#4CAF50",
                                    "neutral": "#2196F3",
                                    "negative": "#F44336",
                                },
                                nbins=20,
                            )

                            col1, col2 = st.columns(2)
                            with col1:
                                st.plotly_chart(fig1, use_container_width=True)
                            with col2:
                                st.plotly_chart(fig2, use_container_width=True)

                            # Display phrase heatmap
                            phrase_fig = create_phrase_heatmap(st.session_state.phrases)
                            if phrase_fig:
                                st.subheader("Key Phrases by Sentiment")
                                st.plotly_chart(phrase_fig, use_container_width=True)

                            # Display comments with sentiment
                            st.subheader("Comments with AI Sentiment Analysis")

                            # Create a dataframe with sentiment
                            comments_with_sentiment = (
                                st.session_state.comments_df.copy()
                            )
                            comments_with_sentiment["sentiment"] = [
                                r.get("sentiment", "neutral")
                                for r in st.session_state.sentiment_results
                            ]
                            comments_with_sentiment["sentiment_score"] = [
                                r.get("score", 0.0)
                                for r in st.session_state.sentiment_results
                            ]
                            comments_with_sentiment["explanation"] = [
                                r.get("explanation", "")
                                for r in st.session_state.sentiment_results
                            ]

                            # Group by video
                            for video in st.session_state.selected_videos:
                                video_id = video["id"]
                                title = video["title"]

                                video_comments = comments_with_sentiment[
                                    comments_with_sentiment["video_id"] == video_id
                                ]

                                if not video_comments.empty:
                                    with st.expander(
                                        f"Sentiment Analysis for: {title}"
                                    ):
                                        # Show sentiment distribution for this video
                                        video_sentiments = Counter(
                                            video_comments["sentiment"]
                                        )
                                        video_fig = px.pie(
                                            values=[
                                                video_sentiments.get("positive", 0),
                                                video_sentiments.get("neutral", 0),
                                                video_sentiments.get("negative", 0),
                                            ],
                                            names=["Positive", "Neutral", "Negative"],
                                            title=f"Sentiment Distribution for {title}",
                                            color=["Positive", "Neutral", "Negative"],
                                            color_discrete_map={
                                                "Positive": "#4CAF50",
                                                "Neutral": "#2196F3",
                                                "Negative": "#F44336",
                                            },
                                        )
                                        st.plotly_chart(
                                            video_fig, use_container_width=True
                                        )

                                        # Display comments with sentiment
                                        for i, comment in video_comments.iterrows():
                                            sentiment_color = {
                                                "positive": "positive",
                                                "neutral": "neutral",
                                                "negative": "negative",
                                            }.get(comment["sentiment"], "neutral")

                                            st.markdown(
                                                f"""
                                            **{comment["author"]}** ({comment["votes"]} votes) - {comment["time"]}  
                                            {comment["text"]}
                                            
                                            <span class="{sentiment_color}">Sentiment: {comment["sentiment"].capitalize()} ({comment["sentiment_score"]:.2f})</span>  
                                            <small>{comment["explanation"]}</small>
                                            """,
                                                unsafe_allow_html=True,
                                            )
                                            st.markdown("---")

                with tab4:
                    # Engagement metrics analysis
                    for video in st.session_state.selected_videos:
                        video_id = video["id"]
                        title = video["title"]

                        video_comments = st.session_state.comments_df[
                            st.session_state.comments_df["video_id"] == video_id
                        ]

                        if not video_comments.empty:
                            with st.expander(f"Engagement Analysis for: {title}"):
                                # Convert votes column to numeric
                                video_comments["votes"] = pd.to_numeric(
                                    video_comments["votes"].fillna(0), errors="coerce"
                                )

                                # Top 10 most upvoted comments
                                top_comments = video_comments.sort_values(
                                    "votes", ascending=False
                                ).head(10)

                                st.subheader("Top 10 Most Upvoted Comments")
                                for _, comment in top_comments.iterrows():
                                    st.markdown(f"""
                                    **{comment["author"]}** ({comment["votes"]} votes) - {comment["time"]}  
                                    {comment["text"]}
                                    """)
                                    st.markdown("---")

                                # Comments over time (based on time_parsed)
                                if "time_parsed" in video_comments.columns:
                                    video_comments["date"] = pd.to_datetime(
                                        video_comments["time_parsed"], unit="s"
                                    ).dt.date
                                    comments_by_date = (
                                        video_comments.groupby("date")
                                        .size()
                                        .reset_index(name="count")
                                    )

                                    fig = px.line(
                                        comments_by_date,
                                        x="date",
                                        y="count",
                                        title="Comments Over Time",
                                        labels={
                                            "date": "Date",
                                            "count": "Number of Comments",
                                        },
                                    )
                                    st.plotly_chart(fig, use_container_width=True)

                with tab5:
                    # NEW: AI Summary & Insights
                    if not openai_api_key:
                        st.warning(
                            "Please enter your OpenAI API key in the sidebar to enable AI summary and insights."
                        )
                    elif st.session_state.sentiment_results is None:
                        st.info(
                            "Run the AI analysis in the 'AI Sentiment Analysis' tab first."
                        )
                    else:
                        # Generate summary if not already generated
                        if st.session_state.summary is None:
                            with st.spinner("Generating AI summary and insights..."):
                                st.session_state.summary = generate_comment_summary(
                                    st.session_state.comments_df,
                                    st.session_state.sentiment_results,
                                    st.session_state.phrases,
                                    llm_model,
                                )

                        # Display summary
                        st.subheader("AI-Generated Summary")
                        st.markdown(st.session_state.summary)

                        # Display key topics
                        st.subheader("Key Topics")
                        key_topics = st.session_state.phrases.get("key_topics", [])
                        if key_topics:
                            cols = st.columns(min(5, len(key_topics)))
                            for i, topic in enumerate(key_topics):
                                with cols[i % len(cols)]:
                                    st.markdown(f"🔑 **{topic}**")

                        # Display common positive and negative phrases
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Common Positive Phrases")
                            positive_phrases = st.session_state.phrases.get(
                                "positive_phrases", []
                            )
                            if positive_phrases:
                                for phrase in positive_phrases:
                                    st.markdown(f"✅ {phrase}")
                        with col2:
                            st.subheader("Common Negative Phrases")
                            negative_phrases = st.session_state.phrases.get(
                                "negative_phrases", []
                            )
                            if negative_phrases:
                                for phrase in negative_phrases:
                                    st.markdown(f"❌ {phrase}")

                # Download options for the dataframe
                st.subheader("Download Data")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(
                        get_download_link(
                            st.session_state.comments_df, "youtube_comments.csv"
                        ),
                        unsafe_allow_html=True,
                    )

                # If sentiment analysis has been run, offer to download that too
                if st.session_state.sentiment_results is not None:
                    with col2:
                        sentiment_df = st.session_state.comments_df.copy()
                        sentiment_df["sentiment"] = [
                            r.get("sentiment", "neutral")
                            for r in st.session_state.sentiment_results
                        ]
                        sentiment_df["sentiment_score"] = [
                            r.get("score", 0.0)
                            for r in st.session_state.sentiment_results
                        ]

                        st.markdown(
                            get_download_link(
                                sentiment_df, "youtube_comments_with_sentiment.csv"
                            ),
                            unsafe_allow_html=True,
                        )

    else:
        # Show instructions when no search has been performed
        st.info(
            "Enter a search keyword in the sidebar and click 'Search YouTube' to begin."
        )


if __name__ == "__main__":
    main()

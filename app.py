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


# Main app
def main():
    st.title("YouTube Comment Analysis Dashboard")

    # Sidebar for search
    with st.sidebar:
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
                    st.success(
                        f"Successfully fetched comments for {len(selected_video_ids)} videos!"
                    )
                else:
                    st.error("No comments were retrieved. Try different videos.")

            # Display analysis if comments are available
            if not st.session_state.comments_df.empty:
                st.subheader("Comment Analysis")

                # Create tabs for different analyses
                tab1, tab2, tab3, tab4 = st.tabs(
                    [
                        "Comments Table",
                        "Word Clouds",
                        "Sentiment Analysis",
                        "Engagement Metrics",
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
                    # Simple sentiment visualization based on common positive/negative words
                    # For a real implementation, you would use a sentiment analysis library
                    st.info(
                        "This is a simplified sentiment visualization. For more accurate results, integrate a sentiment analysis library."
                    )

                    positive_words = [
                        "good",
                        "great",
                        "awesome",
                        "excellent",
                        "love",
                        "amazing",
                        "best",
                        "nice",
                        "perfect",
                        "beautiful",
                    ]
                    negative_words = [
                        "bad",
                        "worst",
                        "hate",
                        "terrible",
                        "awful",
                        "poor",
                        "horrible",
                        "disappointing",
                        "waste",
                        "useless",
                    ]

                    for video in st.session_state.selected_videos:
                        video_id = video["id"]
                        title = video["title"]

                        video_comments = st.session_state.comments_df[
                            st.session_state.comments_df["video_id"] == video_id
                        ]

                        if not video_comments.empty:
                            with st.expander(f"Sentiment Analysis for: {title}"):
                                all_text = " ".join(
                                    video_comments["text"].astype(str)
                                ).lower()

                                positive_count = sum(
                                    all_text.count(word) for word in positive_words
                                )
                                negative_count = sum(
                                    all_text.count(word) for word in negative_words
                                )

                                fig = go.Figure()
                                fig.add_trace(
                                    go.Bar(
                                        x=["Positive", "Negative"],
                                        y=[positive_count, negative_count],
                                        marker_color=["#4CAF50", "#F44336"],
                                    )
                                )
                                fig.update_layout(
                                    title="Simple Sentiment Analysis",
                                    xaxis_title="Sentiment",
                                    yaxis_title="Count of Words",
                                    height=400,
                                )
                                st.plotly_chart(fig, use_container_width=True)

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

                # Download option for the dataframe
                st.subheader("Download Comments Data")
                st.markdown(
                    get_download_link(st.session_state.comments_df),
                    unsafe_allow_html=True,
                )

    else:
        # Show instructions when no search has been performed
        st.info(
            "Enter a search keyword in the sidebar and click 'Search YouTube' to begin."
        )


if __name__ == "__main__":
    main()

# YouTubeSenseAI

This Streamlit application allows you to search for YouTube videos, fetch comments, and analyze them through an interactive dashboard.

## Features

- Search YouTube videos by keyword
- View video thumbnails, titles, and metadata
- Select videos to analyze their comments
- View comments in expandable tables
- Generate word clouds for video comments
- Perform basic sentiment analysis
- Analyze engagement metrics
- Download the comment data as CSV

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:

```bash
streamlit run app.py
```

2. Enter a search keyword in the sidebar
3. Adjust the settings for maximum videos and comments
4. Click "Search YouTube" to fetch videos
5. Select videos you want to analyze by checking the "Analyze Comments" box
6. Click "Fetch Comments for Selected Videos" to retrieve comments
7. Explore the different analysis tabs
8. Download the comment data as CSV if needed

## Configuration

You can modify the `config.yaml` file to adjust settings:

- `max_videos`: Maximum number of videos to fetch in search results
- `max_comments`: Maximum number of comments to fetch per video
- `wordcloud_max_words`: Maximum number of words in word cloud visualization
- `stopwords`: List of common words to exclude from word clouds

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- Plotly
- WordCloud
- YouTube Search
- YouTube Comment Downloader

## Screenshots

![Search Results](https://via.placeholder.com/800x400?text=Search+Results)
![Comments Table](https://via.placeholder.com/800x400?text=Comments+Table)
![Word Cloud](https://via.placeholder.com/800x400?text=Word+Cloud)

## Notes

- The YouTube API has rate limits, so excessive usage may result in temporary blocks
- Comment fetching may take time for popular videos
- Sentiment analysis is simplified and may not be fully accurate
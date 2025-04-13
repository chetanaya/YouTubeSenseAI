# YouTubeSenseAI: Comment & Transcript Analysis Dashboard

YouTubeSenseAI is a Streamlit dashboard designed to analyze YouTube videos. It allows users to search for videos, fetch comments and transcripts, perform sentiment analysis, generate summaries, and interact with video content using AI.

## Features

The application is divided into two main sections accessible via the sidebar navigation:

### 1. YouTube Comment Analysis

*   **Video Search:** Search YouTube based on keywords.
*   **Video Selection:** Select multiple videos from search results for analysis.
*   **Comment Fetching:** Retrieve comments (sorted by popularity) for selected videos.
*   **Comment Display:** View comments grouped by video.
*   **Word Clouds:** Generate word clouds from comments for each video to visualize frequent terms.
*   **AI Sentiment Analysis:**
    *   Analyze the sentiment (positive, negative, neutral) and score of individual comments using an LLM (e.g., GPT-4o Mini).
    *   Visualize overall sentiment distribution (pie chart) and score distribution (histogram).
    *   Identify and display key positive and negative phrases using AI.
    *   Visualize key phrases with a heatmap.
*   **Engagement Metrics:**
    *   Display the top 10 most upvoted comments for each video.
    *   (Partially Implemented) Visualize comment activity over time.
*   **AI Summary & Insights:**
    *   Generate an overall summary of comment sentiment, key topics, and potential recommendations using an LLM.
    *   Display extracted key topics and common positive/negative phrases.
*   **Data Download:** Download fetched comments and sentiment analysis results as CSV files.

### 2. Transcript Analysis

*   **URL Input:** Analyze a specific video by pasting its YouTube URL.
*   **Transcript Fetching:**
    *   Retrieve video transcripts using the `youtube-transcript-api`.
    *   Supports selecting preferred language and automatically falls back to available languages if the preferred one isn't found.
*   **Transcript Display:**
    *   Show the full transcript text.
    *   Display a formatted version with clickable timestamps that seek the embedded video player.
*   **AI Summarization:**
    *   Generate a comprehensive summary of the transcript using an LLM.
    *   Option to provide a custom summarization prompt.
*   **AI Q&A Generation:** Automatically generate potential questions and answers based on the transcript content.
*   **Translation:** Translate the transcript or the summary into various languages using an LLM.
*   **Interactive Q&A (RAG):**
    *   Ask questions about the video content in a chat interface.
    *   Uses a Retrieval-Augmented Generation (RAG) system (built with LangChain and LangGraph) to answer questions based *only* on the transcript content.
    *   Maintains chat history per video session.
*   **Download:** Download the transcript text and the generated summary.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/chetanaya/YouTubeSenseAI.git
    cd YouTubeSenseAI
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up API Keys:**
    *   Copy the `.env.example` file to `.env` in the project root directory.
    *   Add your OpenAI API key to the `.env` file:
        ```
        OPENAI_API_KEY=your_openai_api_key_here
        ```
    *   Alternatively, you can enter the API key directly in the application's sidebar when prompted.

## Usage

Run the Streamlit application from your terminal:

```bash
streamlit run app.py
```

Navigate through the sidebar to switch between the "YouTube Comment Analysis" and "Transcript Analysis" dashboards. Follow the on-screen instructions to search for videos, fetch data, and perform analyses.

## Configuration

Basic settings can be adjusted in the `config.yaml` file:

*   `max_videos`: Default maximum number of videos to fetch in search results.
*   `max_comments`: Default maximum number of comments to fetch per video.
*   `wordcloud_max_words`: Maximum words in the generated word clouds.
*   `title_max_chars`: Maximum characters to display for video titles in the search results.
*   `stopwords`: A list of common words to exclude from word clouds.

## Dependencies

The application uses several key libraries:

*   `streamlit`: For building the web application interface.
*   `pandas`, `numpy`: For data manipulation.
*   `plotly`, `matplotlib`, `wordcloud`: For data visualization.
*   `youtube-search`, `youtube-comment-downloader`, `youtube-transcript-api`: For interacting with YouTube data.
*   `langchain`, `langchain-community`, `langchain-openai`, `openai`: For Large Language Model integration (summarization, sentiment analysis, Q&A).
*   `langgraph`: For building the RAG agent.
*   `faiss-cpu`: For vector storage in the RAG system.
*   `pyyaml`: For loading configuration.
*   `python-dotenv`: For managing environment variables (API keys).

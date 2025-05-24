import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from utils.transcript_utils import (
    extract_video_id,
    translate_text,
    create_rag_system,
)
import assemblyai as aai
import yt_dlp
import shutil  # Import shutil to check for ffmpeg
from modules.nav import Navbar
from utils.history_manager import render_video_history_widget

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="YouTube Transcript Analysis Dashboard (Using Assembly AI)",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

Navbar()


def transcribe_youtube_video(video_url: str, api_key: str) -> dict:
    """
    Transcribe a YouTube video given its URL using AssemblyAI.

    Args:
        video_url: The YouTube video URL to transcribe
        api_key: AssemblyAI API key

    Returns:
        A dictionary containing the transcript and analysis results
    """
    # Extract video ID
    video_id = extract_video_id(video_url)
    if not video_id:
        st.error("Invalid YouTube URL")
        return None

    # Configure yt-dlp options for audio extraction
    ydl_opts = {
        "format": "m4a/bestaudio/best",
        "outtmpl": "%(id)s.%(ext)s",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "m4a",
            }
        ],
    }

    try:
        # Download and extract audio
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
            # Get video ID from info dict
            info = ydl.extract_info(video_url, download=False)
            video_id = info["id"]

        # Configure AssemblyAI
        aai.settings.api_key = api_key

        # Set up AssemblyAI transcription configuration with enhanced features
        config = aai.TranscriptionConfig(
            sentiment_analysis=True,
            speaker_labels=True,
            iab_categories=True,
            summarization=True,
            summary_model=aai.SummarizationModel.informative,
            summary_type=aai.SummarizationType.bullets,
        )

        # Transcribe the downloaded audio file
        transcript = aai.Transcriber().transcribe(f"{video_id}.m4a", config)

        # Clean up the downloaded file
        if os.path.exists(f"{video_id}.m4a"):
            os.remove(f"{video_id}.m4a")

        # Prepare topic information
        topics_summary = {}
        for topic, relevance in transcript.iab_categories.summary.items():
            topics_summary[topic] = relevance * 100

        topic_segments = []
        if transcript.iab_categories and hasattr(transcript.iab_categories, "results"):
            for result in transcript.iab_categories.results:
                segment = {
                    "text": result.text,
                    "timestamp": f"{result.timestamp.start} - {result.timestamp.end}",
                    "labels": [
                        (label.label, label.relevance) for label in result.labels
                    ],
                }
                topic_segments.append(segment)

        # Prepare sentiment analysis data
        sentiment_data = []
        if transcript.sentiment_analysis:
            for result in transcript.sentiment_analysis:
                sentiment_item = {
                    "text": result.text,
                    "sentiment": result.sentiment,
                    "confidence": result.confidence,
                    "timestamp": f"{result.start} - {result.end}",
                    "speaker": result.speaker,
                }
                sentiment_data.append(sentiment_item)

        # Return comprehensive analysis results
        return {
            "text": transcript.text,
            "summary": transcript.summary if hasattr(transcript, "summary") else "",
            "topics_summary": topics_summary,
            "topic_segments": topic_segments,
            "sentiment_analysis": sentiment_data,
            "transcript_id": transcript.id,
        }
    except Exception as e:
        st.error(f"Error fetching or transcribing audio: {str(e)}")
        # Clean up any partial downloads
        if video_id and os.path.exists(f"{video_id}.m4a"):
            os.remove(f"{video_id}.m4a")
        return None


def app():
    st.title("📝 YouTube Transcript Analysis (Assembly AI Edition)")

    # Simplified CSS with only essential styles
    st.markdown(
        """
        <style>
        .transcript-container {
            max-height: 400px;
            overflow-y: auto;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 5px;
            border: 1px solid #ddd;
            font-family: monospace;
            white-space: pre-wrap;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("API Settings")
        assemblyai_api_key = os.getenv("ASSEMBLYAI_API_KEY", "")
        if not assemblyai_api_key:
            assemblyai_api_key = st.text_input("AssemblyAI API Key", type="password")
            if assemblyai_api_key:
                os.environ["ASSEMBLYAI_API_KEY"] = assemblyai_api_key
            else:
                st.warning(
                    "AssemblyAI API Key is required for transcription and analysis."
                )
        elif "ASSEMBLYAI_API_KEY" not in os.environ:
            os.environ["ASSEMBLYAI_API_KEY"] = (
                assemblyai_api_key  # Ensure it's set if provided via env
            )

        # Add ffmpeg check and info message
        st.header("System Dependencies")
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            st.warning(
                """
                **FFmpeg not found.**

                FFmpeg is required for processing audio files downloaded from YouTube.
                Please install it to ensure proper functionality.

                **Installation:**
                - **macOS (using Homebrew):** `brew install ffmpeg`
                - **Debian/Ubuntu:** `sudo apt update && sudo apt install ffmpeg`
                - **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to your system's PATH.

                After installation, please restart the Streamlit app.
                """
            )
        else:
            st.success(f"✅ FFmpeg found at: `{ffmpeg_path}`")

        st.header("LLM Model Settings")
        llm_model = st.selectbox(
            "LLM Model for Analysis",
            ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4", "gpt-4o"],
            index=0,
        )
        # Removed transcript language selection

        # Add video history to sidebar
        render_video_history_widget(analysis_method="assemblyai")

    # Check if URL is provided in query parameters
    if "url" in st.query_params:
        video_url = st.query_params["url"]
        # Remove the URL from query params to avoid loops
        st.query_params.clear()
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(
                '<p class="big-font">Enter YouTube Video URL</p>',
                unsafe_allow_html=True,
            )
            video_url = st.text_input(
                "Paste the YouTube video URL here",
                placeholder="https://www.youtube.com/watch?v=...",
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

    if video_url:
        video_id = extract_video_id(video_url)

        if not video_id:
            st.error("Invalid YouTube URL. Please enter a valid YouTube video URL.")
        else:
            st.video(f"https://www.youtube.com/watch?v={video_id}")

            with st.expander("Advanced Summarization Options", expanded=False):
                custom_prompt = st.text_area(
                    "Custom Summarization Prompt (leave empty to use default)",
                    height=100,
                    help="Customize how the transcript is summarized. Use {transcript} as a placeholder for the transcript text.",
                )
                if not custom_prompt:
                    custom_prompt = None

            fetch_btn = st.button(
                "🎤 Fetch Audio & Analyze Transcript (Assembly AI)",
                type="primary",
                use_container_width=True,
            )

            # Initialize session state keys
            if "transcript_text" not in st.session_state:
                st.session_state.transcript_text = ""
            if "summary" not in st.session_state:
                st.session_state.summary = ""
            if "qa_pairs" not in st.session_state:
                st.session_state.qa_pairs = ""
            if "current_video_id" not in st.session_state:
                st.session_state.current_video_id = None
            if "chat_histories" not in st.session_state:
                st.session_state.chat_histories = {}
            if "rag_chains" not in st.session_state:
                st.session_state.rag_chains = {}

            if fetch_btn:
                if not os.environ.get("ASSEMBLYAI_API_KEY"):
                    st.error(
                        "Please enter your AssemblyAI API Key in the sidebar first."
                    )
                    st.stop()

                if st.session_state.current_video_id != video_id:
                    # Reset state for new video
                    st.session_state.current_video_id = video_id
                    st.session_state.transcript_text = ""
                    st.session_state.summary = ""
                    st.session_state.qa_pairs = ""
                    st.session_state.chat_histories[video_id] = []
                    st.session_state.rag_chains[video_id] = None

                with st.status(
                    "Processing video with AssemblyAI...", expanded=True
                ) as status:
                    st.write(
                        "Downloading audio and transcribing (this may take a while)..."
                    )
                    transcript_result = transcribe_youtube_video(
                        video_url, os.environ.get("ASSEMBLYAI_API_KEY")
                    )

                    if transcript_result:
                        st.session_state.transcript_text = transcript_result["text"]
                        st.session_state.summary = transcript_result["summary"]
                        st.session_state.topics_summary = transcript_result[
                            "topics_summary"
                        ]
                        st.session_state.topic_segments = transcript_result[
                            "topic_segments"
                        ]
                        st.session_state.sentiment_analysis = transcript_result[
                            "sentiment_analysis"
                        ]
                        st.session_state.transcript_id = transcript_result[
                            "transcript_id"
                        ]

                        st.write("✅ Transcription complete!")
                        time.sleep(0.5)

                        st.write("Setting up question answering system...")
                        rag_chain_instance = create_rag_system(
                            st.session_state.transcript_text, llm_model
                        )
                        if rag_chain_instance:
                            st.session_state.rag_chains[video_id] = rag_chain_instance
                            st.write("✅ RAG system ready.")
                        else:
                            st.warning("Could not initialize RAG system.")
                            st.session_state.rag_chains[video_id] = None

                        if video_id not in st.session_state.chat_histories:
                            st.session_state.chat_histories[video_id] = []

                        # Add video to history
                        if "history_manager" not in st.session_state:
                            from utils.history_manager import VideoHistoryManager

                            st.session_state.history_manager = VideoHistoryManager()

                        # Get video title from YouTube if possible (simplified here)
                        video_title = f"Assembly AI Analysis: {video_id}"

                        # Add to history
                        st.session_state.history_manager.add_video(
                            video_id=video_id,
                            video_url=video_url,
                            analysis_method="assemblyai",
                            title=video_title,
                            metadata={
                                "transcript_length": len(
                                    st.session_state.transcript_text
                                ),
                                "transcript_id": st.session_state.transcript_id
                                if hasattr(st.session_state, "transcript_id")
                                else None,
                                "summary_length": len(st.session_state.summary)
                                if hasattr(st.session_state, "summary")
                                else 0,
                            },
                        )

                        status.update(
                            label="✅ Analysis complete!",
                            state="complete",
                            expanded=False,
                        )
                    else:
                        status.update(label="❌ Transcription failed", state="error")
                        st.error("Could not get transcript using AssemblyAI.")
                        # Clear potentially stale data
                        st.session_state.transcript_text = ""
                        st.session_state.summary = ""
                        st.session_state.qa_pairs = ""
                        if video_id in st.session_state.rag_chains:
                            st.session_state.rag_chains[video_id] = None
                        st.session_state.current_video_id = (
                            None  # Reset current video ID on failure
                        )

            current_video_id = st.session_state.get("current_video_id")
            # Check if the current video ID matches and transcript text exists
            if (
                current_video_id
                and current_video_id == video_id
                and st.session_state.transcript_text
            ):
                tabs = st.tabs(
                    [
                        "📊 Summary",
                        "💡 Topics & Insights",
                        "😀 Sentiment Analysis",
                        "📜 Transcript (Raw)",
                        "🌐 Translations",
                    ]
                )

                with tabs[0]:  # Summary Tab
                    if st.session_state.summary:
                        with st.container():
                            summary_container = st.expander(
                                "AssemblyAI Summary", expanded=True
                            )
                            with summary_container:
                                st.markdown(st.session_state.summary)

                        st.download_button(
                            label="📥 Download Summary",
                            data=st.session_state.summary,
                            file_name=f"summary_assemblyai_{video_id}.md",
                            mime="text/markdown",
                        )
                    else:
                        st.info("Summary will appear here once generated.")

                with tabs[1]:  # Topics & Insights Tab
                    st.subheader("💡 Topic Analysis")
                    if (
                        hasattr(st.session_state, "topics_summary")
                        and st.session_state.topics_summary
                    ):
                        # Create two columns for better organization
                        topic_col1, topic_col2 = st.columns([3, 2])

                        with topic_col1:
                            # Display overall topic relevance
                            st.markdown("#### Overall Topic Relevance")
                            # Sort topics by relevance percentage in descending order
                            sorted_topics = sorted(
                                st.session_state.topics_summary.items(),
                                key=lambda x: x[1],
                                reverse=True,
                            )

                            # Only show top 10 topics by default
                            top_topics = sorted_topics[:10]
                            for topic, relevance in top_topics:
                                st.progress(
                                    min(relevance / 100, 1.0),
                                    text=f"{topic}: {relevance:.1f}%",
                                )

                            # Show remaining topics in an expander if there are more than 10
                            if len(sorted_topics) > 10:
                                with st.expander("View more topics"):
                                    for topic, relevance in sorted_topics[10:]:
                                        st.progress(
                                            min(relevance / 100, 1.0),
                                            text=f"{topic}: {relevance:.1f}%",
                                        )

                        with topic_col2:
                            # Create a simple pie chart for visual representation of top 5 topics
                            if len(sorted_topics) > 0:
                                st.markdown("#### Top 5 Topics Distribution")
                                top5_topics = sorted_topics[:5]

                                # Create data for pie chart
                                topic_labels = [t[0] for t in top5_topics]
                                topic_values = [t[1] for t in top5_topics]

                                # Using Streamlit's native chart
                                topic_chart_data = {
                                    label: value
                                    for label, value in zip(topic_labels, topic_values)
                                }
                                st.bar_chart(topic_chart_data)

                        # Display topic segments in a more compact way
                        if (
                            hasattr(st.session_state, "topic_segments")
                            and st.session_state.topic_segments
                        ):
                            st.markdown("#### 🔍 Detailed Topic Segments")

                            # Group segments by similar topics
                            with st.expander(
                                "View detailed topic segments", expanded=False
                            ):
                                for i, segment in enumerate(
                                    st.session_state.topic_segments
                                ):
                                    # Create a more compact segment display
                                    col1, col2 = st.columns([1, 3])
                                    with col1:
                                        st.markdown(
                                            f"**Segment {i + 1}**  \n{segment['timestamp']}"
                                        )
                                    with col2:
                                        # Get top 3 labels for compact display
                                        top_labels = sorted(
                                            segment["labels"],
                                            key=lambda x: x[1],
                                            reverse=True,
                                        )[:3]
                                        labels_str = ", ".join(
                                            [
                                                f"{label} ({relevance * 100:.0f}%)"
                                                for label, relevance in top_labels
                                            ]
                                        )
                                        st.markdown(f"**Key Topics:** {labels_str}")
                                        st.markdown(f"**Text:** {segment['text']}")

                                    if i < len(st.session_state.topic_segments) - 1:
                                        st.divider()
                    else:
                        st.info(
                            "Topic analysis will appear here once the transcript is analyzed."
                        )

                with tabs[2]:  # Sentiment Analysis Tab
                    st.subheader("😀 Sentiment Analysis")
                    if (
                        hasattr(st.session_state, "sentiment_analysis")
                        and st.session_state.sentiment_analysis
                    ):
                        # Create two columns for summary and details
                        sent_col1, sent_col2 = st.columns([1, 1])

                        with sent_col1:
                            # Count sentiments for the visualization
                            sentiment_counts = {
                                "POSITIVE": 0,
                                "NEUTRAL": 0,
                                "NEGATIVE": 0,
                            }
                            for item in st.session_state.sentiment_analysis:
                                sentiment = item.get("sentiment", "NEUTRAL")
                                sentiment_counts[sentiment] += 1

                            total_segments = sum(sentiment_counts.values())
                            if total_segments > 0:
                                st.markdown("#### Overall Sentiment")

                                # Use emoticons for better visual display
                                st.markdown(
                                    f"""
                                    <div style="display: flex; justify-content: space-between; text-align: center; margin-bottom: 20px;">
                                        <div>
                                            <div style="font-size: 2em;">😀</div>
                                            <div><b>Positive</b></div>
                                            <div>{sentiment_counts["POSITIVE"]} segments</div>
                                            <div>({sentiment_counts["POSITIVE"] / total_segments * 100:.1f}%)</div>
                                        </div>
                                        <div>
                                            <div style="font-size: 2em;">😐</div>
                                            <div><b>Neutral</b></div>
                                            <div>{sentiment_counts["NEUTRAL"]} segments</div>
                                            <div>({sentiment_counts["NEUTRAL"] / total_segments * 100:.1f}%)</div>
                                        </div>
                                        <div>
                                            <div style="font-size: 2em;">😟</div>
                                            <div><b>Negative</b></div>
                                            <div>{sentiment_counts["NEGATIVE"]} segments</div>
                                            <div>({sentiment_counts["NEGATIVE"] / total_segments * 100:.1f}%)</div>
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                                # Display bar chart
                                st.bar_chart(sentiment_counts)

                        with sent_col2:
                            # Create a timeline representation
                            st.markdown("#### Sentiment Flow")
                            st.markdown("Sentiment changes throughout the video:")

                            # Create a simplified timeline visualization
                            timeline_data = []
                            for item in st.session_state.sentiment_analysis[
                                :30
                            ]:  # Limit to first 30 for performance
                                sentiment = item.get("sentiment", "NEUTRAL")
                                # Convert sentiment to numeric value for visualization
                                value = (
                                    1
                                    if sentiment == "POSITIVE"
                                    else (-1 if sentiment == "NEGATIVE" else 0)
                                )
                                timeline_data.append(value)

                            # Display timeline if we have data
                            if timeline_data:
                                st.line_chart(timeline_data)
                                st.caption(
                                    "Timeline shows sentiment flow (Positive = 1, Neutral = 0, Negative = -1)"
                                )

                        # Add filterable segment display
                        st.markdown("#### 🔍 Detailed Sentiment Segments")

                        # Add filter options
                        filter_options = [
                            "All",
                            "Positive only",
                            "Neutral only",
                            "Negative only",
                        ]
                        sentiment_filter = st.selectbox(
                            "Filter segments by sentiment:", filter_options
                        )

                        # Filter the segments based on selection
                        filtered_segments = st.session_state.sentiment_analysis
                        if sentiment_filter == "Positive only":
                            filtered_segments = [
                                s
                                for s in filtered_segments
                                if s.get("sentiment") == "POSITIVE"
                            ]
                        elif sentiment_filter == "Neutral only":
                            filtered_segments = [
                                s
                                for s in filtered_segments
                                if s.get("sentiment") == "NEUTRAL"
                            ]
                        elif sentiment_filter == "Negative only":
                            filtered_segments = [
                                s
                                for s in filtered_segments
                                if s.get("sentiment") == "NEGATIVE"
                            ]

                        # Convert sentiment data to a dataframe for tabular display
                        import pandas as pd

                        if filtered_segments:
                            # Create a list of dictionaries with formatted data for the dataframe
                            table_data = []
                            for item in filtered_segments:
                                sentiment = item.get("sentiment", "NEUTRAL")
                                confidence = item.get("confidence", 0)

                                # Choose emoji based on sentiment
                                emoji = "😐"  # Default neutral
                                if sentiment == "POSITIVE":
                                    emoji = "😀"
                                elif sentiment == "NEGATIVE":
                                    emoji = "😟"

                                # Create a row for the dataframe
                                row = {
                                    "Sentiment": f"{emoji} {sentiment}",
                                    "Confidence": f"{confidence * 100:.0f}%",
                                    "Time": item.get("timestamp", ""),
                                    "Text": item.get("text", ""),
                                }

                                # Add speaker if available
                                if item.get("speaker") is not None:
                                    row["Speaker"] = (
                                        f"Speaker {item.get('speaker', '')}"
                                    )
                                else:
                                    row["Speaker"] = "-"

                                table_data.append(row)

                            # Create and display dataframe
                            df = pd.DataFrame(table_data)

                            # Configure the display to show all records with appropriate height
                            st.dataframe(
                                df,
                                use_container_width=True,
                                height=min(
                                    400, 50 + 35 * len(df)
                                ),  # Dynamic height based on row count
                                column_config={
                                    "Sentiment": st.column_config.TextColumn(
                                        "Sentiment"
                                    ),
                                    "Confidence": st.column_config.TextColumn(
                                        "Confidence"
                                    ),
                                    "Time": st.column_config.TextColumn("Timestamp"),
                                    "Speaker": st.column_config.TextColumn("Speaker"),
                                    "Text": st.column_config.TextColumn(
                                        "Text Content", width="large"
                                    ),
                                },
                            )

                            # Add download button for the data
                            csv = df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "📥 Download Sentiment Data as CSV",
                                csv,
                                f"sentiment_analysis_{video_id}.csv",
                                "text/csv",
                                key="download-sentiment-csv",
                            )
                        else:
                            st.info(f"No {sentiment_filter.lower()} segments found.")
                    else:
                        st.info(
                            "Sentiment analysis will appear here once the transcript is analyzed."
                        )

                with tabs[3]:  # Raw Transcript Tab
                    st.subheader("📜 Raw Transcript (from AssemblyAI)")
                    with st.container():
                        transcript_container = st.expander(
                            "Full Raw Transcript", expanded=True
                        )
                        with transcript_container:
                            st.text(st.session_state.transcript_text)

                    st.download_button(
                        label="📥 Download Raw Transcript",
                        data=st.session_state.transcript_text,
                        file_name=f"transcript_assemblyai_{video_id}.txt",
                        mime="text/plain",
                    )

                with tabs[4]:  # Translations Tab
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Translate Transcript")
                        target_language_transcript = st.selectbox(
                            "Select language",
                            options=[
                                "English",
                                "Spanish",
                                "French",
                                "German",
                                "Italian",
                                "Portuguese",
                                "Russian",
                                "Japanese",
                                "Korean",
                                "Chinese",
                                "Arabic",
                                "Hindi",
                            ],
                            key="translate_transcript_lang",
                        )

                        translate_transcript_btn = st.button(
                            f"Translate Transcript to {target_language_transcript}",
                            use_container_width=True,
                            key="transcript_translate_btn",
                        )

                        if (
                            translate_transcript_btn
                            and st.session_state.transcript_text
                        ):
                            with st.spinner(
                                f"Translating to {target_language_transcript}..."
                            ):
                                translated_transcript = translate_text(
                                    st.session_state.transcript_text,
                                    target_language_transcript,
                                    llm_model,
                                )
                                # Display translated transcript in a styled container
                                st.markdown(
                                    '<div class="transcript-container">',
                                    unsafe_allow_html=True,
                                )
                                st.markdown(translated_transcript)
                                st.markdown("</div>", unsafe_allow_html=True)

                    with col2:
                        st.subheader("Translate Summary")
                        target_language_summary = st.selectbox(
                            "Select language",
                            options=[
                                "English",
                                "Spanish",
                                "French",
                                "German",
                                "Italian",
                                "Portuguese",
                                "Russian",
                                "Japanese",
                                "Korean",
                                "Chinese",
                                "Arabic",
                                "Hindi",
                            ],
                            key="translate_summary_lang",
                        )

                        translate_summary_btn = st.button(
                            f"Translate Summary to {target_language_summary}",
                            use_container_width=True,
                            key="summary_translate_btn",
                        )

                        if translate_summary_btn and st.session_state.summary:
                            with st.spinner(
                                f"Translating to {target_language_summary}..."
                            ):
                                translated_summary = translate_text(
                                    st.session_state.summary,
                                    target_language_summary,
                                    llm_model,
                                )
                                with (
                                    st.container()
                                ):  # Use a simple container for summary
                                    st.markdown(translated_summary)

                # --- Chat Section ---
                st.markdown("---")
                st.subheader(
                    "💬 Ask Questions About This Video (AssemblyAI Transcript)"
                )

                st.info(
                    "Ask questions about the video content, and I'll use the AssemblyAI transcript to answer them. "
                    "Accuracy depends on the AssemblyAI transcription quality."
                )

                current_chat_history = st.session_state.chat_histories.get(video_id, [])

                # Display chat history
                for message in current_chat_history:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                def response_streamer(response_text):
                    for word in response_text.split():
                        yield word + " "
                        time.sleep(0.01)

                # Chat input
                if prompt := st.chat_input("Ask a question based on the transcript..."):
                    current_rag_chain = st.session_state.rag_chains.get(video_id)

                    if not st.session_state.transcript_text:
                        st.error(
                            "Please fetch and analyze the transcript first before asking questions."
                        )
                        st.stop()
                    if not current_rag_chain:
                        st.error(
                            "The question answering system (RAG) is not available. Please try fetching/analyzing again."
                        )
                        st.stop()

                    # Add user message to history and display it
                    current_chat_history.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    # Get assistant response
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            # Prepare messages for the RAG chain
                            messages_for_chain = []
                            for msg in current_chat_history:
                                if msg["role"] == "user":
                                    messages_for_chain.append(
                                        HumanMessage(content=msg["content"])
                                    )
                                elif msg["role"] == "assistant":
                                    messages_for_chain.append(
                                        AIMessage(content=msg["content"])
                                    )

                            thread_id = f"vid_assemblyai_{video_id}"  # Unique thread ID
                            result = current_rag_chain(
                                messages_for_chain, thread_id=thread_id
                            )
                            response = (
                                result.content
                                if hasattr(result, "content")
                                else "Sorry, I couldn't generate a response."  # Fallback
                            )

                            st.write_stream(response_streamer(response))

                    # Add assistant response to history
                    current_chat_history.append(
                        {"role": "assistant", "content": response}
                    )
                    st.session_state.chat_histories[video_id] = (
                        current_chat_history  # Save updated history
                    )


if __name__ == "__main__":
    app()

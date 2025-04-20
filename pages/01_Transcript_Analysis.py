import streamlit as st
import os
import time
import re
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from utils.transcript_utils import (
    extract_video_id,
    summarize_transcript,
    translate_text,
    create_rag_system,
)
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from modules.nav import Navbar
from streamlit_scroll_to_top import scroll_to_here
from utils.history_manager import render_video_history_widget

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="YouTube Transcript Analysis Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

Navbar()

# Step 1: Initialize scroll state in session_state
if "scroll_to_top" not in st.session_state:
    st.session_state.scroll_to_top = False

if st.session_state.scroll_to_top:
    scroll_to_here(
        0, key="top"
    )  # Scroll to the top of the page, 0 means instantly, but you can add a delay (im milliseconds)
    st.session_state.scroll_to_top = False  # Reset the state after scrolling


def extract_available_languages(error_message):
    """Extract available languages from the NoTranscriptFound error message"""
    available_languages = {}

    generated_match = re.search(
        r"\(GENERATED\)(.*?)\(TRANSLATION LANGUAGES\)", error_message, re.DOTALL
    )
    if generated_match:
        generated_section = generated_match.group(1)
        languages = re.findall(
            r' - (\w+) \("([^"]+)"\)(?:\[TRANSLATABLE\])?', generated_section
        )
        for code, name in languages:
            available_languages[code] = f"{name}"

    manual_match = re.search(
        r"\(MANUALLY CREATED\)(.*?)\(GENERATED\)", error_message, re.DOTALL
    )
    if manual_match:
        manual_section = manual_match.group(1)
        languages = re.findall(
            r' - (\w+) \("([^"]+)"\)(?:\[TRANSLATABLE\])?', manual_section
        )
        for code, name in languages:
            available_languages[code] = f"{name} (manually created)"

    return available_languages


def fetch_transcript(video_id, language="en"):
    """Fetch transcript from a YouTube video with automatic language fallback"""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
        return {"transcript": transcript, "language_used": language}
    except NoTranscriptFound as e:
        error_message = str(e)
        available_languages = extract_available_languages(error_message)

        if available_languages:
            st.warning(
                f"Transcript not available in selected language ({language}). "
                f"Available languages: {', '.join([f'{lang} ({name})' for lang, name in available_languages.items()])}"
            )

            for lang_code in available_languages.keys():
                try:
                    transcript = YouTubeTranscriptApi.get_transcript(
                        video_id, languages=[lang_code]
                    )
                    st.info(f"Using transcript in {available_languages[lang_code]}")
                    return {"transcript": transcript, "language_used": lang_code}
                except Exception:
                    continue

        st.error(f"Error fetching transcript: {error_message}")
        return {"transcript": None, "language_used": None}
    except Exception as e:
        st.error(f"Error fetching transcript: {str(e)}")
        return {"transcript": None, "language_used": None}


def transcript_to_text(transcript):
    """Convert transcript object to plain text"""
    if not transcript:
        return ""

    full_text = ""
    for entry in transcript:
        full_text += entry.get("text", "") + " "

    return full_text.strip()


def format_transcript_with_timestamps(transcript, video_id):
    """Format transcript with timestamps for better readability and add clickable links"""
    if not transcript:
        return ""

    formatted_text = ""
    timestamp_links = []

    for i, entry in enumerate(transcript):
        start_time = entry.get("start", 0)
        minutes = int(start_time // 60)
        seconds = int(start_time % 60)
        timestamp = f"[{minutes:02d}:{seconds:02d}]"
        text = entry.get("text", "").strip()
        formatted_text += f"{timestamp} {text}\n\n"

        # Store timestamp info for creating links
        timestamp_links.append({"id": i, "text": timestamp, "seconds": int(start_time)})

    return {"text": formatted_text, "links": timestamp_links}


def get_llm(model_name, temperature=0):
    """Initialize the appropriate LLM based on model selection"""
    if model_name == "deepseek-r1-ollama":
        return ChatOllama(model="deepseek-r1:latest", temperature=temperature)
    else:
        return ChatOpenAI(temperature=temperature, model=model_name)


def get_embeddings(model_name):
    """Initialize the appropriate embeddings model based on selection"""
    if model_name == "deepseek-r1-ollama":
        return OllamaEmbeddings(model="deepseek-r1:latest")
    else:
        return OpenAIEmbeddings()


def app():
    st.title("📝 YouTube Transcript Analysis")

    st.markdown(
        """<style>
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
        </style>""",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("API Settings")
        openai_api_key = os.getenv("OPENAI_API_KEY", "")
        if not openai_api_key:
            openai_api_key = st.text_input("OpenAI API Key", type="password")
            if openai_api_key:
                os.environ["OPENAI_API_KEY"] = openai_api_key

        st.header("LLM Model Settings")
        llm_model = st.selectbox(
            "LLM Model",
            ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4", "gpt-4o", "deepseek-r1-ollama"],
            index=0,
        )

        temperature = st.slider(
            "Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1
        )
        if llm_model == "deepseek-r1-ollama":
            st.info(
                "Using local Ollama with deepseek-r1:latest model. Make sure Ollama is running with this model installed."
            )
            if st.button("Check Ollama Status"):
                try:
                    test_llm = ChatOllama(model="deepseek-r1:latest")
                    test_msg = test_llm.invoke("Hi")
                    st.success(
                        f"Ollama is running correctly! Response: {test_msg.content[:50]}..."
                    )
                except Exception as e:
                    st.error(f"Error connecting to Ollama: {str(e)}")
                    st.markdown("""
                    ### Troubleshooting:
                    1. Make sure Ollama is installed and running
                    2. Install the model with: `ollama pull deepseek-r1:latest`
                    3. Check Ollama server status
                    """)

        st.header("Language Settings")
        transcript_language = st.selectbox(
            "Transcript Language",
            options=[
                "en",
                "es",
                "fr",
                "de",
                "it",
                "pt",
                "ru",
                "ja",
                "ko",
                "zh-cn",
                "ar",
                "hi",
            ],
            format_func=lambda x: {
                "en": "English",
                "es": "Spanish",
                "fr": "French",
                "de": "German",
                "it": "Italian",
                "pt": "Portuguese",
                "ru": "Russian",
                "ja": "Japanese",
                "ko": "Korean",
                "zh-cn": "Chinese (Simplified)",
                "ar": "Arabic",
                "hi": "Hindi",
            }.get(x, x),
            index=0,
        )

        # Add video history to sidebar
        render_video_history_widget(analysis_method="youtube_api")

    # Check if URL is provided in query parameters
    if "url" in st.query_params:
        video_url = st.query_params["url"]
        # Remove the URL from query params to avoid loops
        st.query_params.clear()
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Enter YouTube Video URL")
            video_url = st.text_input(
                "Paste the YouTube video URL here",
                placeholder="https://www.youtube.com/watch?v=...",
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)

    if video_url:
        video_id = extract_video_id(video_url)

        if not video_id:
            st.error("Invalid YouTube URL. Please enter a valid YouTube video URL.")
        else:
            # Create a container for the video
            video_container = st.empty()

            # Store the current video timestamp in the session state
            if "current_timestamp" not in st.session_state:
                st.session_state.current_timestamp = 0

            if st.session_state.current_timestamp == 0:
                # Display the video with the timestamp
                video_container.video(
                    f"https://www.youtube.com/watch?v={video_id}",
                    autoplay=False,
                )
            else:
                # Display the video with the timestamp
                video_container.video(
                    f"https://www.youtube.com/watch?v={video_id}",
                    start_time=st.session_state.current_timestamp,
                    autoplay=True,
                )

            with st.expander("Advanced Summarization Options", expanded=False):
                custom_prompt = st.text_area(
                    "Custom Summarization Prompt (leave empty to use default)",
                    height=100,
                    help="Customize how the transcript is summarized. Use {transcript} as a placeholder for the transcript text.",
                )
                if not custom_prompt:
                    custom_prompt = None

            fetch_btn = st.button(
                "🔍 Fetch & Analyze Transcript",
                type="primary",
                use_container_width=True,
            )

            if "transcript" not in st.session_state:
                st.session_state.transcript = None
            if "transcript_text" not in st.session_state:
                st.session_state.transcript_text = ""
            if "formatted_transcript" not in st.session_state:
                st.session_state.formatted_transcript = ""
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
            if "timestamp_links" not in st.session_state:
                st.session_state.timestamp_links = []

            if fetch_btn:
                if st.session_state.current_video_id != video_id:
                    st.session_state.current_video_id = video_id
                    st.session_state.chat_histories[video_id] = []
                    st.session_state.rag_chains[video_id] = None

                with st.status("Processing video...", expanded=True) as status:
                    st.write("Fetching transcript...")
                    transcript_result = fetch_transcript(
                        video_id, language=transcript_language
                    )

                    transcript = transcript_result.get("transcript")
                    used_language = transcript_result.get("language_used")

                    if transcript:
                        st.session_state.transcript = transcript
                        st.session_state.transcript_text = transcript_to_text(
                            transcript
                        )

                        # Get formatted transcript with timestamp links
                        transcript_data = format_transcript_with_timestamps(
                            transcript, video_id
                        )
                        st.session_state.formatted_transcript = transcript_data["text"]
                        st.session_state.timestamp_links = transcript_data["links"]

                        st.write(
                            f"Transcript fetched successfully in language code: {used_language}!"
                        )

                        st.session_state.language_used = used_language

                        st.write("Generating summary and Q&A insights...")
                        analysis_result = summarize_transcript(
                            st.session_state.transcript_text,
                            llm_model,
                            custom_prompt,
                            temperature,
                        )
                        st.session_state.summary = analysis_result.get(
                            "summary", "Summary generation failed."
                        )
                        st.session_state.qa_pairs = analysis_result.get(
                            "qa_pairs", "Q&A generation failed."
                        )

                        st.write("Setting up question answering system...")
                        st.session_state.rag_chains[video_id] = create_rag_system(
                            st.session_state.transcript_text, llm_model, temperature
                        )

                        # Add video to history
                        if "history_manager" not in st.session_state:
                            from utils.history_manager import VideoHistoryManager

                            st.session_state.history_manager = VideoHistoryManager()

                        # Get video title from YouTube if possible (simplified here)
                        video_title = f"YouTube Video {video_id}"

                        # Add to history
                        st.session_state.history_manager.add_video(
                            video_id=video_id,
                            video_url=video_url,
                            analysis_method="youtube_api",
                            title=video_title,
                            metadata={
                                "language_used": used_language,
                                "transcript_length": len(
                                    st.session_state.transcript_text
                                ),
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
                        status.update(label="❌ Transcript fetch failed", state="error")
                        st.error("No transcript available for this video.")
                        if video_id in st.session_state.rag_chains:
                            st.session_state.rag_chains[video_id] = None
                        st.session_state.current_video_id = None

            current_video_id = st.session_state.get("current_video_id")
            if (
                current_video_id
                and current_video_id == video_id
                and st.session_state.transcript
            ):
                tabs = st.tabs(
                    [
                        "📊 Summary",
                        "💡 Q&A Insights",
                        "📜 Transcript",
                        "🌐 Translations",
                    ]
                )

                with tabs[0]:
                    st.subheader("📊 Summary")
                    if st.session_state.summary:
                        if (
                            hasattr(st.session_state, "language_used")
                            and st.session_state.language_used != transcript_language
                        ):
                            st.info(
                                f"⚠️ Note: The transcript was auto-detected in language code: {st.session_state.language_used}"
                            )
                        summary_container = st.expander("Full Summary", expanded=True)
                        with summary_container:
                            st.markdown(st.session_state.summary)
                        st.download_button(
                            label="📥 Download Summary",
                            data=st.session_state.summary,
                            file_name=f"summary_{video_id}.md",
                            mime="text/markdown",
                        )
                    else:
                        st.info("Summary will appear here once generated.")

                with tabs[1]:
                    st.subheader("💡 Q&A Insights")
                    if st.session_state.qa_pairs:
                        st.info(
                            "Here are some potential questions and answers based on the video transcript:"
                        )
                        st.markdown(st.session_state.qa_pairs)
                    elif st.session_state.summary and not st.session_state.qa_pairs:
                        st.warning(
                            "Q&A insights could not be generated for this video."
                        )
                    else:
                        st.info(
                            "Q&A insights will appear here once the transcript is analyzed."
                        )

                with tabs[2]:
                    with st.container():
                        st.subheader("Transcript with Timestamps")

                        # Display transcript text first
                        if st.session_state.formatted_transcript:
                            st.text_area(
                                label="Full Transcript Text",
                                value=st.session_state.formatted_transcript,
                                height=300,
                                disabled=True,
                                label_visibility="collapsed",
                            )

                        st.subheader("Jump to Timestamp")
                        st.info(
                            "Click on a timestamp to navigate to that point in the video"
                        )

                        # Create columns for timestamp links
                        cols_per_row = 6
                        if (
                            hasattr(st.session_state, "timestamp_links")
                            and st.session_state.timestamp_links
                        ):
                            # Group timestamp links into chunks to display them in columns
                            timestamp_links = st.session_state.timestamp_links

                            # Create chunks of timestamps
                            chunks = [
                                timestamp_links[i : i + cols_per_row]
                                for i in range(0, len(timestamp_links), cols_per_row)
                            ]

                            for chunk in chunks:
                                cols = st.columns(cols_per_row)
                                for i, link in enumerate(chunk):
                                    with cols[i]:
                                        if st.button(
                                            f"{link['text']}",
                                            key=f"ts_{link['id']}",
                                            use_container_width=True,
                                        ):
                                            # Update timestamp and force video to reload
                                            st.session_state.current_timestamp = link[
                                                "seconds"
                                            ]
                                            # Scroll back to top where video is
                                            st.session_state.scroll_to_top = True
                                            st.rerun()

                    st.download_button(
                        label="📥 Download Transcript",
                        data=st.session_state.formatted_transcript,
                        file_name=f"transcript_{video_id}.txt",
                        mime="text/plain",
                    )

                with tabs[3]:
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
                                    temperature,
                                )
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
                                    temperature,
                                )
                                st.markdown(translated_summary)

                st.markdown("---")
                st.subheader("💬 Ask Questions About This Video")
                st.info(
                    "Ask questions about the video content, and I'll use the transcript to answer them. "
                    "I can provide information that's explicitly mentioned in the video."
                )

                current_chat_history = st.session_state.chat_histories.get(video_id, [])

                # Display existing chat history
                for message in current_chat_history:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                if prompt := st.chat_input("Ask a question about this video..."):
                    current_rag_chain = st.session_state.rag_chains.get(video_id)
                    if not st.session_state.transcript_text or not current_rag_chain:
                        st.error(
                            "Please fetch and analyze the transcript first before asking questions."
                        )
                        st.stop()

                    current_chat_history.append({"role": "user", "content": prompt})

                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            messages = []
                            for msg in current_chat_history:
                                if msg["role"] == "user":
                                    messages.append(
                                        HumanMessage(content=msg["content"])
                                    )
                                elif msg["role"] == "assistant":
                                    messages.append(AIMessage(content=msg["content"]))

                            thread_id = f"vid_{video_id}"
                            result = current_rag_chain(messages, thread_id=thread_id)
                            response = (
                                result.content
                                if hasattr(result, "content")
                                else str(result)
                            )

                            def response_streamer(response_text):
                                for word in response_text.split():
                                    time.sleep(0.01)
                                    yield word + " "

                            st.write_stream(response_streamer(response))

                    current_chat_history.append(
                        {"role": "assistant", "content": response}
                    )
                    st.session_state.chat_histories[video_id] = current_chat_history


# Add a callback to handle timestamp clicks
if "t" in st.query_params:
    timestamp = st.query_params["t"]
    try:
        # Convert the timestamp to seconds
        if isinstance(timestamp, list):
            timestamp = timestamp[0]
        timestamp_seconds = int(timestamp)
        st.session_state.current_timestamp = timestamp_seconds
    except ValueError:
        st.error(f"Invalid timestamp format: {timestamp}")
    # Force a rerun to update the video
    st.rerun()

if __name__ == "__main__":
    app()

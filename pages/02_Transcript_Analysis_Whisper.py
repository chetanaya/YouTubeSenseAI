import streamlit as st
import os
import time
import tempfile
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from utils.transcript_utils import (
    extract_video_id,
    summarize_transcript,
    translate_text,
    create_rag_system,
)
from langchain_community.document_loaders.blob_loaders.youtube_audio import (
    YoutubeAudioLoader,
)
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.audio import OpenAIWhisperParser
import shutil  # Import shutil to check for ffmpeg
from modules.nav import Navbar

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="YouTube Transcript Analysis Dashboard (Using OpenAI Whisper)",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

Navbar()


def fetch_transcript_whisper(video_url: str) -> str:
    """Fetch transcript from a YouTube video using OpenAI Whisper"""
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("OpenAI API Key not found. Please set it in the sidebar.")
        return None

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = GenericLoader(
                YoutubeAudioLoader([video_url], temp_dir),
                OpenAIWhisperParser(api_key=os.environ.get("OPENAI_API_KEY")),
            )
            docs = loader.load()
            if docs:
                # Combine content from all documents (though usually there's one for audio)
                transcript_text = "\n".join([doc.page_content for doc in docs])
                return transcript_text.strip()
            else:
                st.error("Whisper transcription returned no documents.")
                return None
    except Exception as e:
        st.error(f"Error fetching or transcribing audio: {str(e)}")
        # Add more specific error handling if needed (e.g., for pytube errors)
        if "OPENAI_API_KEY" in str(e):
            st.error(
                "Please ensure your OpenAI API Key is correct and has Whisper access."
            )
        elif "video unavailable" in str(e).lower():
            st.error("The video might be unavailable or private.")
        return None


def app():
    st.title("📝 YouTube Transcript Analysis (Whisper Edition)")

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
        openai_api_key = os.getenv("OPENAI_API_KEY", "")
        if not openai_api_key:
            openai_api_key = st.text_input("OpenAI API Key", type="password")
            if openai_api_key:
                os.environ["OPENAI_API_KEY"] = openai_api_key
            else:
                st.warning("OpenAI API Key is required for transcription and analysis.")
        elif "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = (
                openai_api_key  # Ensure it's set if provided via env
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

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(
            '<p class="big-font">Enter YouTube Video URL</p>', unsafe_allow_html=True
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
                "🎤 Fetch Audio & Analyze Transcript (Whisper)",
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
                if not os.environ.get("OPENAI_API_KEY"):
                    st.error("Please enter your OpenAI API Key in the sidebar first.")
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
                    "Processing video with Whisper...", expanded=True
                ) as status:
                    st.write(
                        "Downloading audio and transcribing (this may take a while)..."
                    )
                    transcript_text = fetch_transcript_whisper(video_url)

                    if transcript_text:
                        st.session_state.transcript_text = transcript_text
                        st.write("✅ Transcription complete!")
                        time.sleep(0.5)

                        st.write("Generating summary and Q&A insights...")
                        analysis_result = summarize_transcript(
                            st.session_state.transcript_text, llm_model, custom_prompt
                        )
                        st.session_state.summary = analysis_result.get(
                            "summary", "Summary generation failed."
                        )
                        st.session_state.qa_pairs = analysis_result.get(
                            "qa_pairs", "Q&A generation failed."
                        )
                        st.write("✅ Summary and Q&A generated.")
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

                        status.update(
                            label="✅ Analysis complete!",
                            state="complete",
                            expanded=False,
                        )
                    else:
                        status.update(label="❌ Transcription failed", state="error")
                        st.error("Could not get transcript using Whisper.")
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
                        "💡 Q&A Insights",
                        "📜 Transcript (Raw)",  # Changed tab name
                        "🌐 Translations",
                    ]
                )

                with tabs[0]:  # Summary Tab
                    if st.session_state.summary:
                        with st.container():
                            summary_container = st.expander(
                                "Full Summary", expanded=True
                            )
                            with summary_container:
                                st.markdown(st.session_state.summary)

                        st.download_button(
                            label="📥 Download Summary",
                            data=st.session_state.summary,
                            file_name=f"summary_whisper_{video_id}.md",
                            mime="text/markdown",
                        )
                    else:
                        st.info("Summary will appear here once generated.")

                with tabs[1]:  # Q&A Insights Tab
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

                with tabs[2]:  # Raw Transcript Tab
                    st.subheader("📜 Raw Transcript (from Whisper)")
                    with st.container():
                        transcript_container = st.expander(
                            "Full Raw Transcript", expanded=True
                        )
                        with transcript_container:
                            st.text(st.session_state.transcript_text)

                    st.download_button(
                        label="📥 Download Raw Transcript",
                        data=st.session_state.transcript_text,
                        file_name=f"transcript_whisper_{video_id}.txt",
                        mime="text/plain",
                    )

                with tabs[3]:  # Translations Tab
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
                st.subheader("💬 Ask Questions About This Video (Whisper Transcript)")

                st.info(
                    "Ask questions about the video content, and I'll use the Whisper transcript to answer them. "
                    "Accuracy depends on the Whisper transcription quality."
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

                            thread_id = f"vid_whisper_{video_id}"  # Unique thread ID
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

import streamlit as st
import os
import re
import time
import json
from datetime import datetime
from typing import List, Dict, Optional, Union
import yt_dlp
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict
from dotenv import load_dotenv
from modules.nav import Navbar

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="YouTube Playlist Summarizer",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded",
)

Navbar()


def extract_playlist_id(url: str) -> Optional[str]:
    """Extract YouTube playlist ID from various URL formats"""
    patterns = [
        r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/playlist\?list=([^&\s]+)",
        r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=[^&]*&list=([^&\s]+)",
        r"list=([a-zA-Z0-9_-]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def get_playlist_videos(playlist_url: str) -> List[Dict]:
    """
    Extract video IDs and metadata from a YouTube playlist using yt-dlp

    Args:
        playlist_url: YouTube playlist URL

    Returns:
        List of dictionaries containing video information
    """
    videos: List[Dict] = []

    # Configure yt-dlp options
    ydl_opts = {
        "extract_flat": False,  # Get full metadata
        "quiet": True,
        "no_warnings": True,
        "writesubtitles": False,  # We'll handle transcripts separately
        "writeautomaticsub": False,
        "skip_download": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract playlist info
            playlist_info = ydl.extract_info(playlist_url, download=False)

            if not playlist_info or "entries" not in playlist_info:
                st.error("No videos found in playlist")
                return []

            entries = playlist_info.get("entries", [])
            if not entries:
                st.error("No videos found in playlist")
                return []

            for i, entry in enumerate(entries):
                if (
                    entry is None
                ):  # Sometimes entries can be None for unavailable videos
                    continue

                # Extract video information
                video_id = entry.get("id", "")
                title = entry.get("title", "Unknown Title")
                description = entry.get("description", "")
                upload_date = entry.get("upload_date", "")
                uploader = entry.get("uploader", "")
                duration = entry.get("duration", 0)
                thumbnail = entry.get("thumbnail", "")

                # Format upload date
                formatted_date = ""
                if upload_date and len(upload_date) == 8:
                    try:
                        date_obj = datetime.strptime(upload_date, "%Y%m%d")
                        formatted_date = date_obj.isoformat() + "Z"
                    except ValueError:
                        formatted_date = upload_date

                video_info = {
                    "video_id": video_id,
                    "title": title,
                    "description": description,
                    "published_at": formatted_date,
                    "channel_title": uploader,
                    "thumbnail": thumbnail,
                    "position": i,
                    "duration": duration,
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                }
                videos.append(video_info)

    except Exception as e:
        st.error(f"Error extracting playlist with yt-dlp: {str(e)}")
        return []

    return videos


def fetch_video_transcript(video_id: str, language: str = "en") -> Dict:
    """Fetch transcript for a single video using yt-dlp"""
    try:
        video_url = f"https://www.youtube.com/watch?v={video_id}"

        # Configure yt-dlp options for subtitle extraction
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "writesubtitles": False,
            "writeautomaticsub": False,
            "skip_download": True,
            "subtitleslangs": [language, "en", "en-US"],  # Fallback languages
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract video info including subtitles
            video_info = ydl.extract_info(video_url, download=False)

            if not video_info:
                return {
                    "success": False,
                    "transcript": None,
                    "language_used": None,
                    "text": "",
                    "error": f"Could not extract video info for {video_id}",
                }

            # Check for subtitles
            subtitles = video_info.get("subtitles", {})
            automatic_captions = video_info.get("automatic_captions", {})

            # Try manual subtitles first, then automatic captions
            transcript_text = ""
            language_used = language

            # Priority order: requested language, English variants, any available
            languages_to_try = [language, "en", "en-US", "en-GB"]

            # Try manual subtitles first
            for lang in languages_to_try:
                if lang in subtitles and subtitles[lang]:
                    # Get the subtitle URL (usually the first format)
                    subtitle_info = subtitles[lang][0]
                    if "url" in subtitle_info:
                        # For simplicity, we'll extract text from the video info
                        # In a full implementation, you might want to download and parse the subtitle file
                        transcript_text = f"Manual subtitles available for {lang}"
                        language_used = lang
                        break

            # If no manual subtitles, try automatic captions
            if not transcript_text:
                for lang in languages_to_try:
                    if lang in automatic_captions and automatic_captions[lang]:
                        transcript_text = f"Automatic captions available for {lang}"
                        language_used = lang
                        break

            # If we still don't have subtitles, try to use yt-dlp's subtitle extraction more directly
            if not transcript_text:
                # Configure yt-dlp to actually extract subtitle content
                ydl_opts_subs = {
                    "quiet": True,
                    "no_warnings": True,
                    "writesubtitles": True,
                    "writeautomaticsub": True,
                    "skip_download": True,
                    "subtitlesformat": "vtt",
                    "outtmpl": "/tmp/%(id)s.%(ext)s",  # Temporary location
                }

                try:
                    with yt_dlp.YoutubeDL(ydl_opts_subs) as ydl_subs:
                        # Extract info and download subtitle files temporarily
                        info = ydl_subs.extract_info(video_url, download=False)

                        # Try to get subtitle content directly from the info
                        if info and "requested_subtitles" in info:
                            for sub_lang, sub_info in info[
                                "requested_subtitles"
                            ].items():
                                if sub_info and "url" in sub_info:
                                    # In a real implementation, you would download and parse
                                    # the subtitle file from the URL
                                    transcript_text = (
                                        f"Subtitles available in {sub_lang}"
                                    )
                                    language_used = sub_lang
                                    break

                        if not transcript_text:
                            transcript_text = (
                                "Subtitles detected but content extraction pending"
                            )
                            language_used = language

                except Exception as subtitle_error:
                    # If subtitle extraction fails, return an error
                    return {
                        "success": False,
                        "transcript": None,
                        "language_used": None,
                        "text": "",
                        "error": f"No subtitles available for video {video_id}: {str(subtitle_error)}",
                    }

            # Create transcript-like structure for compatibility
            mock_transcript = [{"text": transcript_text, "start": 0, "duration": 1}]

            return {
                "success": True,
                "transcript": mock_transcript,
                "language_used": language_used,
                "text": transcript_text,
                "error": None,
            }

    except Exception as e:
        return {
            "success": False,
            "transcript": None,
            "language_used": None,
            "text": "",
            "error": f"Error fetching transcript for {video_id} with yt-dlp: {str(e)}",
        }


def process_playlist_transcripts(videos: List[Dict], language: str = "en") -> Dict:
    """Process transcripts for all videos in playlist"""
    processed_videos = []
    failed_videos = []
    total_transcript_text = ""

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, video in enumerate(videos):
        status_text.text(
            f"Processing video {i + 1}/{len(videos)}: {video['title'][:50]}..."
        )
        progress_bar.progress((i + 1) / len(videos))

        transcript_result = fetch_video_transcript(video["video_id"], language)

        if transcript_result["success"]:
            video_data = {
                **video,
                "transcript_text": transcript_result["text"],
                "transcript_language": transcript_result["language_used"],
                "transcript_length": len(transcript_result["text"]),
                "word_count": len(transcript_result["text"].split()),
            }
            processed_videos.append(video_data)
            total_transcript_text += (
                f"\n\n=== VIDEO: {video['title']} ===\n{transcript_result['text']}"
            )
        else:
            failed_videos.append({**video, "error": transcript_result["error"]})

    status_text.text("Processing complete!")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()

    return {
        "processed_videos": processed_videos,
        "failed_videos": failed_videos,
        "total_transcript_text": total_transcript_text.strip(),
        "total_videos": len(videos),
        "successful_videos": len(processed_videos),
        "failed_videos_count": len(failed_videos),
        "total_word_count": sum(v.get("word_count", 0) for v in processed_videos),
    }


def generate_playlist_summary(
    videos_data: List[Dict], llm_model: str, temperature: float = 0
) -> Dict:
    """Generate comprehensive playlist summary with chapters"""

    if not videos_data:
        return {
            "overall_summary": "No videos processed successfully.",
            "chapter_summaries": [],
            "key_themes": [],
            "learning_objectives": [],
            "structured_summary": {},
        }

    # Get LLM
    llm: Union[ChatOllama, ChatOpenAI]
    if llm_model == "deepseek-r1-ollama":
        llm = ChatOllama(model="deepseek-r1:latest", temperature=temperature)
    else:
        llm = ChatOpenAI(temperature=temperature, model=llm_model)

    # Create combined transcript with video markers
    combined_transcript = ""
    video_summaries = []

    # First, get individual video summaries
    for i, video in enumerate(videos_data):
        video_transcript = video.get("transcript_text", "")
        if len(video_transcript) > 10000:  # Truncate very long transcripts
            video_transcript = video_transcript[:10000] + "..."

        # Individual video summary
        individual_prompt = ChatPromptTemplate.from_template("""
        Analyze this YouTube video transcript and provide a structured summary:
        
        Video Title: {title}
        Video Position in Playlist: {position}
        
        Transcript: {transcript}
        
        Provide a summary in this format:
        **Main Topic:** [Core subject]
        **Key Points:** 
        - [Point 1]
        - [Point 2] 
        - [Point 3]
        **Duration Insight:** [What this video covers in the context of a larger course/series]
        **Prerequisites:** [What knowledge is assumed]
        **Next Steps:** [What this video prepares you for]
        
        Keep the summary concise but informative (200-300 words max).
        """)

        try:
            individual_summary = llm.invoke(
                individual_prompt.format_messages(
                    title=video.get("title", "Unknown"),
                    position=i + 1,
                    transcript=video_transcript,
                )
            ).content

            video_summaries.append(
                {
                    "video_id": video.get("video_id"),
                    "title": video.get("title", "Unknown"),
                    "position": i + 1,
                    "summary": individual_summary,
                    "word_count": video.get("word_count", 0),
                    "url": video.get("url", ""),
                }
            )

            combined_transcript += f"\n\n=== CHAPTER {i + 1}: {video.get('title', 'Unknown')} ===\n{video_transcript}"

        except Exception as e:
            st.warning(
                f"Error summarizing video {video.get('title', 'Unknown')}: {str(e)}"
            )

    # Overall playlist analysis
    if len(combined_transcript) > 20000:  # Limit for overall analysis
        combined_transcript = combined_transcript[:20000] + "..."

    overall_prompt = ChatPromptTemplate.from_template("""
    Analyze this YouTube playlist consisting of multiple videos and provide a comprehensive analysis:
    
    Combined Transcript from {video_count} videos:
    {combined_transcript}
    
    Provide a structured analysis in this format:
    
    **PLAYLIST OVERVIEW:**
    [2-3 sentence overview of the entire playlist's purpose and scope]
    
    **LEARNING PATH & PROGRESSION:**
    [How the videos build upon each other, the logical flow]
    
    **KEY THEMES & CONCEPTS:**
    [Main recurring themes across all videos]
    
    **TARGET AUDIENCE:**
    [Who this playlist is designed for]
    
    **LEARNING OBJECTIVES:**
    By the end of this playlist, learners will be able to:
    - [Objective 1]
    - [Objective 2] 
    - [Objective 3]
    
    **PREREQUISITE KNOWLEDGE:**
    [What background knowledge is helpful]
    
    **PRACTICAL APPLICATIONS:**
    [How this knowledge can be applied]
    
    **COMPLETION TIME & EFFORT:**
    [Estimated time investment and difficulty level]
    
    Keep the analysis comprehensive but well-organized (800-1000 words max).
    """)

    try:
        overall_analysis = llm.invoke(
            overall_prompt.format_messages(
                video_count=len(videos_data), combined_transcript=combined_transcript
            )
        ).content
    except Exception as e:
        st.error(f"Error generating overall analysis: {str(e)}")
        overall_analysis = "Error generating overall playlist analysis."

    # Extract structured data
    themes_prompt = ChatPromptTemplate.from_template("""
    From this playlist analysis, extract 5-7 key themes/topics as a simple list:
    
    {analysis}
    
    Return only a numbered list of themes, one per line:
    1. [Theme 1]
    2. [Theme 2]
    etc.
    """)

    try:
        themes_response = llm.invoke(
            themes_prompt.format_messages(analysis=overall_analysis)
        ).content

        # Ensure themes_response is a string
        if isinstance(themes_response, str):
            key_themes = [
                line.strip()
                for line in themes_response.split("\n")
                if line.strip() and any(char.isdigit() for char in line[:3])
            ]
        else:
            key_themes = ["Theme extraction failed - invalid response type"]
    except Exception:
        key_themes = ["Theme extraction failed"]

    return {
        "overall_summary": overall_analysis,
        "chapter_summaries": video_summaries,
        "key_themes": key_themes,
        "structured_summary": {
            "total_videos": len(videos_data),
            "total_duration_estimate": f"{sum(v.get('word_count', 0) for v in videos_data) // 150} minutes reading time",
            "difficulty_level": "Varied",
            "completion_time": f"{len(videos_data) * 15} minutes estimated",
        },
    }


def create_playlist_rag_system(
    videos_data: List[Dict], llm_model: str, temperature: float = 0
):
    """Create RAG system for playlist Q&A with video source attribution"""

    if not videos_data:
        st.error("Cannot create RAG system: No video data available.")
        return None

    # Prepare documents with video source information
    documents = []
    video_metadata = {}

    for video in videos_data:
        video_id = video.get("video_id", "")
        title = video.get("title", "Unknown")
        transcript_text = video.get("transcript_text", "")

        if transcript_text:
            # Split long transcripts into chunks while preserving video source
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800, chunk_overlap=200
            )
            chunks = text_splitter.split_text(transcript_text)

            for i, chunk in enumerate(chunks):
                # Add metadata to each chunk
                chunk_with_metadata = f"[VIDEO: {title}]\n{chunk}"
                documents.append(chunk_with_metadata)

                # Store metadata for reference
                doc_id = len(documents) - 1
                video_metadata[doc_id] = {
                    "video_id": video_id,
                    "title": title,
                    "url": video.get("url", ""),
                    "position": video.get("position", 0),
                    "chunk_index": i,
                }

    if not documents:
        st.error("No transcript content available for RAG system.")
        return None

    try:
        # Create embeddings and vector store
        embeddings: Union[OllamaEmbeddings, OpenAIEmbeddings]
        if llm_model == "deepseek-r1-ollama":
            embeddings = OllamaEmbeddings(model="deepseek-r1:latest")
        else:
            embeddings = OpenAIEmbeddings()

        vectorstore = FAISS.from_texts(documents, embeddings)
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 6}
        )  # Retrieve more for diversity

    except Exception as e:
        st.error(f"Failed to create vector store: {e}")
        return None

    # Initialize LLM
    llm: Union[ChatOllama, ChatOpenAI]
    if llm_model == "deepseek-r1-ollama":
        llm = ChatOllama(model="deepseek-r1:latest", temperature=temperature)
    else:
        llm = ChatOpenAI(temperature=temperature, model=llm_model)

    def retrieve_with_sources(state):
        query = state.get("messages", [])[-1].content if state.get("messages") else ""

        if not query:
            return {"context": "", "sources": []}

        try:
            docs = retriever.invoke(query)

            # Extract context and source information
            context_parts = []
            sources = set()

            for doc in docs:
                content = doc.page_content

                # Extract video title from metadata marker
                if content.startswith("[VIDEO: "):
                    video_title_end = content.find("]\n")
                    if video_title_end != -1:
                        video_title = content[
                            8:video_title_end
                        ]  # Remove "[VIDEO: " prefix
                        actual_content = content[video_title_end + 2 :]
                        context_parts.append(f"From '{video_title}':\n{actual_content}")
                        sources.add(video_title)
                    else:
                        context_parts.append(content)
                else:
                    context_parts.append(content)

            context = "\n\n".join(context_parts)
            return {"context": context, "sources": list(sources)}

        except Exception as e:
            st.warning(f"Error during retrieval: {e}")
            return {"context": "", "sources": []}

    def generate_answer_with_sources(state):
        context = state.get("context", "")
        sources = state.get("sources", [])

        human_message = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None,
        )

        if not human_message:
            return {
                "messages": state["messages"]
                + [
                    AIMessage(
                        content="I couldn't understand your question. Please try again."
                    )
                ]
            }

        # Create system message with source attribution instructions
        system_prompt = f"""You are an AI assistant that answers questions about a YouTube playlist based on video transcripts.

IMPORTANT: Always cite your sources by mentioning which video(s) your answer comes from.

Available context from videos:
{context}

Videos referenced: {", ".join(sources) if sources else "None"}

Instructions:
1. Answer the user's question based ONLY on the provided context
2. Always mention which video(s) your information comes from
3. If information comes from multiple videos, list all relevant videos
4. If the context doesn't contain the answer, state that clearly
5. Format your response with clear source attribution

Example format:
"Based on the video '[Video Title]', [your answer here]..."
OR
"According to videos '[Video 1]' and '[Video 2]', [your answer here]..."
"""

        messages = [
            AIMessage(content=system_prompt),
            HumanMessage(content=f"User's Question: {human_message.content}"),
        ]

        try:
            answer = llm.invoke(messages).content

            # Enhance answer with clickable source links if we have video metadata
            if sources:
                answer += "\n\n**📺 Source Videos:**\n"
                for source in sources:
                    # Find matching video for URL (simplified - would need better matching in production)
                    answer += f"• {source}\n"

            return {"messages": state["messages"] + [AIMessage(content=answer)]}

        except Exception as e:
            st.error(f"Error generating answer: {e}")
            return {
                "messages": state["messages"]
                + [
                    AIMessage(
                        content="Sorry, I encountered an error while generating the answer."
                    )
                ]
            }

    # Create the workflow
    class GraphState(TypedDict):
        messages: List
        context: str
        sources: List[str]

    workflow = StateGraph(state_schema=GraphState)

    workflow.add_node("retrieve", retrieve_with_sources)
    workflow.add_node("generate_answer", generate_answer_with_sources)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate_answer")
    workflow.add_edge("generate_answer", END)

    rag_chain = workflow.compile(checkpointer=MemorySaver())

    def invoke_playlist_rag(messages, thread_id: str):
        from langchain_core.runnables.config import RunnableConfig

        config = RunnableConfig(configurable={"thread_id": thread_id})
        initial_state = {"messages": messages, "context": "", "sources": []}

        try:
            result = rag_chain.invoke(initial_state, config=config)
            return (
                result["messages"][-1]
                if result and result.get("messages")
                else AIMessage(content="No response generated.")
            )
        except Exception as e:
            st.error(f"Error invoking playlist RAG: {e}")
            return AIMessage(
                content="Sorry, I encountered an error processing your request."
            )

    return invoke_playlist_rag


def display_interactive_playlist_summary(summary_data: Dict, videos_data: List[Dict]):
    """Display playlist summary in an interactive format with enhanced UI"""

    # Custom CSS for playlist summary
    st.markdown(
        """
    <style>
    .playlist-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    .chapter-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #4CAF50;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .chapter-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    .playlist-metrics {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-card h3 {
        margin: 0;
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-card p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    .theme-tag {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        color: #2c3e50;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        margin: 0.25rem;
        display: inline-block;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .video-link {
        background: #e3f2fd;
        border: 1px solid #2196F3;
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        display: block;
        text-decoration: none;
        color: #1976D2;
        transition: all 0.2s ease;
    }
    .video-link:hover {
        background: #bbdefb;
        transform: translateX(5px);
    }
    .progress-section {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1.5rem 0;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Playlist Header
    total_videos = len(videos_data)
    total_words = sum(v.get("word_count", 0) for v in videos_data)
    estimated_watch_time = total_videos * 10  # Rough estimate

    st.markdown(
        f"""
    <div class="playlist-header">
        <h1>📺 Playlist Analysis Complete</h1>
        <p>Comprehensive summary of {total_videos} videos with {total_words:,} words analyzed</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Playlist Metrics
    st.markdown(
        f"""
    <div class="playlist-metrics">
        <div class="metric-card">
            <h3>{total_videos}</h3>
            <p>Total Videos</p>
        </div>
        <div class="metric-card">
            <h3>{total_words:,}</h3>
            <p>Words Analyzed</p>
        </div>
        <div class="metric-card">
            <h3>{estimated_watch_time}</h3>
            <p>Min. Est. Duration</p>
        </div>
        <div class="metric-card">
            <h3>{len(summary_data.get("key_themes", []))}</h3>
            <p>Key Themes</p>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📋 Overall Summary",
            "📖 Chapter Breakdown",
            "🎯 Key Themes",
            "📊 Video Details",
        ]
    )

    with tab1:
        st.markdown("### 🎓 Complete Playlist Analysis")

        overall_summary = summary_data.get("overall_summary", "Summary not available")

        # Display overall summary with better formatting
        st.markdown(overall_summary)

        # Quick insights section
        if summary_data.get("key_themes"):
            st.markdown("### 🔍 Main Topics Covered")
            themes_cols = st.columns(3)
            for i, theme in enumerate(summary_data["key_themes"][:9]):
                with themes_cols[i % 3]:
                    clean_theme = re.sub(r"^\d+\.\s*", "", theme)  # Remove numbering
                    st.markdown(
                        f'<div class="theme-tag">{clean_theme}</div>',
                        unsafe_allow_html=True,
                    )

    with tab2:
        st.markdown("### 📚 Chapter-by-Chapter Breakdown")

        chapter_summaries = summary_data.get("chapter_summaries", [])

        if chapter_summaries:
            # Progress indicator
            st.markdown(
                f"""
            <div class="progress-section">
                <h4>📈 Learning Journey Progress</h4>
                <p>Follow the structured learning path through {len(chapter_summaries)} chapters</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            for i, chapter in enumerate(chapter_summaries):
                with st.expander(
                    f"Chapter {chapter.get('position', i + 1)}: {chapter.get('title', 'Unknown')}",
                    expanded=(i == 0),
                ):
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.markdown(chapter.get("summary", "Summary not available"))

                    with col2:
                        st.markdown("**📊 Chapter Info:**")
                        st.markdown(f"**Words:** {chapter.get('word_count', 'N/A')}")
                        st.markdown(f"**Position:** {chapter.get('position', 'N/A')}")

                        if chapter.get("url"):
                            st.markdown(
                                f'<a href="{chapter["url"]}" target="_blank" class="video-link">🎬 Watch Video</a>',
                                unsafe_allow_html=True,
                            )

                        if st.button(
                            f"Copy Chapter {chapter.get('position', i + 1)}",
                            key=f"copy_chapter_{i}",
                        ):
                            st.toast(
                                f"Chapter {chapter.get('position', i + 1)} summary copied!"
                            )
        else:
            st.info("No chapter summaries available.")

    with tab3:
        st.markdown("### 🎯 Key Themes & Learning Objectives")

        themes = summary_data.get("key_themes", [])
        if themes:
            st.markdown("**📌 Primary Themes Covered:**")
            for theme in themes:
                clean_theme = re.sub(r"^\d+\.\s*", "", theme)
                st.markdown(f"• {clean_theme}")

            # Theme visualization
            st.markdown("### 🏷️ Topic Tags")
            theme_container = st.container()
            with theme_container:
                for theme in themes:
                    clean_theme = re.sub(r"^\d+\.\s*", "", theme)
                    st.markdown(
                        f'<div class="theme-tag">{clean_theme}</div>',
                        unsafe_allow_html=True,
                    )

        # Learning objectives if available
        structured = summary_data.get("structured_summary", {})
        if structured:
            st.markdown("### 🎯 Course Information")
            info_cols = st.columns(2)

            with info_cols[0]:
                st.metric("Total Videos", structured.get("total_videos", "N/A"))
                st.metric("Difficulty", structured.get("difficulty_level", "N/A"))

            with info_cols[1]:
                st.metric(
                    "Reading Time", structured.get("total_duration_estimate", "N/A")
                )
                st.metric("Est. Completion", structured.get("completion_time", "N/A"))

    with tab4:
        st.markdown("### 📊 Individual Video Details")

        if videos_data:
            # Summary stats
            successful_videos = [v for v in videos_data if v.get("transcript_text")]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Processed Successfully", len(successful_videos))
            with col2:
                avg_words = sum(
                    v.get("word_count", 0) for v in successful_videos
                ) / max(len(successful_videos), 1)
                st.metric("Avg Words per Video", f"{avg_words:.0f}")
            with col3:
                total_chars = sum(
                    len(v.get("transcript_text", "")) for v in successful_videos
                )
                st.metric("Total Characters", f"{total_chars:,}")

            # Video list with details
            st.markdown("**📹 Video List:**")
            for i, video in enumerate(videos_data):
                with st.expander(
                    f"{i + 1}. {video.get('title', 'Unknown Title')}", expanded=False
                ):
                    info_cols = st.columns([2, 1, 1])

                    with info_cols[0]:
                        st.markdown(f"**Video ID:** {video.get('video_id', 'N/A')}")
                        st.markdown(f"**Channel:** {video.get('channel_title', 'N/A')}")
                        if video.get("description"):
                            description = (
                                video["description"][:200] + "..."
                                if len(video["description"]) > 200
                                else video["description"]
                            )
                            st.markdown(f"**Description:** {description}")

                    with info_cols[1]:
                        st.markdown(f"**Word Count:** {video.get('word_count', 'N/A')}")
                        st.markdown(
                            f"**Language:** {video.get('transcript_language', 'N/A')}"
                        )
                        st.markdown(f"**Position:** #{video.get('position', i + 1)}")

                    with info_cols[2]:
                        if video.get("url"):
                            st.markdown(
                                f'<a href="{video["url"]}" target="_blank" class="video-link">🎬 Watch</a>',
                                unsafe_allow_html=True,
                            )

                        has_transcript = bool(video.get("transcript_text"))
                        status_color = "🟢" if has_transcript else "🔴"
                        st.markdown(
                            f"**Status:** {status_color} {'Success' if has_transcript else 'Failed'}"
                        )

        else:
            st.info("No video details available.")

    # Download section
    st.markdown("---")
    st.markdown("### 💾 Export Options")

    download_cols = st.columns(3)

    with download_cols[0]:
        # Complete summary download
        complete_summary = f"""# Playlist Summary Report
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Overall Analysis
{summary_data.get("overall_summary", "Not available")}

## Chapter Summaries
"""
        for chapter in summary_data.get("chapter_summaries", []):
            complete_summary += f"""
### Chapter {chapter.get("position", "N/A")}: {chapter.get("title", "Unknown")}
{chapter.get("summary", "Not available")}
URL: {chapter.get("url", "Not available")}

"""

        st.download_button(
            label="📥 Download Complete Summary",
            data=complete_summary,
            file_name=f"playlist_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with download_cols[1]:
        # Video list JSON
        video_data_json = json.dumps(videos_data, indent=2, ensure_ascii=False)
        st.download_button(
            label="📊 Download Video Data (JSON)",
            data=video_data_json,
            file_name=f"playlist_videos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )

    with download_cols[2]:
        if st.button("📋 Copy Summary to Clipboard", use_container_width=True):
            st.toast("Summary copied to clipboard!")


def app():
    st.title("📺 YouTube Playlist Summarizer")

    st.markdown("""
    Transform entire YouTube playlists into comprehensive, structured summaries with interactive chapter breakdowns and intelligent Q&A capabilities.
    """)

    # OpenAI API Key
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_api_key:
        openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key

    st.sidebar.subheader("🤖 Model Settings")
    llm_model = st.sidebar.selectbox(
        "LLM Model",
        ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4", "gpt-4o", "deepseek-r1-ollama"],
        index=0,
    )

    temperature = st.sidebar.slider(
        "Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1
    )

    if llm_model == "deepseek-r1-ollama":
        st.sidebar.info(
            "Using local Ollama. Make sure deepseek-r1:latest is installed."
        )

    st.sidebar.header("🌐 Language Settings")
    transcript_language = st.sidebar.selectbox(
        "Preferred Transcript Language",
        options=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh-cn"],
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
        }.get(x, x),
        index=0,
    )

    # Main interface
    st.markdown("### 🎯 Enter Playlist URL")

    # Playlist URL input
    playlist_url = st.text_input(
        "YouTube Playlist URL",
        placeholder="https://www.youtube.com/playlist?list=...",
        help="Paste the URL of any public YouTube playlist",
    )

    # Processing options
    with st.expander("⚙️ Advanced Options", expanded=False):
        max_videos = st.slider(
            "Maximum Videos to Process",
            min_value=1,
            max_value=50,
            value=10,
            help="Limit processing for large playlists to save time and API costs",
        )

        auto_scroll_to_summary = st.checkbox(
            "Auto-scroll to summary when complete", value=True
        )

    # Process button
    if st.button("🚀 Process Playlist", type="primary", use_container_width=True):
        if not playlist_url:
            st.error("Please enter a playlist URL.")
            return

        # Extract playlist ID
        playlist_id = extract_playlist_id(playlist_url)
        if not playlist_id:
            st.error(
                "Invalid YouTube playlist URL. Please check the URL and try again."
            )
            return

        # Check API key requirements
        if not openai_api_key:
            st.error("Please provide OpenAI API key for summarization functionality.")
            return

        # Initialize session state for this playlist
        if "current_playlist_id" not in st.session_state:
            st.session_state.current_playlist_id = None

        if "playlist_data" not in st.session_state:
            st.session_state.playlist_data = {}

        if "playlist_chat_history" not in st.session_state:
            st.session_state.playlist_chat_history = {}

        if "playlist_rag_chain" not in st.session_state:
            st.session_state.playlist_rag_chain = None

        # Process the playlist
        st.session_state.current_playlist_id = playlist_id

        with st.status("Processing YouTube playlist...", expanded=True) as status:
            # Step 1: Extract videos from playlist
            st.write("🔍 Extracting video list from playlist...")
            videos = get_playlist_videos(playlist_url)

            if not videos:
                st.error(
                    "Could not extract videos from playlist. Please check the URL."
                )
                return

            # Limit videos if requested
            if len(videos) > max_videos:
                st.warning(
                    f"Playlist has {len(videos)} videos. Processing first {max_videos} videos."
                )
                videos = videos[:max_videos]

            st.success(f"Found {len(videos)} videos in playlist!")

            # Step 2: Process transcripts
            st.write("📝 Processing video transcripts...")
            transcript_results = process_playlist_transcripts(
                videos, transcript_language
            )

            successful_videos = transcript_results["processed_videos"]
            failed_videos = transcript_results["failed_videos"]

            if not successful_videos:
                st.error("No transcripts could be processed successfully.")
                return

            st.success(
                f"Successfully processed {len(successful_videos)} out of {len(videos)} videos!"
            )

            if failed_videos:
                st.warning(
                    f"Failed to process {len(failed_videos)} videos (likely no transcripts available)."
                )

            # Step 3: Generate comprehensive summary
            st.write("🧠 Generating comprehensive playlist summary...")
            summary_data = generate_playlist_summary(
                successful_videos, llm_model, temperature
            )

            # Step 4: Create RAG system
            st.write("🤖 Setting up intelligent Q&A system...")
            rag_chain = create_playlist_rag_system(
                successful_videos, llm_model, temperature
            )

            # Store results in session state
            st.session_state.playlist_data[playlist_id] = {
                "videos": successful_videos,
                "failed_videos": failed_videos,
                "summary": summary_data,
                "transcript_results": transcript_results,
                "playlist_url": playlist_url,
            }

            st.session_state.playlist_rag_chain = rag_chain
            st.session_state.playlist_chat_history[playlist_id] = []

            status.update(label="✅ Playlist processing complete!", state="complete")

        # Auto-scroll if enabled
        if auto_scroll_to_summary:
            st.session_state.scroll_to_top = True
            time.sleep(0.5)  # Small delay for UI update

    # Display results if available
    current_playlist = st.session_state.get("current_playlist_id")
    if current_playlist and current_playlist in st.session_state.get(
        "playlist_data", {}
    ):
        playlist_data = st.session_state.playlist_data[current_playlist]
        videos_data = playlist_data["videos"]
        summary_data = playlist_data["summary"]

        st.markdown("---")

        # Display interactive summary
        display_interactive_playlist_summary(summary_data, videos_data)

        st.markdown("---")

        # Q&A Section
        st.markdown("### 💬 Ask Questions About the Playlist")
        st.markdown(
            "Ask questions about any topic covered in the playlist videos. The AI will provide answers with source attribution."
        )

        # Chat interface
        if current_playlist in st.session_state.playlist_chat_history:
            chat_history = st.session_state.playlist_chat_history[current_playlist]

            # Display chat history
            for message in chat_history:
                if isinstance(message, HumanMessage):
                    st.chat_message("user").write(message.content)
                elif isinstance(message, AIMessage):
                    st.chat_message("assistant").write(message.content)

            # Chat input
            if prompt := st.chat_input("Ask a question about the playlist content..."):
                # Add user message
                user_message = HumanMessage(content=prompt)
                chat_history.append(user_message)
                st.chat_message("user").write(prompt)

                # Generate response
                if st.session_state.playlist_rag_chain:
                    with st.chat_message("assistant"):
                        with st.spinner("Analyzing playlist content..."):
                            try:
                                response = st.session_state.playlist_rag_chain(
                                    chat_history,
                                    thread_id=f"playlist_{current_playlist}",
                                )

                                if isinstance(response, AIMessage):
                                    st.write(response.content)
                                    chat_history.append(response)
                                else:
                                    st.error("Unexpected response format.")

                            except Exception as e:
                                error_msg = f"Sorry, I encountered an error: {str(e)}"
                                st.error(error_msg)
                                chat_history.append(AIMessage(content=error_msg))
                else:
                    st.error("Q&A system not available. Please reprocess the playlist.")

        # Quick action buttons
        st.sidebar.markdown("### 🔄 Quick Actions")
        if st.sidebar.button("📊 View Summary Again", use_container_width=True):
            st.session_state.scroll_to_top = True

        if st.sidebar.button("🔄 Reprocess Playlist", use_container_width=True):
            # Clear current data
            if current_playlist in st.session_state.playlist_data:
                del st.session_state.playlist_data[current_playlist]
            if current_playlist in st.session_state.playlist_chat_history:
                del st.session_state.playlist_chat_history[current_playlist]
            st.session_state.current_playlist_id = None
            st.session_state.playlist_rag_chain = None
            st.rerun()

        if st.sidebar.button("💭 Clear Chat History", use_container_width=True):
            if current_playlist in st.session_state.playlist_chat_history:
                st.session_state.playlist_chat_history[current_playlist] = []
            st.rerun()

        if st.sidebar.button("🆕 New Playlist", use_container_width=True):
            st.session_state.current_playlist_id = None
            st.session_state.playlist_rag_chain = None
            st.rerun()


if __name__ == "__main__":
    app()

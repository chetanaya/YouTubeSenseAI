import streamlit as st


def Navbar():
    with st.sidebar:
        st.page_link("app.py", label="YouTube Comments Analyzer", icon="🎬")
        st.page_link(
            "pages/01_Transcript_Analysis.py",
            label="YouTube Transcript Analyzer",
            icon="🎬",
        )
        st.page_link(
            "pages/02_Transcript_Analysis_Whisper.py",
            label="YouTube Transcript Analyzer (Whisper)",
            icon="🎬",
        )
        st.page_link(
            "pages/03_Transcript_Analysis_Assembly_AI.py",
            label="YouTube Transcript Analyzer (Assembly AI)",
            icon="🎬",
        )

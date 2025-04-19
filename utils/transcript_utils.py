import re
import streamlit as st
from typing import List
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict


def extract_video_id(url):
    """Extract YouTube video ID from various URL formats"""
    patterns = [
        r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([^&\s]+)",
        r"(?:https?:\/\/)?(?:www\.)?youtu\.be\/([^\?\s]+)",
        r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([^\?\s]+)",
        r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/v\/([^\?\s]+)",
        r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/shorts\/([^\?\s]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def summarize_transcript(
    transcript_text, llm_model="gpt-4o-mini", custom_prompt=None, temperature=0
):
    """Generate a summary and potential Q&A pairs of the transcript using LLM"""
    if not transcript_text or len(transcript_text) < 50:
        return {"summary": "Transcript is too short to summarize.", "qa_pairs": ""}

    llm = ChatOpenAI(temperature=temperature, model=llm_model)
    output_parser = StrOutputParser()

    max_length = 15000
    if len(transcript_text) > max_length:
        transcript_text = transcript_text[:max_length] + "..."

    # --- Summary Generation ---
    # Base prompt template
    summary_prompt_template = """
    Analyze the following YouTube video transcript and generate a comprehensive yet concise summary.

    Transcript: {transcript}

    **Comprehensive Summary Generation Process:**

    1.  **Core Topic & Purpose:** Identify the main subject and the video's primary goal (e.g., inform, teach, persuade, review, entertain).
    2.  **Context & Audience:** Determine the setting (interview, lecture, tutorial, vlog) and intended audience, if discernible.
    3.  **Key Figures:** Note any important speakers, interviewees, or mentioned individuals.
    4.  **Structure & Flow:** Understand how the video is organized (e.g., introduction, key sections, conclusion; chronological order; problem/solution).
    5.  **Main Points & Arguments:** Extract the key messages, central arguments, steps demonstrated, or significant events discussed.
    6.  **Supporting Details & Keywords:** Include crucial data, evidence, examples, or specific terminology/keywords central to the topic.
    7.  **Actions, Recommendations & Solutions:** Capture any calls to action, solutions proposed, tips, or specific advice given.
    8.  **Broader Themes & Implications:** Recognize underlying themes, the video's relevance or timeliness, and any future predictions or outlooks mentioned.
    9.  **Key Visuals/Audio (If Inferrable):** Briefly note if the transcript implies reliance on essential visuals (graphs, demonstrations) or specific audio cues, if central to understanding.

    **Formatting and Style Guidelines:**

    *   **Objectivity:** Report neutrally what the transcript says. Attribute claims or opinions to the video's speakers (e.g., "The speaker argues...").
    *   **Conciseness:** Focus on the most valuable information for someone who hasn't seen the video. Omit fluff, excessive detail, and repetitive points.
    *   **Clarity:** Use clear headings/subheadings (Markdown). Employ bullet points for lists (key takeaways, steps). Use brief paragraphs for context.
    *   **Emphasis:** **Bold** critical terms, conclusions, or key takeaways.
    """

    # Add custom prompt as additional instructions when provided
    if custom_prompt:
        summary_prompt_template += f"\n\n**Additional Instructions:**\n{custom_prompt}"

    # Add the generated summary header
    summary_prompt_template += "\n\n**Generated Summary:**"

    summary_prompt = ChatPromptTemplate.from_template(summary_prompt_template)
    summary_chain = summary_prompt | llm | output_parser

    # --- Q&A Generation ---
    qa_prompt_template = """
    Based *only* on the provided YouTube video transcript, generate 3-5 insightful questions that a viewer might have after watching. For each question, provide a concise answer derived *directly* from the transcript content. Do not add information not present in the transcript.

    Transcript: {transcript}

    Format the Q&A pairs clearly:
    **Q1:** [Question 1]?
    **A1:** [Answer 1 derived from transcript].

    **Q2:** [Question 2]?
    **A2:** [Answer 2 derived from transcript].
    ...

    **Generated Q&A Pairs:**
    """
    qa_prompt = ChatPromptTemplate.from_template(qa_prompt_template)
    qa_chain = qa_prompt | llm | output_parser

    summary_result = "Error generating summary."
    qa_result = "Error generating Q&A."

    try:
        # Invoke summary chain
        summary_result = summary_chain.invoke({"transcript": transcript_text})
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        summary_result = "Unable to generate summary due to an error."

    try:
        # Invoke Q&A chain
        qa_result = qa_chain.invoke({"transcript": transcript_text})
    except Exception as e:
        st.error(f"Error generating Q&A: {str(e)}")
        qa_result = "Unable to generate Q&A pairs due to an error."

    return {"summary": summary_result.strip(), "qa_pairs": qa_result.strip()}


def translate_text(text, target_language, llm_model="gpt-4o-mini", temperature=0):
    """Translate text to the target language using LLM"""
    if not text:
        return "No text to translate."

    llm = ChatOpenAI(temperature=temperature, model=llm_model)

    max_length = 15000
    if len(text) > max_length:
        text = text[:max_length] + "..."

    prompt = ChatPromptTemplate.from_template(
        """Translate the following text to {language}. Maintain the formatting and structure as much as possible:

        {text}

        Translation:
        """
    )

    chain = prompt | llm

    try:
        response = chain.invoke({"language": target_language, "text": text})
        return response.content
    except Exception as e:
        st.error(f"Error translating text: {str(e)}")
        return "Unable to translate text due to an error."


def create_rag_system(
    transcript_text: str, llm_model: str = "gpt-4o-mini", temperature=0
):
    """Create a RAG system with chat history for transcript Q&A"""
    if not transcript_text:
        st.error("Cannot create RAG system: Transcript text is empty.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(transcript_text)

    if not chunks:
        st.error("Cannot create RAG system: Failed to split transcript into chunks.")
        return None

    try:
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    except Exception as e:
        st.error(f"Failed to create vector store for RAG: {e}")
        return None

    llm = ChatOpenAI(temperature=temperature, model=llm_model)

    def retrieve(state):
        query = (
            state["messages"][-1].content
            if isinstance(state["messages"][-1], HumanMessage)
            else ""
        )
        if not query:
            return {"context": ""}  # Avoid error if query is empty
        try:
            docs = retriever.invoke(query)
            context = "\n\n".join([doc.page_content for doc in docs])
            return {"context": context}
        except Exception as e:
            st.warning(f"Error during retrieval: {e}")
            return {"context": ""}

    def generate_answer(state):
        context = state.get("context", "")
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

        messages = [
            AIMessage(
                content="I am an AI assistant that helps answer questions about the YouTube video transcript."
            ),
            HumanMessage(
                content=f"""First, analyze the provided context from the video transcript and identify 2-3 key potential questions and their answers that could be derived from this specific context.

            Context:
            {context}

            Potential Q&A based on context:
            1. Q: ... A: ...
            2. Q: ... A: ...
            3. Q: ... A: ...

            Now, answer the user's question based *only* on the provided context. If the context doesn't contain the answer, state that clearly. Do not make up information.

            User's Question: {human_message.content}

            Final Answer:"""
            ),
        ]
        try:
            answer = llm.invoke(messages).content
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

    class GraphState(TypedDict):
        messages: List
        context: str

    workflow = StateGraph(state_schema=GraphState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate_answer", generate_answer)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate_answer")
    workflow.add_edge("generate_answer", END)

    rag_chain = workflow.compile(checkpointer=MemorySaver())

    def invoke_rag_chain(messages, thread_id: str):
        config = {"configurable": {"thread_id": thread_id}}
        initial_state = {"messages": messages, "context": ""}
        try:
            result = rag_chain.invoke(initial_state, config=config)
            return (
                result["messages"][-1]
                if result and result.get("messages")
                else AIMessage(content="No response generated.")
            )
        except Exception as e:
            st.error(f"Error invoking RAG chain: {e}")
            return AIMessage(
                content="Sorry, I encountered an error processing your request."
            )

    return invoke_rag_chain

"""
YouTube Video Q&A Bot
=====================
Paste a YouTube URL → auto-fetches transcript → chat with the video using RAG.
Built with: youtube-transcript-api · LangChain · ChromaDB · OpenAI
"""

import streamlit as st
from rag_pipeline import build_qa_chain, ask_question
from transcript_fetcher import fetch_transcript, extract_video_id

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YouTube Q&A Bot",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

    .main { background-color: #0d0d0d; }
    h1, h2, h3 { font-family: 'Space Mono', monospace; color: #ffffff; }
    p, label, div { font-family: 'Inter', sans-serif; }

    .chat-user {
        background: #1e1e2e;
        border-left: 3px solid #7c3aed;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        color: #e2e8f0;
    }
    .chat-assistant {
        background: #1a1a2e;
        border-left: 3px solid #06b6d4;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        color: #e2e8f0;
    }
    .timestamp-badge {
        background: #7c3aed22;
        border: 1px solid #7c3aed55;
        color: #a78bfa;
        font-size: 12px;
        padding: 2px 8px;
        border-radius: 20px;
        font-family: 'Space Mono', monospace;
        margin-right: 4px;
        text-decoration: none;
        display: inline-block;
    }
    .video-info-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .stTextInput > div > div > input {
        background-color: #1e1e2e !important;
        color: white !important;
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
    }
    .status-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        font-family: 'Space Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    openai_key = st.text_input("OpenAI API Key", type="password", help="Required for embeddings + LLM")
    st.markdown("---")
    st.markdown("### 📖 How it works")
    st.markdown("""
1. **Paste** a YouTube URL  
2. **Transcript** is auto-fetched  
3. **RAG pipeline** chunks + embeds it  
4. **Ask anything** — get answers with timestamps  
    """)
    st.markdown("---")
    st.markdown("### 🛠️ Stack")
    st.markdown("`youtube-transcript-api` · `LangChain` · `ChromaDB` · `OpenAI`")

    if st.button("🗑️ Clear Chat & Reset", use_container_width=True):
        for key in ["messages", "qa_chain", "video_id", "transcript_loaded"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()


# ─── Session State Init ───────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "transcript_loaded" not in st.session_state:
    st.session_state.transcript_loaded = False
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("# 🎬 YouTube Q&A Bot")
st.markdown("*Paste a video URL and start chatting with its content.*")
st.markdown("---")


# ─── URL Input ────────────────────────────────────────────────────────────────
col1, col2 = st.columns([4, 1])
with col1:
    youtube_url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        label_visibility="collapsed"
    )
with col2:
    load_button = st.button("🚀 Load Video", use_container_width=True, type="primary")


# ─── Load Transcript + Build RAG ─────────────────────────────────────────────
if load_button and youtube_url:
    if not openai_key:
        st.error("⚠️ Please enter your OpenAI API key in the sidebar.")
    else:
        with st.spinner("📥 Fetching transcript..."):
            try:
                video_id = extract_video_id(youtube_url)
                transcript_chunks, metadata = fetch_transcript(video_id)

                st.session_state.video_id = video_id
                st.session_state.video_title = metadata.get("title", "YouTube Video")
                st.session_state.transcript_loaded = False  # reset while building

                with st.spinner("🧠 Building RAG pipeline (chunking + embedding)..."):
                    qa_chain = build_qa_chain(transcript_chunks, openai_key)
                    st.session_state.qa_chain = qa_chain
                    st.session_state.transcript_loaded = True
                    st.session_state.messages = []

                st.success(f"✅ Ready! Loaded **{len(transcript_chunks)}** transcript chunks.")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")


# ─── Video Info Card ──────────────────────────────────────────────────────────
if st.session_state.transcript_loaded and "video_id" in st.session_state:
    vid_id = st.session_state.video_id
    st.markdown(f"""
    <div class="video-info-card">
        <b>🎥 Video loaded</b><br/>
        <a href="https://youtube.com/watch?v={vid_id}" target="_blank" style="color:#a78bfa;">
            youtube.com/watch?v={vid_id}
        </a>
        &nbsp;&nbsp;
        <span class="status-pill" style="background:#05966922; border:1px solid #059669; color:#34d399;">
            ● READY
        </span>
    </div>
    """, unsafe_allow_html=True)


# ─── Chat Interface ───────────────────────────────────────────────────────────
if st.session_state.transcript_loaded:
    st.markdown("### 💬 Chat with the Video")

    # Display conversation history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">🙋 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            answer_html = msg["content"].replace("\n", "<br>")
            timestamps_html = ""
            if msg.get("timestamps"):
                for ts in msg["timestamps"]:
                    url = f"https://youtube.com/watch?v={st.session_state.video_id}&t={ts['start_seconds']}s"
                    timestamps_html += f'<a href="{url}" target="_blank" class="timestamp-badge">⏱ {ts["label"]}</a>'
            st.markdown(
                f'<div class="chat-assistant">🤖 {answer_html}'
                + (f'<br><br><b>Sources:</b> {timestamps_html}' if timestamps_html else '')
                + '</div>',
                unsafe_allow_html=True
            )

    # Question input
    question = st.chat_input("Ask anything about this video...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner("🔍 Searching transcript..."):
            if st.session_state.qa_chain is None:
                st.error("QA chain not ready. Please load a video first.")
                st.stop()
            result = ask_question(st.session_state.qa_chain, question)

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "timestamps": result.get("timestamps", [])
        })
        st.rerun()

else:
    # Placeholder state
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px; color: #4b5563;">
        <div style="font-size: 48px; margin-bottom: 16px;">🎬</div>
        <div style="font-size: 18px; font-family: 'Space Mono', monospace;">
            Paste a YouTube URL above to get started
        </div>
        <div style="font-size: 14px; margin-top: 8px; color: #374151;">
            Works with lectures, podcasts, interviews, tutorials — any video with captions
        </div>
    </div>
    """, unsafe_allow_html=True)
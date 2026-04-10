# 🎬 YouTube Video Q&A Bot

> Paste any YouTube URL → auto-fetch transcript → chat with the video. 
Answers include clickable timestamps linking back to the exact moment in the video.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)

---

## 📈 Results

Benchmarked on a 77-minute video — run `evaluate.py` to reproduce on any video:

| Metric | Baseline | This System | Δ |
|---|---|---|---|
| Tokens per query | ~19,909 (full transcript) | ~1,292 (RAG) | **−93.5%** |
| Retrieval diversity (pairwise sim ↓) | 0.285 (naive Top-K) | 0.243 (MMR) | **−14.7% redundancy** |
| Average query latency | — | **0.56s** | p95: 0.72s |
| Answer consistency (temp=0) | — | **100%** | 3 runs, same question |

---

## 🧠 How It Works

```
YouTube URL
    │
    ▼
[youtube-transcript-api] ── Fetch transcript with timestamps
    │
    ▼
[LangChain TextSplitter] ── Chunk into ~1000 token segments (60s time windows)
    │
    ▼
[HuggingFace all-MiniLM-L6-v2] ── Embed each chunk locally (free, no API)
    │
    ▼
[ChromaDB] ── Store vectors in-memory
    │
    ▼
[User Question] ── MMR Retrieval → Top-5 chunks → Groq LLaMA 3.1 → Answer + Timestamps

```

**Key design decisions:**

**Timestamped chunking** — chunks by 60s time windows (not character count), so every answer maps to a clickable video timestamp.

**MMR retrieval** — Maximal Marginal Relevance retrieves diverse chunks from across the video instead of 5 near-identical chunks from the same segment. Measured −14.7% pairwise similarity vs naive Top-K.

**RAG over full-transcript prompting** — embeds once, retrieves ~5 relevant chunks per query instead of sending the entire transcript. Measured −93.5% token reduction per query.

**HuggingFace embeddings** — runs locally, zero API cost, cached after first download (~80MB).

**Groq LLM** — sub-second inference via LPU hardware. 0.56s avg latency vs 2-4s typical for cloud LLMs.

---

## 🚀 Getting Started

### 1. Clone and set up

```bash
git clone https://github.com/YOUR_USERNAME/youtube-qa-bot.git
cd youtube-qa-bot

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Get a Groq API key
Sign up at [console.groq.com](https://console.groq.com) — free, no credit card needed.

### 3. Run

```bash
streamlit run app.py
```

Enter your Groq API key in the sidebar, paste a YouTube URL, and start asking questions.

---

## 📊 Run the Evaluation

```bash
python evaluate.py --video_id YOUR_VIDEO_ID --api_key YOUR_GROQ_KEY
```

Measures and prints resume-ready metrics:
- MMR vs Top-K retrieval diversity
- RAG vs full-transcript token cost
- p50 / p95 query latency
- Answer consistency at temperature=0

Results saved to `eval_results.json`.

---

## 📁 Project Structure

```
youtube-qa-bot/
├── app.py                  # Streamlit UI — chat interface + timestamp badges
├── transcript_fetcher.py   # Time-window chunking with timestamp metadata
├── rag_pipeline.py         # LangChain + ChromaDB + MMR retrieval chain
├── evaluate.py             # Benchmarking suite — produces resume metrics
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Component | Tool | Why |
|---|---|---|
| Transcript | `youtube-transcript-api` | Manual + auto-generated caption support |
| Orchestration | `LangChain` (LCEL) | Modular, composable RAG pipeline |
| Embeddings | `HuggingFace all-MiniLM-L6-v2` | Free, local, no API quota |
| Vector Store | `ChromaDB` | Fast in-memory retrieval, no infra needed |
| LLM | `Groq llama-3.1-8b-instant` | Sub-second inference, free tier |
| UI | `Streamlit` | Rapid prototype, deployable in 1 click |

---

## 📄 License

MIT [LICENSE]

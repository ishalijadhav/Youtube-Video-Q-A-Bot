import argparse
import time
import json
import statistics
from typing import List, Dict

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from transcript_fetcher import fetch_transcript
from rag_pipeline import build_qa_chain, ask_question


# Evaluation Questions
EVAL_QUESTIONS = [
    "What is the main topic of this video?",
    "What are the key points discussed?",
    "What conclusions or recommendations are made?",
    "Can you summarize what was said in the first part of the video?",
    "What evidence or examples are given?",
]


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = sum(a ** 2 for a in v1) ** 0.5
    mag2 = sum(b ** 2 for b in v2) ** 0.5
    return dot / (mag1 * mag2 + 1e-9)


def avg_pairwise_similarity(embeddings: List[List[float]]) -> float:
    """Lower = more diverse retrieved chunks (better MMR performance)."""
    if len(embeddings) < 2:
        return 1.0
    sims = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sims.append(cosine_similarity(embeddings[i], embeddings[j]))
    return statistics.mean(sims)


# Experiment 1: MMR vs Naive Top-K Retrieval
def experiment_retrieval_diversity(vectorstore: Chroma, embeddings_model, questions: List[str]) -> Dict:
    print("\n📊 Experiment 1: MMR vs Naive Top-K — Source Diversity")
    print("─" * 55)

    mmr_sims = []
    topk_sims = []

    for q in questions:
        mmr_docs = vectorstore.max_marginal_relevance_search(q, k=5, fetch_k=10)
        mmr_embeddings = [embeddings_model.embed_query(d.page_content) for d in mmr_docs]
        mmr_avg_sim = avg_pairwise_similarity(mmr_embeddings)
        mmr_sims.append(mmr_avg_sim)

        topk_docs = vectorstore.similarity_search(q, k=5)
        topk_embeddings = [embeddings_model.embed_query(d.page_content) for d in topk_docs]
        topk_avg_sim = avg_pairwise_similarity(topk_embeddings)
        topk_sims.append(topk_avg_sim)

    avg_mmr = statistics.mean(mmr_sims)
    avg_topk = statistics.mean(topk_sims)
    diversity_gain = ((avg_topk - avg_mmr) / avg_topk) * 100

    print(f"  Naive Top-K avg pairwise similarity : {avg_topk:.4f}")
    print(f"  MMR avg pairwise similarity          : {avg_mmr:.4f}")
    print(f"  → Diversity improvement (MMR)        : {diversity_gain:+.1f}%")
    print("  (lower similarity = more diverse retrieved chunks = better coverage)")

    return {
        "topk_avg_similarity": round(avg_topk, 4),
        "mmr_avg_similarity": round(avg_mmr, 4),
        "diversity_gain_pct": round(diversity_gain, 1),
    }


# Experiment 2: Chunking vs Full Transcript
def experiment_token_cost(transcript_chunks: List[Dict]) -> Dict:
    print("\n📊 Experiment 2: RAG Chunking vs Full Transcript — Token Efficiency")
    print("─" * 55)

    full_transcript_text = " ".join(c["text"] for c in transcript_chunks)
    full_tokens_estimate = len(full_transcript_text.split()) * 1.3

    avg_chunk_tokens = statistics.mean(
        len(c["text"].split()) * 1.3 for c in transcript_chunks
    )
    rag_tokens_per_query = avg_chunk_tokens * 5
    reduction_pct = ((full_tokens_estimate - rag_tokens_per_query) / full_tokens_estimate) * 100

    print(f"  Full transcript tokens (est.)        : {int(full_tokens_estimate):,}")
    print(f"  RAG top-5 chunks tokens (est.)       : {int(rag_tokens_per_query):,}")
    print(f"  → Token reduction per query          : {reduction_pct:.1f}%")

    return {
        "full_transcript_tokens": int(full_tokens_estimate),
        "rag_tokens_per_query": int(rag_tokens_per_query),
        "token_reduction_pct": round(reduction_pct, 1),
    }


# Experiment 3: Latency Benchmark
def experiment_latency(qa_chain_obj: Dict, questions: List[str], n_runs: int = 3) -> Dict:
    print("\n📊 Experiment 3: End-to-End Query Latency")
    print("─" * 55)

    latencies = []
    for q in questions:
        for _ in range(n_runs):
            start = time.perf_counter()
            ask_question(qa_chain_obj, q)
            latencies.append(time.perf_counter() - start)

    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[int(len(latencies) * 0.95)]
    avg = statistics.mean(latencies)

    print(f"  Queries tested                       : {len(questions)} questions × {n_runs} runs")
    print(f"  Average latency                      : {avg:.2f}s")
    print(f"  p50 (median) latency                 : {p50:.2f}s")
    print(f"  p95 latency                          : {p95:.2f}s")

    return {
        "avg_latency_s": round(avg, 2),
        "p50_latency_s": round(p50, 2),
        "p95_latency_s": round(p95, 2),
        "total_queries_benchmarked": len(questions) * n_runs,
    }


# Experiment 4: Answer Self-Consistency
def experiment_consistency(qa_chain_obj: Dict, question: str, n_runs: int = 3) -> Dict:
    print(f"\n📊 Experiment 4: Answer Self-Consistency (temp=0)")
    print("─" * 55)

    answers = []
    for i in range(n_runs):
        result = ask_question(qa_chain_obj, question)
        answers.append(result["answer"])
        print(f"  Run {i+1}: {result['answer'][:80]}...")

    unique = len(set(answers))
    consistency = ((n_runs - unique + 1) / n_runs) * 100

    print(f"  Unique answers out of {n_runs} runs        : {unique}")
    print(f"  → Consistency rate                   : {consistency:.0f}%")

    return {
        "question_tested": question,
        "n_runs": n_runs,
        "unique_answers": unique,
        "consistency_pct": round(consistency, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate YouTube Q&A Bot")
    parser.add_argument("--video_id", required=True, help="YouTube video ID (e.g. dQw4w9WgXcQ)")
    parser.add_argument("--api_key", required=True, help="Groq API key (console.groq.com)")
    parser.add_argument("--output", default="eval_results.json", help="Output file for results")
    args = parser.parse_args()

    print(f"\n🎬 YouTube Q&A Bot — Evaluation Suite")
    print(f"   Video ID : {args.video_id}")
    print("=" * 55)

    print("\n⏳ Loading transcript and building pipeline...")
    transcript_chunks, metadata = fetch_transcript(args.video_id)
    print(f"   ✓ {len(transcript_chunks)} transcript chunks loaded")

    qa_chain_obj = build_qa_chain(transcript_chunks, args.api_key)
    vectorstore = qa_chain_obj["vectorstore"]

    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("   ✓ RAG pipeline ready")

    results = {"video_id": args.video_id, "metadata": metadata}

    results["retrieval_diversity"] = experiment_retrieval_diversity(
        vectorstore, embeddings_model, EVAL_QUESTIONS
    )
    results["token_efficiency"] = experiment_token_cost(transcript_chunks)
    results["latency"] = experiment_latency(
        qa_chain_obj, EVAL_QUESTIONS[:3], n_runs=2
    )
    results["consistency"] = experiment_consistency(
        qa_chain_obj, EVAL_QUESTIONS[0], n_runs=3
    )

    print("\n" + "=" * 55)
    print("✅ EVALUATION COMPLETE — Key Metrics for Resume")
    print("=" * 55)
    print(f"  MMR diversity gain over Top-K        : +{results['retrieval_diversity']['diversity_gain_pct']}%")
    print(f"  Token reduction vs full-transcript   : {results['token_efficiency']['token_reduction_pct']}%")
    print(f"  Avg end-to-end query latency         : {results['latency']['avg_latency_s']}s")
    print(f"  Answer consistency (temp=0)          : {results['consistency']['consistency_pct']}%")
    print()

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"💾 Full results saved to {args.output}")


if __name__ == "__main__":
    main()
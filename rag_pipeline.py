import os
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# Custom Prompt 
QA_PROMPT = PromptTemplate.from_template(
    """You are an assistant that answers questions about a YouTube video.
Use ONLY the transcript excerpts below to answer. If the answer is not in the excerpts, say so.
Be concise and direct. Do not fabricate information.

Transcript excerpts:
{context}

Question: {question}

Answer:"""
)


def build_qa_chain(transcript_chunks: List[Dict], openai_api_key: str) -> Dict[str, Any]:
    os.environ["OPENAI_API_KEY"] = openai_api_key

    # Step 1 — Convert to LangChain Documents
    documents = []
    for chunk in transcript_chunks:
        doc = Document(
            page_content=chunk["text"],
            metadata={
                "start": chunk["start"],
                "end": chunk["end"],
                "label": chunk["label"],
                "start_seconds": chunk["start_seconds"],
            }
        )
        documents.append(doc)

    # Step 2 — Optional: further split very long chunks (edge case for long videos)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=[". ", "! ", "? ", "\n", " "]
    )
    split_docs = splitter.split_documents(documents)

    # Step 3 — Embed + store in Chroma
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
    )
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        collection_name="youtube_transcript"
    )

    # Step 4 — Build LCEL chain
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )
    retriever = vectorstore.as_retriever(
        search_type="mmr",             # Maximal Marginal Relevance → diverse results
        search_kwargs={"k": 5, "fetch_k": 10}
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"])),
            source_documents=lambda x: retriever.invoke(x["question"]),
        )
        | RunnablePassthrough.assign(
            answer=QA_PROMPT | llm | StrOutputParser()
        )
    )

    return {
        "chain": chain,
        "vectorstore": vectorstore,
        "retriever": retriever,
    }


def ask_question(qa_chain_obj: Dict[str, Any], question: str) -> Dict[str, Any]:

    chain = qa_chain_obj["chain"]
    result = chain.invoke({"question": question})

    answer = result.get("answer", "").strip()
    source_docs = result.get("source_documents", [])

    # Deduplicate timestamps from source docs
    seen = set()
    timestamps = []
    for doc in source_docs:
        meta = doc.metadata
        ts_key = meta.get("start_seconds", 0)
        if ts_key not in seen:
            seen.add(ts_key)
            timestamps.append({
                "label": meta.get("label", "0:00"),
                "start_seconds": ts_key,
            })

    # Sort by video position
    timestamps.sort(key=lambda x: x["start_seconds"])

    return {
        "answer": answer,
        "timestamps": timestamps[:4],
    }
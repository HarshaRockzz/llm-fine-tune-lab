"""Qdrant vector store — index and search experiment runs semantically."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

COLLECTION_NAME = "llm_experiments"
VECTOR_SIZE = 1024  # nvidia/llama-nemotron-embed-vl-1b-v2:free output dim


def _get_qdrant_client():
    from qdrant_client import QdrantClient

    url = os.environ.get("QDRANT_URL", "https://your-cluster.qdrant.io")
    api_key = os.environ.get("QDRANT_API_KEY")
    if api_key:
        return QdrantClient(url=url, api_key=api_key)
    return QdrantClient(":memory:")  # fallback for local dev


def _run_to_text(run: dict) -> str:
    """Flatten a run dict to a searchable text document."""
    return (
        f"Run: {run['run_name']}. "
        f"Model: {run['model']}. "
        f"Method: {run['method']} with rank {run['rank']} and alpha {run['lora_alpha']}. "
        f"Learning rate: {run['lr']}, epochs: {run['epochs']}, batch size: {run['batch_size']}, "
        f"grad accum: {run['grad_accum']}, NEFTune: {run['neftune']}, scheduler: {run['scheduler']}. "
        f"Train loss: {run['train_loss_final']:.3f}, eval loss: {run['eval_loss_final']:.3f}. "
        f"MMLU accuracy: {run['mmlu_overall']:.1%}, TruthfulQA MC1: {run['truthfulqa_mc1']:.1%}, "
        f"custom accuracy: {run['custom_acc']:.1%}, judge composite: {run['judge_composite']:.1f}/10. "
        f"GPU memory: {run['gpu_mem_gb']:.1f} GB, setup time: {run['setup_time_min']} minutes, "
        f"tokens per second: {run['tokens_per_sec']}."
    )


def index_experiments(experiments: list[dict], force_reindex: bool = False) -> int:
    """Index all experiment runs into Qdrant. Returns number of indexed points."""
    from qdrant_client.models import Distance, VectorParams, PointStruct
    from src.utils.openrouter import embed

    client = _get_qdrant_client()

    # Create collection if needed
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing and not force_reindex:
        count = client.count(COLLECTION_NAME).count
        if count == len(experiments):
            logger.info(f"Qdrant: {count} points already indexed — skipping")
            return count
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    # Embed in batches of 8
    texts = [_run_to_text(r) for r in experiments]
    batch_size = 8
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vecs = embed(batch)
        embeddings.extend(vecs)
        logger.info(
            f"Embedded {min(i + batch_size, len(texts))}/{len(texts)} experiments"
        )

    points = [
        PointStruct(
            id=i,
            vector=embeddings[i],
            payload={**experiments[i], "document": texts[i]},
        )
        for i in range(len(experiments))
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    logger.info(f"Indexed {len(points)} experiments into Qdrant ({COLLECTION_NAME})")
    return len(points)


def search_experiments(query: str, top_k: int = 5) -> list[dict]:
    """Semantic search over indexed experiments."""
    from src.utils.openrouter import embed

    client = _get_qdrant_client()
    query_vec = embed([query])[0]

    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec,
        limit=top_k,
        with_payload=True,
    )
    return [{**hit.payload, "_score": round(hit.score, 4)} for hit in hits]


def rag_answer(query: str, top_k: int = 5) -> tuple[str, list[dict]]:
    """RAG: retrieve relevant experiments + generate a grounded answer via OpenRouter."""
    from src.utils.openrouter import chat

    results = search_experiments(query, top_k=top_k)
    if not results:
        return "No relevant experiments found.", []

    context = "\n\n".join(f"[{i + 1}] {r['document']}" for i, r in enumerate(results))

    system = (
        "You are an ML engineering assistant with deep expertise in LLM fine-tuning. "
        "You have access to a set of experiment results. "
        "Answer the user's question based ONLY on the provided experiment data. "
        "Be specific — reference run names, exact metrics, and hyperparameters. "
        "If the answer is not in the data, say so clearly."
    )
    prompt = f"""Based on these experiment results:

{context}

Question: {query}

Provide a precise, data-driven answer."""

    answer = chat(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        model=os.environ.get(
            "OPENROUTER_MODEL", "meta-llama/llama-3.1-70b-instruct:free"
        ),
        temperature=0.2,
        max_tokens=1024,
    )
    return answer, results

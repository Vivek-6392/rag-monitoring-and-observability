"""
Cost and token metrics.
Groq free-tier pricing approximated; update MODEL_COST_PER_1K_TOKENS as needed.
"""
import tiktoken

# Cost in USD per 1K tokens (input, output)
MODEL_COST_PER_1K_TOKENS = {
    # ✅ Active Llama models
    "llama-3.1-8b-instant":      (0.00005, 0.00008),
    "llama-3.3-70b-versatile":   (0.00059, 0.00079),

    # Mixtral
    "mixtral-8x7b-32768":        (0.00027, 0.00027),

    # Gemma2
    "gemma2-9b-it":              (0.00020, 0.00020),

    # DeepSeek
    "deepseek-r1-distill-llama-70b": (0.00075, 0.00075),
    "deepseek-r1-distill-qwen-32b":  (0.00029, 0.00029),

    # GPT-OSS
    "gpt-oss-20b":               (0.00010, 0.00010),
    "gpt-oss-120b":              (0.00015, 0.00015),

    # Groq Compound
    "groq/compound":             (0.00010, 0.00010),

    # Default fallback
    "default":                   (0.00010, 0.00010),
}


_enc = None


def _get_encoder():
    global _enc
    if _enc is None:
        try:
            _enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _enc = None
    return _enc


def count_tokens(text: str) -> int:
    enc = _get_encoder()
    if enc:
        return len(enc.encode(text))
    # rough fallback: ~4 chars per token
    return max(1, len(text) // 4)


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    inp_rate, out_rate = MODEL_COST_PER_1K_TOKENS.get(
        model, MODEL_COST_PER_1K_TOKENS["default"]
    )
    cost = (prompt_tokens / 1000) * inp_rate + (completion_tokens / 1000) * out_rate
    return round(cost, 8)


def build_metrics_record(
    *,
    query: str,
    answer: str,
    context_docs: list[str],
    model: str,
    latency_retrieval_ms: float,
    latency_generation_ms: float,
    trace_id: str,
) -> dict:
    import time, json

    prompt_text = query + "\n".join(context_docs)
    prompt_tokens = count_tokens(prompt_text)
    completion_tokens = count_tokens(answer)
    total_tokens = prompt_tokens + completion_tokens
    cost = calculate_cost(model, prompt_tokens, completion_tokens)
    total_ms = latency_retrieval_ms + latency_generation_ms

    return {
        "ts": time.time(),
        "query": query,
        "answer": answer,
        "model": model,
        "sources": json.dumps(context_docs),  # ✅ removed [:3] cap
        "latency_retrieval_ms": round(latency_retrieval_ms, 2),
        "latency_generation_ms": round(latency_generation_ms, 2),
        "latency_total_ms": round(total_ms, 2),
        "tokens_prompt": prompt_tokens,
        "tokens_completion": completion_tokens,
        "tokens_total": total_tokens,
        "cost_usd": cost,
        "trace_id": trace_id,
    }
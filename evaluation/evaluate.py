from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from mcp_suggester.catalog import get_catalog
from mcp_suggester.config import has_openai_api_key
from mcp_suggester.intent import extract_intent
from mcp_suggester.scoring import ScoredEntry, score_entries
from mcp_suggester.server import suggest_mcp_servers_impl

DATASET_PATH = Path(__file__).with_name("eval_dataset.csv")
EVAL_DEPTH = 10
TOP_N = 3


@dataclass
class EvalRecord:
    query_id: str
    user_query: str
    expected_server: str
    expected_tool: str
    filter_tags: List[str]

    @property
    def expected_pair(self) -> Tuple[str, str]:
        return (self.expected_server.lower(), self.expected_tool.lower())


def load_dataset() -> List[EvalRecord]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Evaluation dataset missing at {DATASET_PATH}")
    records: List[EvalRecord] = []
    with DATASET_PATH.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            filter_field = (row.get("filter_tags") or "").strip()
            tags = [tag.strip() for tag in filter_field.split(";") if tag.strip()]
            records.append(
                EvalRecord(
                    query_id=str(row.get("query_id") or ""),
                    user_query=row.get("user_query") or "",
                    expected_server=row.get("expected_server") or "",
                    expected_tool=row.get("expected_tool") or "",
                    filter_tags=tags,
                )
            )
    return records


def flatten_scored_entries(scored_entries: Sequence[ScoredEntry]) -> List[Tuple[str, str]]:
    flat: List[Tuple[str, str]] = []
    for scored in scored_entries[:EVAL_DEPTH]:
        flat.append((scored.entry.server_name.lower(), scored.entry.tool_name.lower()))
    return flat


def flatten_server_suggestions(suggestions: Sequence[Dict]) -> List[Tuple[str, str]]:
    flat: List[Tuple[str, str]] = []
    for server in suggestions:
        server_name = (server.get("server_name") or "").lower()
        tools = server.get("tools") or []
        for tool in tools:
            flat.append((server_name, (tool.get("tool_name") or "").lower()))
            if len(flat) >= EVAL_DEPTH:
                return flat
    return flat


def find_rank(expected_pair: Tuple[str, str], ranked_pairs: Sequence[Tuple[str, str]]) -> Optional[int]:
    for idx, candidate in enumerate(ranked_pairs, start=1):
        if candidate == expected_pair:
            return idx
    return None


def recall_at_k(predicted: List[Tuple[str, str]], gold: List[Tuple[str, str]], k: int = 3) -> float:
    # How many of the correct answers did I find in my top-k predictions?
    if not gold:
        return 0.0
    
    top_k = set(predicted[:k])
    gold_set = set(gold)
    found = len(top_k & gold_set)
    return found / len(gold_set)


def evaluate_strategy(
    name: str,
    rankings: Iterable[Tuple[EvalRecord, List[Tuple[str, str]]]],
) -> Dict[str, float]:
    total = 0
    hits_top1 = 0
    hits_top3 = 0
    mrr_total = 0.0
    recall3_scores = []

    for record, ranked_pairs in rankings:
        total += 1
        rank = find_rank(record.expected_pair, ranked_pairs)
        if rank is not None:
            if rank == 1:
                hits_top1 += 1
            if rank <= 3:
                hits_top3 += 1
            mrr_total += 1.0 / rank
        
        # Calculate recall@3 for this query
        r3 = recall_at_k(ranked_pairs, [record.expected_pair], k=3)
        recall3_scores.append(r3)

    precision1 = hits_top1 / total if total else 0.0
    precision3 = hits_top3 / total if total else 0.0
    mrr = mrr_total / total if total else 0.0
    mean_recall3 = sum(recall3_scores) / len(recall3_scores) if recall3_scores else 0.0

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Precision@1: {precision1:.3f}")
    print(f"Precision@3: {precision3:.3f}")
    print(f"Recall@3:    {mean_recall3:.3f}")
    print(f"MRR:         {mrr:.3f}")

    return {
        "precision@1": precision1,
        "precision@3": precision3,
        "recall@3": mean_recall3,
        "mrr": mrr,
    }


def run_lexical(records: Sequence[EvalRecord]) -> Iterable[Tuple[EvalRecord, List[Tuple[str, str]]]]:
    catalog = get_catalog()
    for record in records:
        intent = extract_intent(record.user_query, use_llm=False)
        scored = score_entries(
            user_query=record.user_query,
            entries=catalog,
            intent=intent,
            filter_tags=record.filter_tags,
            use_embeddings=False,
            apply_intent=False,
            apply_capability_bonus=False,
        )
        yield record, flatten_scored_entries(scored)


def run_hybrid(records: Sequence[EvalRecord]) -> Iterable[Tuple[EvalRecord, List[Tuple[str, str]]]]:
    catalog = get_catalog()
    use_llm = has_openai_api_key()
    for record in records:
        intent = extract_intent(record.user_query, use_llm=use_llm)
        scored = score_entries(
            user_query=record.user_query,
            entries=catalog,
            intent=intent,
            filter_tags=record.filter_tags,
            use_embeddings=True,
            apply_intent=True,
            apply_capability_bonus=True,
        )
        yield record, flatten_scored_entries(scored)


def run_hybrid_llm(records: Sequence[EvalRecord]) -> Iterable[Tuple[EvalRecord, List[Tuple[str, str]]]]:
    for record in records:
        suggestions = suggest_mcp_servers_impl(
            user_query=record.user_query,
            top_n=TOP_N,
            max_candidates=20,
            filter_tags=record.filter_tags or None,
        )
        yield record, flatten_server_suggestions(suggestions)


def main() -> None:
    records = load_dataset()
    evaluate_strategy("Lexical-only (TF-IDF)", run_lexical(records))
    evaluate_strategy("Hybrid (Lexical + Embedding + Intent)", run_hybrid(records))
    if has_openai_api_key():
        evaluate_strategy("Hybrid + LLM Rerank", run_hybrid_llm(records))
    else:
        print("\nHybrid + LLM Rerank skipped (OPENAI_API_KEY not set).")


if __name__ == "__main__":
    main()

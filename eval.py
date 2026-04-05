"""
eval.py — Evaluation harness for Paper Scout

Measures:
  - Stage 1 filter recall  (did fast filter drop any ground-truth papers?)
  - Score consistency       (variance across two runs of the same query)
  - Score distribution      (is the model discriminating?)
  - Precision@N             (how many ground-truth papers appear in top N?)
  - Re-rank improvement     (does rerank() improve order?)

Usage:
  python eval.py                # run full eval
  python eval.py --collect      # run pipeline, mark good papers, save as ground truth

Ground truth is saved to eval_ground_truth.json and loaded automatically on next run.
You can also hardcode known IDs in EVAL_SET below as a starting point.
"""

import os
import sys
import json
import time
import statistics
from dotenv import load_dotenv
load_dotenv()

GROUND_TRUTH_FILE = os.path.join(os.path.dirname(__file__), "eval_ground_truth.json")

from paper_scout import (
    arxiv_search, llm_summarize,
    compute_score, rerank, recency_boost,
    client, FAST_MODEL, DETAIL_MODEL, QUICK_THRESHOLD, QUICK_FILTER_PROMPT,
)
import json as _json

def quick_filter_debug(papers, query, threshold):
    """quick_filter with visible per-paper scores for eval diagnostics."""
    papers_text = "\n\n".join(
        f"[{i}] Title: {p.title}\nAbstract: {p.abstract[:300]}"
        for i, p in enumerate(papers)
    )
    msg = client.chat.completions.create(
        model=FAST_MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": QUICK_FILTER_PROMPT.format(
            query=query, papers=papers_text
        )}],
        response_format={"type": "json_object"},
    )
    try:
        data = _json.loads(msg.choices[0].message.content)
        results = data.get("results", [])
        score_map = {r["index"]: float(r.get("score", 0)) for r in results}
        print(f"  Stage 1 scores: { {i: f'{s:.1f}' for i, s in sorted(score_map.items())} }")
        keep = {idx for idx, s in score_map.items() if s >= threshold}
        filtered = [p for i, p in enumerate(papers) if i in keep]
        return filtered if filtered else papers
    except (_json.JSONDecodeError, KeyError, TypeError):
        return papers

# ── Eval dataset ──────────────────────────────────────────────────────────────
# Add arXiv IDs you consider ground-truth for each query.
# Format: last segment of the arXiv URL, e.g. "2005.11401" from arxiv.org/abs/2005.11401
# Leave expected_ids as [] to skip precision measurement for that query.

EVAL_SET = [
    {
        "query": "retrieval augmented generation",
        # RAG survey (Gao et al. 2023); GraphRAG (Edge et al. 2024); Self-RAG (Asai et al. 2023)
        "expected_ids": ["2312.10997", "2404.16130", "2310.11511"],
    },
    {
        "query": "diffusion models image generation",
        # DiT (Peebles 2022); SDXL (Podell et al. 2023)
        "expected_ids": ["2212.09748", "2307.01952"],
    },
    {
        "query": "CRAG techniques for RAG",
        # Corrective RAG (Yan et al. 2024); Self-RAG (Asai et al. 2023); RAG survey (Gao et al. 2023)
        "expected_ids": ["2401.15884", "2310.11511", "2312.10997"],
    },
]

TOP_N = 5
W_RELEVANCE, W_NOVELTY, W_CLARITY = 5.0, 3.0, 2.0
CONSISTENCY_RUNS = 2  # number of runs for consistency check
EVAL_FILTER_THRESHOLD = 3.0  # lower than prod (4.0) to avoid over-filtering in eval


# ── Helpers ───────────────────────────────────────────────────────────────────

def strip_version(arxiv_id: str) -> str:
    """Strip version suffix: '2406.00029v1' -> '2406.00029'."""
    return arxiv_id.split("v")[0]


def precision_at_n(results, expected_ids, n):
    if not expected_ids:
        return None
    top_ids = {strip_version(p.arxiv_id) for p in results[:n]}
    expected = {strip_version(e) for e in expected_ids}
    hits = top_ids & expected
    return len(hits) / min(n, len(expected))


def run_pipeline(query):
    """Full pipeline; returns (pre_rerank, post_rerank, filtered, all_papers)."""
    all_papers = arxiv_search(query)
    filtered = quick_filter_debug(all_papers, query, EVAL_FILTER_THRESHOLD)

    scored = []
    for paper in filtered:
        paper = llm_summarize(paper, query)
        paper.score = compute_score(paper, W_RELEVANCE, W_NOVELTY, W_CLARITY)
        scored.append(paper)

    pre_rerank = sorted(scored, key=lambda p: p.score, reverse=True)[:TOP_N * 2]
    post_rerank = rerank(list(pre_rerank), query)
    post_rerank = sorted(post_rerank, key=lambda p: p.score, reverse=True)[:TOP_N]

    return pre_rerank[:TOP_N], post_rerank, filtered, all_papers


def score_variance(runs):
    """Given list of (post_rerank results) per run, compute per-paper score variance."""
    # collect scores by arxiv_id across runs
    scores_by_id = {}
    for result in runs:
        for p in result:
            scores_by_id.setdefault(p.arxiv_id, []).append(p.score)
    variances = [
        statistics.variance(v) for v in scores_by_id.values() if len(v) > 1
    ]
    return statistics.mean(variances) if variances else 0.0


def rerank_improvement(pre, post):
    """Spearman-like: count how many positions improved after rerank."""
    pre_ids = [p.arxiv_id for p in pre]
    post_ids = [p.arxiv_id for p in post]
    improvements = 0
    for pid in post_ids:
        if pid in pre_ids:
            if post_ids.index(pid) < pre_ids.index(pid):
                improvements += 1
    return improvements


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'─'*60}")
    print(f"  Paper Scout Eval")
    print(f"  fast={FAST_MODEL}")
    print(f"  detail={DETAIL_MODEL}")
    print(f"  filter_threshold={QUICK_THRESHOLD}  top_n={TOP_N}")
    print(f"{'─'*60}\n")

    summary = []

    for item in EVAL_SET:
        query = item["query"]
        expected_ids = item["expected_ids"]
        print(f"Query: \"{query}\"")
        print(f"{'·'*50}")

        # ── Stage 1 filter recall ─────────────────────────────
        time.sleep(4)
        all_papers = arxiv_search(query)
        filtered = quick_filter_debug(all_papers, query, EVAL_FILTER_THRESHOLD)
        filter_recall = None
        if expected_ids:
            all_ids = {strip_version(p.arxiv_id) for p in all_papers}
            filtered_ids = {strip_version(p.arxiv_id) for p in filtered}
            expected_clean = {strip_version(e) for e in expected_ids}
            expected_in_corpus = expected_clean & all_ids
            if expected_in_corpus:
                kept = expected_in_corpus & filtered_ids
                filter_recall = len(kept) / len(expected_in_corpus)
                dropped = expected_in_corpus - filtered_ids
                if dropped:
                    print(f"  [!] Filter dropped ground-truth papers: {dropped}")

        print(f"  Stage 1 filter: {len(all_papers)} → {len(filtered)} papers", end="")
        print(f"  (recall={filter_recall:.2f})" if filter_recall is not None else "  (no ground truth)")

        # ── Score distribution ────────────────────────────────
        scored_papers = []
        for paper in filtered:
            paper = llm_summarize(paper, query)
            paper.score = compute_score(paper, W_RELEVANCE, W_NOVELTY, W_CLARITY)
            scored_papers.append(paper)

        scores = [p.score for p in scored_papers]
        print(f"  Score dist: min={min(scores):.1f}  max={max(scores):.1f}  "
              f"mean={statistics.mean(scores):.1f}  stdev={statistics.stdev(scores) if len(scores) > 1 else 0:.1f}")

        # ── Re-rank improvement ───────────────────────────────
        pre_rerank = sorted(scored_papers, key=lambda p: p.score, reverse=True)[:TOP_N * 2]
        post_rerank = rerank(list(pre_rerank), query)
        post_rerank = sorted(post_rerank, key=lambda p: p.score, reverse=True)[:TOP_N]
        improvements = rerank_improvement(pre_rerank[:TOP_N], post_rerank)
        print(f"  Re-rank: {improvements}/{TOP_N} positions improved")

        # ── Consistency (2 runs) ──────────────────────────────
        print(f"  Consistency: running {CONSISTENCY_RUNS} runs...", end="", flush=True)
        run_results = [post_rerank]
        for _ in range(CONSISTENCY_RUNS - 1):
            _, post, _, _ = run_pipeline(query)
            run_results.append(post)
        var = score_variance(run_results)
        print(f" mean score variance = {var:.3f}")

        # ── Precision@N ───────────────────────────────────────
        p_at_n = precision_at_n(post_rerank, expected_ids, TOP_N)
        if p_at_n is not None:
            print(f"  Precision@{TOP_N}: {p_at_n:.2f}")
        else:
            print(f"  Precision@{TOP_N}: n/a (add expected_ids to measure)")

        # ── Top results ───────────────────────────────────────
        print(f"\n  Top {TOP_N} results:")
        for i, p in enumerate(post_rerank, 1):
            marker = " ✓" if strip_version(p.arxiv_id) in {strip_version(e) for e in expected_ids} else ""
            print(f"    {i}. [{p.score:.1f}] {p.title[:60]}{marker}")

        summary.append({
            "query": query,
            "filter_recall": filter_recall,
            "score_stdev": statistics.stdev(scores) if len(scores) > 1 else 0,
            "rerank_improvements": improvements,
            "consistency_variance": var,
            "precision_at_n": p_at_n,
        })
        print()

    # ── Summary table ─────────────────────────────────────────
    print(f"{'─'*60}")
    print(f"  Summary")
    print(f"{'─'*60}")
    print(f"  {'Query':<35} {'P@N':>5} {'StDev':>6} {'Var':>6} {'Rerank':>7}")
    for r in summary:
        p = f"{r['precision_at_n']:.2f}" if r["precision_at_n"] is not None else "  n/a"
        print(f"  {r['query'][:35]:<35} {p:>5} {r['score_stdev']:>6.2f} "
              f"{r['consistency_variance']:>6.3f} {r['rerank_improvements']:>5}/{TOP_N}")
    print(f"{'─'*60}\n")


def load_ground_truth():
    """Load saved ground truth from file, keyed by query."""
    if os.path.exists(GROUND_TRUTH_FILE):
        with open(GROUND_TRUTH_FILE) as f:
            return json.load(f)
    return {}


def save_ground_truth(gt: dict):
    with open(GROUND_TRUTH_FILE, "w") as f:
        json.dump(gt, f, indent=2)
    print(f"\n  Saved to {GROUND_TRUTH_FILE}")


def collect():
    """Interactive mode: run pipeline per query, let user mark good papers, save ground truth."""
    gt = load_ground_truth()

    print(f"\n{'─'*60}")
    print(f"  Paper Scout — Collect Ground Truth")
    print(f"  Results are scored and shown. Mark which ones are relevant.")
    print(f"  Enter numbers separated by spaces (e.g. 1 3 5), or press Enter to skip.")
    print(f"{'─'*60}\n")

    for item in EVAL_SET:
        query = item["query"]
        print(f"Query: \"{query}\"")
        print(f"{'·'*50}")

        time.sleep(4)
        all_papers = arxiv_search(query)
        filtered = quick_filter_debug(all_papers, query, EVAL_FILTER_THRESHOLD)

        scored = []
        for paper in filtered:
            paper = llm_summarize(paper, query)
            paper.score = compute_score(paper, W_RELEVANCE, W_NOVELTY, W_CLARITY)
            scored.append(paper)

        pre_rerank = sorted(scored, key=lambda p: p.score, reverse=True)[:TOP_N * 2]
        post_rerank = rerank(list(pre_rerank), query)
        results = sorted(post_rerank, key=lambda p: p.score, reverse=True)[:TOP_N]

        print(f"\n  Top {TOP_N} results:")
        for i, p in enumerate(results, 1):
            print(f"    {i}. [{p.score:.1f}] {p.title}")
            print(f"         arxiv:{p.arxiv_id}  {p.url}")
            print(f"         {p.why}")

        existing = gt.get(query, [])
        existing_note = f"  (currently saved: {existing})" if existing else ""
        raw = input(f"\n  Mark relevant (1-{TOP_N}){existing_note}: ").strip()

        if raw:
            try:
                indices = [int(x) - 1 for x in raw.split()]
                selected = [results[i].arxiv_id for i in indices if 0 <= i < len(results)]
                merged = list(dict.fromkeys(existing + selected))  # deduplicate, preserve order
                gt[query] = merged
                print(f"  Saved: {merged}")
            except (ValueError, IndexError):
                print("  Invalid input — skipped.")
        else:
            print("  Skipped.")
        print()

    save_ground_truth(gt)


if __name__ == "__main__":
    if "--collect" in sys.argv:
        collect()
    else:
        # Merge saved ground truth into EVAL_SET
        gt = load_ground_truth()
        for item in EVAL_SET:
            saved = gt.get(item["query"], [])
            merged = list(dict.fromkeys(item["expected_ids"] + saved))
            item["expected_ids"] = merged
        main()

"""
paper_scout.py — Research Paper Triage Agent
Search → Filter → Summarize → Rank → Report
"""

import os
import json
from datetime import datetime, timezone
import arxiv
from groq import Groq, BadRequestError
from pydantic import BaseModel
from dataclasses import dataclass, field

# ── Config ────────────────────────────────────────────────────────────────────

MAX_RESULTS    = 20   # papers fetched from arXiv
TOP_N          = 5    # papers in final report
FAST_MODEL     = "llama-3.1-8b-instant"
DETAIL_MODEL   = "moonshotai/kimi-k2-instruct"
QUICK_THRESHOLD = 4.0  # stage 1 filter cutoff

client = Groq(api_key=os.environ["GROQ_API_KEY"])

# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Paper:
    arxiv_id:  str
    title:     str
    authors:   list[str]
    abstract:  str
    url:       str
    published: datetime = field(default_factory=lambda: datetime.min.replace(tzinfo=timezone.utc))
    score:     float = 0.0
    relevance: float = 0.0
    novelty:   float = 0.0
    clarity:   float = 0.0
    summary:   str   = ""
    why:       str   = ""

# ── Tool 1 · arXiv search ─────────────────────────────────────────────────────

def arxiv_search(query: str, max_results: int = MAX_RESULTS) -> list[Paper]:
    """Fetch papers from arXiv matching the query."""
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    papers = []
    for r in arxiv.Client().results(search):
        papers.append(Paper(
            arxiv_id  = r.entry_id.split("/")[-1],
            title     = r.title.strip(),
            authors   = [a.name for a in r.authors[:3]],
            abstract  = r.summary.strip(),
            url       = r.entry_id,
            published = r.published,
        ))
    return papers

# ── Tool 2 · abstract extractor ───────────────────────────────────────────────

def extract_abstract(paper: Paper) -> str:
    """Return the abstract (arXiv API already supplies it)."""
    return paper.abstract   # extend here with PyMuPDF for full-paper extraction

# ── Stage 1 · fast batch filter ───────────────────────────────────────────────

QUICK_FILTER_PROMPT = """\
You are a research assistant. Given a user query and a list of papers (index, title, abstract), \
return a JSON object with a single key "results" containing an array of objects, each with:
- "index": the paper index (integer)
- "score": float 1-10 indicating relevance to the query

User query: {query}

Papers:
{papers}"""

def quick_filter(papers: list[Paper], query: str) -> list[Paper]:
    """Stage 1: one fast batch call to drop obvious irrelevant papers."""
    papers_text = "\n\n".join(
        f"[{i}] Title: {p.title}\nAbstract: {p.abstract[:300]}"
        for i, p in enumerate(papers)
    )
    msg = client.chat.completions.create(
        model=FAST_MODEL,
        max_tokens=512,
        temperature=0,
        messages=[{"role": "user", "content": QUICK_FILTER_PROMPT.format(
            query=query, papers=papers_text
        )}],
        response_format={"type": "json_object"},
    )
    try:
        data = json.loads(msg.choices[0].message.content)
        results = data.get("results", [])
        keep = {r["index"] for r in results if float(r.get("score", 0)) >= QUICK_THRESHOLD}
        filtered = [p for i, p in enumerate(papers) if i in keep]
        return filtered if filtered else papers  # fallback: keep all if filter too aggressive
    except (json.JSONDecodeError, KeyError, TypeError):
        return papers  # fallback: keep all on parse failure

# ── Stage 2 · detailed multi-dim scoring ─────────────────────────────────────

class DetailedScore(BaseModel):
    relevance: float = 0.0
    novelty:   float = 0.0
    clarity:   float = 0.0
    summary:   str   = ""
    why:       str   = ""

DETAIL_PROMPT = """\
You are a research assistant. Given a user query and a paper, respond with a JSON object with exactly these fields:
- "relevance": float 1-10, how closely the paper matches the query topic
- "novelty": float 1-10, how much new insight or method the paper introduces
- "clarity": float 1-10, how well-written and accessible the paper is
- "summary": one sentence summarising the paper
- "why": one sentence explaining why this paper matters for the query

User query: {query}
Title: {title}
Abstract: {abstract}"""

def llm_summarize(paper: Paper, query: str, retries: int = 3) -> Paper:
    """Stage 2: detailed multi-dimensional scoring with structured output."""
    for attempt in range(retries):
        try:
            msg = client.chat.completions.create(
                model=DETAIL_MODEL,
                max_tokens=300,
                temperature=0,
                messages=[{"role": "user", "content": DETAIL_PROMPT.format(
                    query=query, title=paper.title, abstract=paper.abstract
                )}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "DetailedScore", "schema": DetailedScore.model_json_schema()},
                },
            )
            result = DetailedScore.model_validate_json(msg.choices[0].message.content)
            paper.relevance = result.relevance
            paper.novelty   = result.novelty
            paper.clarity   = result.clarity
            paper.summary   = result.summary
            paper.why       = result.why
            return paper
        except BadRequestError as e:
            if "json_validate_failed" not in str(e) or attempt == retries - 1:
                raise
    return paper

# ── Recency boost ─────────────────────────────────────────────────────────────

def recency_boost(paper: Paper) -> float:
    """Return a score boost based on publication age."""
    now = datetime.now(timezone.utc)
    age_days = (now - paper.published).days
    if age_days <= 180:
        return 1.0
    elif age_days <= 365:
        return 0.5
    return 0.0

# ── Weighted score ────────────────────────────────────────────────────────────

def compute_score(paper: Paper, w_relevance: float, w_novelty: float, w_clarity: float) -> float:
    total_weight = w_relevance + w_novelty + w_clarity
    base = (
        w_relevance * paper.relevance +
        w_novelty   * paper.novelty +
        w_clarity   * paper.clarity
    ) / total_weight
    return base + recency_boost(paper)

# ── Relative re-ranking ───────────────────────────────────────────────────────

RERANK_PROMPT = """\
You are a research assistant. Given a user query and a set of candidate papers, \
re-score each paper (1-10) relative to the others — the best paper for this query should \
score highest. Return a JSON object with a single key "results" containing an array of \
objects, each with "index" (integer) and "score" (float).

User query: {query}

Papers:
{papers}"""

def rerank(papers: list[Paper], query: str) -> list[Paper]:
    """Relative re-ranking pass: scores papers against each other."""
    if len(papers) < 3:
        return papers
    papers_text = "\n\n".join(
        f"[{i}] {p.title}\n{p.summary}"
        for i, p in enumerate(papers)
    )
    try:
        msg = client.chat.completions.create(
            model=DETAIL_MODEL,
            max_tokens=256,
            temperature=0,
            messages=[{"role": "user", "content": RERANK_PROMPT.format(
                query=query, papers=papers_text
            )}],
            response_format={"type": "json_object"},
        )
        data = json.loads(msg.choices[0].message.content)
        results = {r["index"]: float(r["score"]) for r in data.get("results", [])}
        for i, paper in enumerate(papers):
            if i in results:
                paper.score = results[i]
    except (json.JSONDecodeError, KeyError, TypeError, BadRequestError):
        pass  # fallback: keep existing scores
    return papers

# ── Agent loop ────────────────────────────────────────────────────────────────

def run(
    query: str,
    top_n: int = TOP_N,
    score_threshold: float = 5.0,
    w_relevance: float = 5.0,
    w_novelty: float = 3.0,
    w_clarity: float = 2.0,
) -> list[Paper]:
    print(f"\n🔍  Searching arXiv for: \"{query}\"")
    papers = arxiv_search(query)
    print(f"    {len(papers)} papers found")

    print(f"\n⚡  Stage 1: fast filter ({FAST_MODEL})...")
    papers = quick_filter(papers, query)
    print(f"    {len(papers)} papers passed filter\n")

    scored = []
    for i, paper in enumerate(papers, 1):
        paper = llm_summarize(paper, query)
        paper.score = compute_score(paper, w_relevance, w_novelty, w_clarity)
        boost = recency_boost(paper)
        status = f"[{i:02d}/{len(papers)}] score={paper.score:.1f} (R:{paper.relevance:.0f} N:{paper.novelty:.0f} C:{paper.clarity:.0f} +{boost:.1f}↑)"
        print(f"    {status}  {paper.title[:50]}…")
        if paper.score >= score_threshold:
            scored.append(paper)

    print(f"\n🔁  Re-ranking top candidates...")
    ranked = sorted(scored, key=lambda p: p.score, reverse=True)[:top_n * 2]
    ranked = rerank(ranked, query)
    ranked = sorted(ranked, key=lambda p: p.score, reverse=True)[:top_n]
    return ranked

# ── Report ────────────────────────────────────────────────────────────────────

def report(papers: list[Paper]) -> None:
    print("\n" + "─" * 70)
    print(f"  📄  Top {len(papers)} papers")
    print("─" * 70)
    for rank, p in enumerate(papers, 1):
        print(f"\n  #{rank}  [{p.score:.1f}/10]  {p.title}")
        print(f"       {', '.join(p.authors)}  ({p.published.strftime('%Y-%m')})")
        print(f"       R:{p.relevance:.1f}  N:{p.novelty:.1f}  C:{p.clarity:.1f}")
        print(f"       {p.summary}")
        print(f"       ↳ {p.why}")
        print(f"       {p.url}")
    print("\n" + "─" * 70)

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    query  = " ".join(sys.argv[1:]) or "CRAG techniques for RAG"
    papers = run(query)
    report(papers)

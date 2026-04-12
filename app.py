"""
app.py — Gradio UI for Paper Scout
"""

import os
from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from paper_scout import arxiv_search, llm_summarize, compute_score, rerank, TOP_N

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill="#f7f3ed",
    body_text_color="#1c1917",
    block_background_fill="#ffffff",
    block_border_color="#d6cfc5",
    block_label_text_color="#57534e",
    input_background_fill="#ffffff",
    input_border_color="#d6cfc5",
    input_placeholder_color="#a8a29e",
    button_primary_background_fill="#1c1917",
    button_primary_background_fill_hover="#3d3733",
    button_primary_text_color="#f7f3ed",
    button_secondary_background_fill="#f7f3ed",
    button_secondary_background_fill_hover="#ede8e0",
    button_secondary_border_color="#d6cfc5",
    button_secondary_text_color="#1c1917",
    slider_color="#b45309",
    border_color_primary="#d6cfc5",
    shadow_drop="0 1px 3px rgba(28,25,23,0.08)",
)

css = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,700&display=swap');

body, gradio-app, .gradio-container, footer { background: #f7f3ed !important; }

/* let the layout grow to fit all report cards */
.gradio-container { overflow: visible !important; }
gradio-app { overflow: visible !important; }

.gradio-container { max-width: 1600px !important; padding: 0 24px !important; }

/* tighten search row vertical alignment */
#status-box textarea { min-height: 0 !important; }

/* blocks */
.block, .gr-block, .gr-form, .gr-box,
div[data-testid="block"], .svelte-1ipelgc,
.wrap, .panel { background: #ffffff !important; border-color: #d6cfc5 !important; }

/* inputs */
input, textarea, .gr-input, [data-testid="textbox"] input,
[data-testid="textbox"] textarea {
    background: #ffffff !important;
    border-color: #d6cfc5 !important;
    color: #1c1917 !important;
}
input::placeholder, textarea::placeholder { color: #a8a29e !important; }

/* labels */
label, .block label span, span.svelte-1b6s6s, .label-wrap span {
    color: #57534e !important;
    font-size: 12px !important;
    font-weight: 500 !important;
}

/* primary button */
button.primary, .gr-button-primary, button[variant="primary"] {
    background: #1c1917 !important;
    color: #f7f3ed !important;
    border: none !important;
    border-radius: 3px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 14px !important;
    width: 100% !important;
}
button.primary:hover { background: #3d3733 !important; }

/* accordion */
.accordion, details { background: #ffffff !important; border-color: #d6cfc5 !important; }

/* status / output */
.output-class, [data-testid="textbox"] { background: #ffffff !important; }

/* form containers inside textboxes */
.form { background: #ffffff !important; border-color: #d6cfc5 !important; }

/* status box — borderless read-only */
#status-box { border: none !important; background: transparent !important; box-shadow: none !important; padding: 4px 0 !important; }
#status-box .form { background: transparent !important; border: none !important; box-shadow: none !important; }
#status-box textarea {
    background: transparent !important;
    border: none !important;
    color: #a8a29e !important;
    font-size: 12px !important;
    font-family: 'JetBrains Mono', monospace !important;
    resize: none !important;
    padding: 0 !important;
}
#status-box label span { color: #c4bdb5 !important; font-size: 11px !important; }

/* results report container — no Gradio chrome */
#results-report { border: none !important; background: transparent !important; box-shadow: none !important; padding: 0 !important; }
#results-report .form { background: transparent !important; border: none !important; box-shadow: none !important; }

/* hide Gradio footer */
footer { display: none !important; }

/* markdown prose */
.gradio-container .prose p,
.gradio-container .prose span,
.gradio-container .md p,
.gradio-container .md span {
    color: #57534e !important;
    font-size: 13px !important;
}

#paper-scout-title {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-size: 42px !important;
    font-weight: 900 !important;
    letter-spacing: -1px !important;
    line-height: 1 !important;
    color: #1c1917 !important;
    margin-bottom: 4px !important;
}
#paper-scout-title em { color: #b45309; font-style: italic; }
#paper-scout-sub {
    color: #57534e !important;
    font-size: 13px !important;
    margin-bottom: 12px !important;
    line-height: 1.5 !important;
}

/* tighten Gradio's default top padding */
.gradio-container > .main > .wrap > .gap { gap: 8px !important; }
.gradio-container { padding-top: 24px !important; }
"""

def _paper_card(rank: int, p) -> str:
    authors_str = ", ".join(p.authors)
    date_str = p.published.strftime("%Y-%m")
    score_pct = int(p.score / 10 * 100)
    return f"""
<div style="
    border: 1px solid #d6cfc5;
    border-radius: 6px;
    padding: 22px 24px;
    margin-bottom: 16px;
    background: #ffffff;
    box-shadow: 0 1px 3px rgba(28,25,23,0.06);
">
    <!-- rank + score header -->
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
        <span style="
            background: #1c1917;
            color: #f7f3ed;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            font-weight: 700;
            padding: 3px 9px;
            border-radius: 3px;
            letter-spacing: 0.04em;
        ">#{rank}</span>
        <span style="
            color: #b45309;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.02em;
        ">{p.score:.1f} / 10</span>
        <!-- score bar -->
        <div style="flex:1; height:4px; background:#ede8e0; border-radius:2px; max-width:120px;">
            <div style="width:{score_pct}%; height:100%; background:#b45309; border-radius:2px;"></div>
        </div>
        <span style="
            color: #a8a29e;
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            margin-left:auto;
        ">R:{p.relevance:.1f} · N:{p.novelty:.1f} · C:{p.clarity:.1f}</span>
    </div>

    <!-- title -->
    <div style="
        font-family: 'Inter', sans-serif;
        font-size: 16px;
        font-weight: 700;
        color: #1c1917;
        line-height: 1.35;
        margin-bottom: 6px;
    ">{p.title}</div>

    <!-- meta -->
    <div style="
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: #a8a29e;
        margin-bottom: 16px;
    ">{authors_str} &nbsp;·&nbsp; {date_str} &nbsp;·&nbsp;
        <a href="{p.url}" target="_blank" style="color:#b45309; text-decoration:none;">{p.url}</a>
    </div>

    <!-- summary -->
    <div style="margin-bottom:14px;">
        <div style="
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #a8a29e;
            margin-bottom: 5px;
            font-family: 'Inter', sans-serif;
        ">Summary</div>
        <div style="
            font-size: 14px;
            color: #57534e;
            line-height: 1.65;
            font-family: 'Inter', sans-serif;
        ">{p.summary}</div>
    </div>

    <!-- why it matters -->
    <div style="
        background: #fef3c7;
        border-left: 3px solid #b45309;
        border-radius: 0 4px 4px 0;
        padding: 10px 14px;
    ">
        <div style="
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #b45309;
            margin-bottom: 5px;
            font-family: 'Inter', sans-serif;
        ">Why it matters</div>
        <div style="
            font-size: 14px;
            color: #1c1917;
            line-height: 1.65;
            font-family: 'Inter', sans-serif;
        ">{p.why}</div>
    </div>
</div>"""


def _report_html(query: str, papers: list) -> str:
    header = f"""
<div style="
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    color: #a8a29e;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid #d6cfc5;
">Top {len(papers)} papers &nbsp;·&nbsp; "{query}"</div>"""
    cards = "".join(_paper_card(rank, p) for rank, p in enumerate(papers, 1))
    return header + cards


def search_papers(
    query: str,
    top_n: int,
    score_threshold: float,
    w_relevance: float,
    w_novelty: float,
    w_clarity: float,
):
    if not query.strip():
        yield "Please enter a search query.", ""
        return

    yield "Searching arXiv...", ""

    papers = arxiv_search(query)
    total = len(papers)

    yield f"Fast-filtering {total} papers...", ""

    from paper_scout import quick_filter
    papers = quick_filter(papers, query)

    yield f"{len(papers)} papers passed filter. Scoring...", ""

    scored = []
    for i, paper in enumerate(papers, 1):
        paper = llm_summarize(paper, query)
        paper.score = compute_score(paper, w_relevance, w_novelty, w_clarity)
        status = f"Scoring [{i}/{len(papers)}]  (score={paper.score:.1f}) {paper.title[:55]}"
        yield status, ""
        if paper.score >= score_threshold:
            scored.append(paper)

    if not scored:
        yield f"No papers found with score >= {score_threshold}.", ""
        return

    yield "Re-ranking candidates...", ""
    candidates = sorted(scored, key=lambda p: p.score, reverse=True)[:int(top_n) * 2]
    candidates = rerank(candidates, query)
    ranked = sorted(candidates, key=lambda p: p.score, reverse=True)[:int(top_n)]

    yield f"Done — showing top {len(ranked)} papers.", _report_html(query, ranked)


with gr.Blocks(title="Paper Scout") as demo:
    gr.HTML('<div id="paper-scout-title">Paper<em>Scout.</em></div>')
    gr.HTML('<div id="paper-scout-sub">Search arXiv, score every result with an LLM, and surface the papers that actually matter to you.</div>')

    with gr.Row(equal_height=True):
        query_box = gr.Textbox(
            label="Research query",
            placeholder="e.g. CRAG techniques for RAG",
            lines=1,
            scale=4,
        )
        with gr.Column(scale=1, min_width=160):
            search_btn = gr.Button("Search", variant="primary", size="lg")
            status_box = gr.Textbox(label="", show_label=False, interactive=False, elem_id="status-box", lines=2)

    with gr.Row(equal_height=True):
        top_n_slider = gr.Slider(1, 20, value=TOP_N, step=1, label="Top N results")
        threshold_slider = gr.Slider(1.0, 10.0, value=5.0, step=0.5, label="Min score threshold")
        with gr.Accordion("Score weights", open=False):
            with gr.Row():
                w_relevance = gr.Slider(1, 10, value=5, step=1, label="Relevance weight")
                w_novelty   = gr.Slider(1, 10, value=3, step=1, label="Novelty weight")
                w_clarity   = gr.Slider(1, 10, value=2, step=1, label="Clarity weight")

    results_report = gr.HTML(elem_id="results-report")

    inputs = [query_box, top_n_slider, threshold_slider, w_relevance, w_novelty, w_clarity]

    search_btn.click(fn=search_papers, inputs=inputs, outputs=[status_box, results_report])
    query_box.submit(fn=search_papers, inputs=inputs, outputs=[status_box, results_report])

if __name__ == "__main__":
    demo.launch(theme=theme, css=css)

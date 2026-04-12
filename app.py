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

/* prevent viewport-fill stretching the layout */
.gradio-container > .main > .wrap { flex-grow: 0 !important; min-height: 0 !important; }

/* virtual table */
.virtual-table-viewport, .table-wrap, .table-container { background: #ffffff !important; }
.virtual-row, .row-odd { background: #ffffff !important; color: #1c1917 !important; }
.virtual-row:hover, .row-odd:hover { background: #fef3c7 !important; }
button.svelte-8prmba, .disable_click {
    background: transparent !important;
    color: #1c1917 !important;
    border-color: #ede8e0 !important;
}

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

/* table */
table, .gr-dataframe { background: #ffffff !important; border-color: #d6cfc5 !important; }
th {
    background: #f7f3ed !important;
    color: #1c1917 !important;
    border-color: #d6cfc5 !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}
td { border-color: #ede8e0 !important; color: #1c1917 !important; }

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

/* cell menu buttons in table */
.cell-menu-button { background: #f7f3ed !important; color: #57534e !important; border-color: #d6cfc5 !important; }

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

def search_papers(
    query: str,
    top_n: int,
    score_threshold: float,
    w_relevance: float,
    w_novelty: float,
    w_clarity: float,
):
    if not query.strip():
        yield "Please enter a search query.", []
        return

    yield "Searching arXiv...", []

    papers = arxiv_search(query)
    total = len(papers)

    yield f"Fast-filtering {total} papers...", []

    from paper_scout import quick_filter
    papers = quick_filter(papers, query)

    yield f"{len(papers)} papers passed filter. Scoring...", []

    scored = []
    for i, paper in enumerate(papers, 1):
        paper = llm_summarize(paper, query)
        paper.score = compute_score(paper, w_relevance, w_novelty, w_clarity)
        status = f"Scoring [{i}/{len(papers)}]  (score={paper.score:.1f}) {paper.title[:55]}"
        yield status, []
        if paper.score >= score_threshold:
            scored.append(paper)

    if not scored:
        yield f"No papers found with score >= {score_threshold}.", []
        return

    yield "Re-ranking candidates...", []
    candidates = sorted(scored, key=lambda p: p.score, reverse=True)[:int(top_n) * 2]
    candidates = rerank(candidates, query)
    ranked = sorted(candidates, key=lambda p: p.score, reverse=True)[:int(top_n)]

    rows = [
        [
            f"#{rank}",
            f"{p.score:.1f}",
            p.title,
            ", ".join(p.authors),
            p.published.strftime("%Y-%m"),
            p.summary,
            p.why,
            f"[{p.url}]({p.url})",
        ]
        for rank, p in enumerate(ranked, 1)
    ]
    yield f"Done — showing top {len(ranked)} papers.", rows


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

    with gr.Row():
        top_n_slider = gr.Slider(1, 20, value=TOP_N, step=1, label="Top N results")
        threshold_slider = gr.Slider(1.0, 10.0, value=5.0, step=0.5, label="Min score threshold")
        with gr.Accordion("Score weights", open=False):
            with gr.Row():
                w_relevance = gr.Slider(1, 10, value=5, step=1, label="Relevance weight")
                w_novelty   = gr.Slider(1, 10, value=3, step=1, label="Novelty weight")
                w_clarity   = gr.Slider(1, 10, value=2, step=1, label="Clarity weight")

    results_table = gr.Dataframe(
        headers=["Rank", "Score", "Title", "Authors", "Published", "Summary", "Why it matters", "Link"],
        datatype=["str", "str", "str", "str", "str", "str", "str", "markdown"],
        wrap=True,
        interactive=False,
        max_height=700,
    )

    inputs = [query_box, top_n_slider, threshold_slider, w_relevance, w_novelty, w_clarity]

    search_btn.click(fn=search_papers, inputs=inputs, outputs=[status_box, results_table])
    query_box.submit(fn=search_papers, inputs=inputs, outputs=[status_box, results_table])

if __name__ == "__main__":
    demo.launch(theme=theme, css=css)

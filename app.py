"""
RAG Testing Dashboard — Full UI
=================================
Screens:
  1 — Setup  (upload, GT JSON, judge selection, model selection, question)
  2 — Single-model Generate result
  3 — Single-model Compare vs Gemini result
  4 — Multi-model Comparison table  ← NEW: visual charts inside expandable sections

NEW ADDITIONS (non-breaking):
  • Score Interpretation Labels   — every numeric score gets a human-readable verdict
  • Tooltips on Every Metric      — ℹ️ hover text explaining each metric in plain English
  • Answer Diff Viewer            — word-level coloured diff on Screen 3
  • Confidence Indicator          — retrieval quality banner on all result screens
  • Document Preview Panel        — file list with type/size shown after upload
  • Export Results Button         — download full results as formatted Excel (.xlsx)
  • Judge Model Selector          — choose GPT-4o-mini or any Ollama model as judge;
                                    selected judge is removed from generation lists
  • [SCREEN 4 ONLY] Visual Comparison Charts — interactive Plotly charts inside
                                    expandable sections with descriptive tooltips
"""

import streamlit as st
import pandas as pd
import zipfile, uuid, json, time, difflib
from datetime import datetime
from pathlib import Path

from Gemini_model import (
    run_rag_evaluation, run_multi_model_comparison,
    get_available_ollama_models, find_ground_truth,
    create_temp_collection, index_uploaded_files, delete_temp_collection,
    RAGSystem, generate_with_ollama, evaluate_with_gpt4o_mini,
    calculate_fast_metrics, calculate_semantic_similarity,
    get_document_preview, export_to_excel,
    QDRANT_URL, TOP_K, JUDGE_GPT,
)

# ==================================================
# PAGE CONFIG & CSS
# ==================================================
st.set_page_config(page_title="RAG Testing Dashboard", layout="wide")
UPLOAD_BASE = Path("uploads")
UPLOAD_BASE.mkdir(exist_ok=True)

st.markdown("""
<style>
.main-title{font-size:32px;font-weight:700;text-align:center;margin-bottom:25px}
.answer-card{background:linear-gradient(135deg,#eef2ff,#f8fafc);padding:22px;border-radius:18px;
  box-shadow:0 6px 25px rgba(0,0,0,.12);margin-bottom:20px;min-height:120px}
.metric-section{background:#f0f4ff;padding:14px 18px;border-radius:12px;margin-bottom:14px;
  border-left:4px solid #4f6ef7}
.metric-section h5{margin:0 0 8px 0;color:#374151;font-size:14px;font-weight:600;
  text-transform:uppercase;letter-spacing:.05em}
.gt-card{background:linear-gradient(135deg,#f0fdf4,#dcfce7);padding:18px;border-radius:14px;
  border-left:4px solid #22c55e;margin-bottom:18px}
.winner-card{background:linear-gradient(135deg,#fef9c3,#fefce8);padding:14px 18px;border-radius:14px;
  border-left:4px solid #eab308;margin-bottom:18px}
.confidence-high{background:#f0fdf4;border-left:4px solid #22c55e;padding:10px 16px;border-radius:8px;margin-bottom:12px}
.confidence-med{background:#fffbeb;border-left:4px solid #f59e0b;padding:10px 16px;border-radius:8px;margin-bottom:12px}
.confidence-low{background:#fef2f2;border-left:4px solid #ef4444;padding:10px 16px;border-radius:8px;margin-bottom:12px}
.diff-box{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:16px;
  font-size:13px;line-height:1.9;overflow-y:auto;max-height:320px}
.doc-preview-row{display:flex;gap:10px;align-items:center;padding:6px 0;border-bottom:1px solid #f1f5f9}
.score-label{font-size:11px;font-weight:700;padding:2px 7px;border-radius:99px;margin-left:4px}
.judge-badge{background:#e0e7ff;color:#3730a3;padding:3px 10px;border-radius:99px;
  font-size:12px;font-weight:700;display:inline-block}
.chart-tooltip-box{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
  padding:14px 18px;margin-bottom:16px;font-size:13px;line-height:1.7;color:#374151}
.chart-tooltip-box b{color:#1e40af}
</style>
""", unsafe_allow_html=True)


# ==================================================
# SCORE INTERPRETATION
# ==================================================

def score_label(value: float, max_val: float = 10.0) -> tuple:
    v = value * (10.0 / max_val) if max_val != 10.0 else value
    if v >= 8.0:   return "Excellent",  "#bbf7d0", "#14532d"
    if v >= 6.5:   return "Good",       "#d1fae5", "#065f46"
    if v >= 5.0:   return "Acceptable", "#fef9c3", "#713f12"
    if v >= 3.0:   return "Poor",       "#fed7aa", "#7c2d12"
    return           "Very Poor",       "#fecaca", "#7f1d1d"


def score_label_normalized(value: float) -> tuple:
    return score_label(value * 10.0)


def fmt_score(value, is_normalized: bool = False) -> str:
    if isinstance(value, float):
        return f"{value:.4f}" if is_normalized else f"{value:.2f}"
    return str(value)


# ==================================================
# METRIC TOOLTIPS
# ==================================================

METRIC_HELP = {
    "Faithfulness":
        "Are all claims in the answer actually supported by the retrieved document chunks? "
        "High score = nothing was made up. Score: 0 (fabricated) → 10 (fully grounded).",
    "Completeness":
        "Does the answer address ALL parts of the question, or does it miss key aspects? "
        "Score: 0 (barely touched the question) → 10 (fully addressed every part).",
    "Correctness":
        "How accurate is the answer compared to the ground truth (or context when no GT)? "
        "Score: 0 (wrong) → 10 (perfectly correct).",
    "Overall Score":
        "Weighted combination: Faithfulness×35% + Completeness×35% + Correctness×30%. "
        "Best single number to summarise answer quality.",
    "Sem.Sim vs Gemini":
        "Cosine similarity (0–1) between this model's answer and Gemini's answer in embedding space. "
        "Always shown for all models. Above 0.85 = very similar meaning to Gemini; "
        "below 0.5 = substantially different answer. "
        "Gemini's own column is 1.0 (comparing with itself).",
    "Sem.Sim vs GT":
        "Cosine similarity (0–1) between this model's answer and the Ground Truth answer "
        "in embedding space. Only shown when a GT JSON is uploaded and matched. "
        "Above 0.85 = answer is very close to the reference; below 0.5 = divergent from GT.",
    "Semantic Sim.":
        "Cosine similarity between the model answer and the reference in "
        "high-dimensional embedding space. Range 0–1. Above 0.85 means very similar meaning.",
    "Semantic Similarity":
        "Cosine similarity between the model answer and the reference in embedding space. "
        "Range 0–1. Above 0.85 = very similar meaning; below 0.5 = divergent.",
    "BERTScore F1":
        "Semantic overlap between the answer and reference using a neural language model. "
        "Range ~0–1 (can be negative with baseline rescaling). Higher = more similar meaning.",
    "BERTScore Prec.":
        "Fraction of the generated answer's content that appears (semantically) in the reference. "
        "High precision = answer doesn't say things the reference doesn't say.",
    "BERTScore Recall":
        "Fraction of the reference's content that is covered by the generated answer. "
        "High recall = answer doesn't miss important parts of the reference.",
    "ROUGE-L F1":
        "Longest Common Subsequence overlap between answer and ground truth. "
        "Captures word order similarity. Range 0–1. Only meaningful with real ground truth.",
    "ROUGE-1 F1":
        "Overlap of individual words (unigrams) between answer and ground truth. "
        "Range 0–1. Only meaningful with real ground truth.",
    "BLEU":
        "Bilingual Evaluation Understudy — measures n-gram overlap with ground truth. "
        "Originally designed for machine translation. Range 0–1. Only with real GT.",
    "METEOR":
        "Like BLEU but also rewards synonyms and stemming. "
        "Range 0–1. Generally more correlated with human judgements than BLEU.",
    "Chunks Retrieved":
        "How many document chunks were retrieved from the vector store to answer this question. "
        "Configured maximum is " + str(TOP_K) + ".",
    "Unique Files":
        "How many different source files contributed to the retrieved chunks. "
        "Low number may mean the answer relies heavily on one source.",
    "Avg Chunk Score":
        "Average cosine similarity score of the retrieved chunks to the query. "
        "Higher = more relevant retrieval. Below 0.4 may indicate weak document match.",
    "Top File":
        "The source file that contributed the highest-scoring chunk to the retrieval.",
    "Latency (s)":
        "End-to-end time from sending the prompt to receiving the full generated answer. "
        "Includes network and model inference time.",
    "Model Latency":
        "Time taken by the selected Ollama model to generate its answer.",
    "Gemini Latency":
        "Time taken by the Gemini endpoint to generate its answer.",
}


# ==================================================
# CONFIDENCE INDICATOR
# ==================================================

def render_confidence_indicator(avg_chunk_score: float, chunks_retrieved: int):
    if chunks_retrieved == 0:
        st.markdown(
            "<div class='confidence-low'>🔴 <b>No chunks retrieved</b> — "
            "the document may not contain relevant content for this question.</div>",
            unsafe_allow_html=True)
        return
    if chunks_retrieved < 3:
        st.markdown(
            f"<div class='confidence-med'>🟡 <b>Very few chunks retrieved ({chunks_retrieved})</b> — "
            "answer coverage may be limited. Consider rephrasing the question.</div>",
            unsafe_allow_html=True)
    elif avg_chunk_score >= 0.60:
        st.markdown(
            f"<div class='confidence-high'>🟢 <b>High retrieval confidence</b> — "
            f"avg chunk relevance score {avg_chunk_score:.3f}. Answer is well-grounded.</div>",
            unsafe_allow_html=True)
    elif avg_chunk_score >= 0.40:
        st.markdown(
            f"<div class='confidence-med'>🟡 <b>Medium retrieval confidence</b> — "
            f"avg chunk relevance score {avg_chunk_score:.3f}. "
            "Some chunks may be loosely related.</div>",
            unsafe_allow_html=True)
    else:
        st.markdown(
            f"<div class='confidence-low'>🔴 <b>Low retrieval confidence</b> — "
            f"avg chunk relevance score {avg_chunk_score:.3f}. "
            "The document may not contain a strong answer to this question. "
            "Treat results with caution.</div>",
            unsafe_allow_html=True)


# ==================================================
# ANSWER DIFF VIEWER
# ==================================================

def build_word_diff_html(text_a: str, text_b: str) -> tuple:
    words_a = text_a.split()
    words_b = text_b.split()
    matcher = difflib.SequenceMatcher(None, words_a, words_b, autojunk=False)

    parts_a, parts_b = [], []
    STYLES = {
        "equal":    "background:#dcfce7;border-radius:3px;padding:1px 3px",
        "replace_a":"background:#fef08a;border-radius:3px;padding:1px 3px",
        "replace_b":"background:#fef08a;border-radius:3px;padding:1px 3px",
        "delete":   "background:#fecaca;border-radius:3px;padding:1px 3px",
        "insert":   "background:#bfdbfe;border-radius:3px;padding:1px 3px",
    }

    def _span(word, style_key):
        s = STYLES[style_key]
        w = word.replace("<","&lt;").replace(">","&gt;")
        return f'<span style="{s}">{w}</span>'

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        chunk_a = words_a[i1:i2]
        chunk_b = words_b[j1:j2]
        if tag == "equal":
            parts_a.extend(_span(w, "equal") for w in chunk_a)
            parts_b.extend(_span(w, "equal") for w in chunk_b)
        elif tag == "replace":
            parts_a.extend(_span(w, "replace_a") for w in chunk_a)
            parts_b.extend(_span(w, "replace_b") for w in chunk_b)
        elif tag == "delete":
            parts_a.extend(_span(w, "delete") for w in chunk_a)
        elif tag == "insert":
            parts_b.extend(_span(w, "insert") for w in chunk_b)

    return " ".join(parts_a), " ".join(parts_b)


def render_answer_diff(text_a: str, text_b: str, label_a: str, label_b: str,
                       similarity: float = None):
    """
    Render a word-level diff between two answers inside an expander.
    If `similarity` is provided (0–1), a semantic similarity badge is shown at the top.
    """
    # Build expander title — include similarity summary so user sees it even collapsed
    sim_title = f" | 🧠 Similarity: {similarity:.4f}" if similarity is not None else ""
    with st.expander(
        f"🔍 Answer Diff — {label_a} vs {label_b}{sim_title}",
        expanded=False,
    ):
        # ── Semantic similarity badge ────────────────────────────────────────
        if similarity is not None:
            lbl, bg, fg = score_label_normalized(float(similarity))
            st.markdown(
                f"<div style='display:inline-flex;align-items:center;gap:10px;"
                f"background:{bg};color:{fg};padding:9px 16px;border-radius:10px;"
                f"margin-bottom:14px;font-size:14px;font-weight:600;width:100%'>"
                f"🧠 Semantic Similarity: <strong>{similarity:.4f}</strong>"
                f"<span style='background:rgba(0,0,0,0.08);border-radius:99px;"
                f"padding:2px 10px;font-size:12px'>{lbl}</span>"
                f"<span style='font-weight:400;font-size:12px;margin-left:4px;opacity:0.75'>"
                f"(cosine similarity in embedding space — 0 = unrelated, 1 = identical meaning)</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        # ── Word-level diff legend ───────────────────────────────────────────
        st.markdown(
            "<div style='display:flex;gap:10px;flex-wrap:wrap;margin-bottom:8px;font-size:13px'>"
            "<span style='background:#dcfce7;border-radius:4px;padding:2px 8px'>🟢 Shared words</span>"
            "<span style='background:#fef08a;border-radius:4px;padding:2px 8px'>🟡 Different words (replaced)</span>"
            "<span style='background:#fecaca;border-radius:4px;padding:2px 8px'>🔴 Only in first answer</span>"
            "<span style='background:#bfdbfe;border-radius:4px;padding:2px 8px'>🔵 Only in second answer</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        html_a, html_b = build_word_diff_html(text_a, text_b)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**{label_a}**")
            st.markdown(f"<div class='diff-box'>{html_a}</div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"**{label_b}**")
            st.markdown(f"<div class='diff-box'>{html_b}</div>", unsafe_allow_html=True)


# ==================================================
# DOCUMENT PREVIEW PANEL
# ==================================================

EXT_ICONS = {
    "PDF":"📄","DOCX":"📝","DOC":"📝","TXT":"📃","MD":"📃",
    "CSV":"📊","XLSX":"📊","XLS":"📊","JSON":"🔧","HTML":"🌐",
    "HTM":"🌐","PPTX":"📋","PPT":"📋","XML":"🔧","YAML":"🔧","YML":"🔧",
}

def render_document_preview(run_path: Path):
    files = get_document_preview(run_path)
    if not files:
        st.info("No readable files found in the uploaded content.")
        return

    st.markdown(f"**{len(files)} indexable file{'s' if len(files)!=1 else ''}:**")
    cols = st.columns([3,1,1])
    cols[0].markdown("**File**"); cols[1].markdown("**Type**"); cols[2].markdown("**Size**")

    for f in files:
        icon = EXT_ICONS.get(f["Type"], "📁")
        c1, c2, c3 = st.columns([3,1,1])
        c1.markdown(f"{icon} `{f['File']}`")
        c2.markdown(f"`{f['Type']}`")
        c3.markdown(f"`{f['Size']}`")


# ==================================================
# EXPORT BUTTON
# ==================================================

def render_export_button(
    question:      str,
    model_answer:  str  = "",
    gemini_answer: str  = "",
    metrics:       dict = None,
    gt_entry:      dict = None,
    comparison:    dict = None,
    judge_model:   str  = JUDGE_GPT,
    key_suffix:    str  = "",
):
    try:
        xls_bytes = export_to_excel(
            question=question, model_answer=model_answer, gemini_answer=gemini_answer,
            metrics=metrics, gt_entry=gt_entry, comparison=comparison, judge_model=judge_model,
        )
        st.download_button(
            label="⬇️ Export Results to Excel",
            data=xls_bytes,
            file_name=f"rag_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"export_btn_{key_suffix}",
            use_container_width=True,
        )
    except Exception as e:
        st.warning(f"Export failed: {e}")


# ==================================================
# HELPERS
# ==================================================

def answer_card(title, content):
    safe = content.replace("<","&lt;").replace(">","&gt;").replace("\n","<br>")
    st.markdown(f"<div class='answer-card'><h4>{title}</h4>"
                f"<div style='font-size:14px;line-height:1.6'>{safe}</div></div>",
                unsafe_allow_html=True)


def save_and_extract(files):
    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:6]
    run_path = UPLOAD_BASE / f"run_{run_id}"
    run_path.mkdir(parents=True, exist_ok=True)
    for f in files:
        sp = run_path / f.name
        with open(sp,"wb") as out: out.write(f.getbuffer())
        if f.name.lower().endswith(".zip"):
            with zipfile.ZipFile(sp,"r") as zf: zf.extractall(run_path)
    return run_path


def gt_badge(gt_entry):
    if gt_entry:
        return ("<span style='background:#dcfce7;color:#15803d;padding:3px 10px;"
                "border-radius:99px;font-size:12px;font-weight:700'>✅ GT Found</span>")
    return ("<span style='background:#fee2e2;color:#b91c1c;padding:3px 10px;"
            "border-radius:99px;font-size:12px;font-weight:700'>❌ No GT Match</span>")


def render_gt_box(gt_entry):
    if not gt_entry: return
    st.markdown("<div class='gt-card'>", unsafe_allow_html=True)
    st.markdown(
        f"**📋 Ground Truth Answer** &nbsp;"
        f"<span style='color:#6b7280;font-size:13px'>"
        f"{gt_entry.get('source_document','')} · {gt_entry.get('id','')} · "
        f"{gt_entry.get('difficulty','')} · {gt_entry.get('question_type','')}"
        f"</span>", unsafe_allow_html=True)
    st.markdown(gt_entry.get("answer",""))
    evid = gt_entry.get("evidence_snippets",[])
    if evid:
        with st.expander("🔍 Evidence Snippets"):
            for i,s in enumerate(evid,1):
                st.markdown(f"**{i}.** {s}")
    st.markdown("</div>", unsafe_allow_html=True)


# ==================================================
# METRIC RENDERING WITH LABELS + TOOLTIPS
# ==================================================

def _metric_with_label(col, display_name: str, raw_value, is_normalized: bool = False,
                        help_key: str = None):
    help_text = METRIC_HELP.get(help_key or display_name, "")
    if isinstance(raw_value, float):
        disp = f"{raw_value:.4f}" if is_normalized else f"{raw_value:.2f}"
    else:
        disp = str(raw_value)

    col.metric(display_name, disp, help=help_text if help_text else None)

    if isinstance(raw_value, (int, float)):
        if is_normalized:
            label, bg, fg = score_label_normalized(float(raw_value))
        else:
            label, bg, fg = score_label(float(raw_value), max_val=10.0)
        col.markdown(
            f"<span class='score-label' style='background:{bg};color:{fg}'>{label}</span>",
            unsafe_allow_html=True)


def render_single_metrics(metrics: dict, mode: str, has_gt: bool, judge_model: str = JUDGE_GPT):
    st.subheader("📊 Evaluation Metrics")
    ref_label  = metrics.get("reference_label", "Reference")
    avg_score  = metrics.get("avg_chunk_score", 0.0)
    n_chunks   = metrics.get("chunks_retrieved", 0)

    render_confidence_indicator(avg_score, n_chunks)

    st.markdown(
        f"<div style='margin-bottom:10px'>"
        f"<span class='judge-badge'>⚖️ Judge: {judge_model}</span> &nbsp;"
        f"<span style='color:#6b7280;font-size:13px'>Reference: {ref_label}</span>"
        f"</div>",
        unsafe_allow_html=True)

    st.markdown("<div class='metric-section'><h5>⚖️ LLM Judge Scores (0–10)</h5></div>",
                unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    _metric_with_label(c1, "Faithfulness",  metrics.get("faithfulness",0),  help_key="Faithfulness")
    _metric_with_label(c2, "Completeness",  metrics.get("completeness",0),  help_key="Completeness")
    _metric_with_label(c3, "Correctness",   metrics.get("correctness",0),   help_key="Correctness")
    _metric_with_label(c4, "Overall Score", metrics.get("overall_score",0), help_key="Overall Score")
    if metrics.get("judge_explanation"):
        st.caption(f"💬 {metrics['judge_explanation']}")

    if mode == "compare" or has_gt:
        lbl = f"🧠 Semantic Quality vs {ref_label}"
        st.markdown(f"<div class='metric-section'><h5>{lbl}</h5></div>", unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        _metric_with_label(c1, "Semantic Sim.", metrics.get("semantic_similarity",0),
                           is_normalized=True, help_key="Semantic Similarity")
        _metric_with_label(c2, "BERTScore F1", metrics.get("bertscore_f1",0),
                           is_normalized=True, help_key="BERTScore F1")
        _metric_with_label(c3, "BERTScore Prec.", metrics.get("bertscore_precision",0),
                           is_normalized=True, help_key="BERTScore Prec.")
        _metric_with_label(c4, "BERTScore Recall", metrics.get("bertscore_recall",0),
                           is_normalized=True, help_key="BERTScore Recall")

    if has_gt:
        st.markdown("<div class='metric-section'><h5>📝 Overlap Metrics (vs Ground Truth)</h5></div>",
                    unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        _metric_with_label(c1, "ROUGE-L F1", metrics.get("rougeL_f1",0),
                           is_normalized=True, help_key="ROUGE-L F1")
        _metric_with_label(c2, "ROUGE-1 F1", metrics.get("rouge1_f1",0),
                           is_normalized=True, help_key="ROUGE-1 F1")
        _metric_with_label(c3, "BLEU",       metrics.get("bleu_score",0),
                           is_normalized=True, help_key="BLEU")
        _metric_with_label(c4, "METEOR",     metrics.get("meteor_score",0),
                           is_normalized=True, help_key="METEOR")

    st.markdown("<div class='metric-section'><h5>🔎 Retrieval Stats</h5></div>", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Chunks Retrieved", n_chunks,          help=METRIC_HELP.get("Chunks Retrieved",""))
    c2.metric("Unique Files", metrics.get("unique_files",0), help=METRIC_HELP.get("Unique Files",""))
    c3.metric("Avg Chunk Score", f"{avg_score:.4f}",  help=METRIC_HELP.get("Avg Chunk Score",""))
    c4.metric("Top File", metrics.get("top_file","none"), help=METRIC_HELP.get("Top File",""))

    st.markdown("<div class='metric-section'><h5>⏱️ Latency</h5></div>", unsafe_allow_html=True)
    if mode == "compare":
        c1,c2 = st.columns(2)
        c1.metric("Model Latency",  f"{metrics.get('model_latency_sec',0):.2f}s",
                  help=METRIC_HELP.get("Model Latency",""))
        c2.metric("Gemini Latency", f"{metrics.get('gemini_latency_sec',0):.2f}s",
                  help=METRIC_HELP.get("Gemini Latency",""))
    else:
        st.metric("Model Latency", f"{metrics.get('model_latency_sec',0):.2f}s",
                  help=METRIC_HELP.get("Model Latency",""))


# ==================================================
# COMPARISON TABLE (Screen 4)
# ==================================================

# Metrics used for "better/worse" verdict + green/red table coloring.
# Only QUALITY metrics belong here — metrics where higher = genuinely better.
# "Sem.Sim vs Gemini" is intentionally excluded: it measures divergence from
# Gemini, not quality. A model that gives a BETTER answer than Gemini will
# score low on this metric simply for being different.
_QUALITY_METRICS = [
    ("overall_score","Overall",True,0.4),
    ("faithfulness","Faithful.",True,0.4),
    ("completeness","Complete.",True,0.4),
    ("correctness","Correct.",True,0.4),
]
# GT-based metrics are also true quality metrics (vs an objective reference)
_GT_METRICS = [
    ("semantic_similarity_vs_gt","Sem.Sim vs GT",True,0.02),
    ("bertscore_f1","BERT F1",True,0.02),
    ("rougeL_f1","ROUGE-L",True,0.02),
    ("bleu_score","BLEU",True,0.02),
    ("meteor_score","METEOR",True,0.02),
]
# Informational-only metrics: shown in table but NEVER used for
# green/red coloring or better/worse verdict.
_INFO_METRICS = [
    ("semantic_similarity_vs_gemini","Sem.Sim vs Gemini"),
    ("bertscore_f1","BERT F1"),   # shown here when no GT; removed from GT block then
]
_LAT_METRIC = ("latency_sec","Latency (s)",False,1.0)


def build_comparison_table(all_results: dict, has_gt: bool):
    GEMINI = "Gemini"

    # ── Quality metrics: used for coloring AND verdict ─────────────────────
    quality_metrics = list(_QUALITY_METRICS)
    if has_gt:
        quality_metrics.extend(_GT_METRICS)

    # ── Informational metrics: shown in table but NO coloring / verdict ────
    # BERTScore appears here when no GT (divergence from Gemini ref),
    # and in quality block when GT is present (GT ref = objective).
    info_cols = [
        ("semantic_similarity_vs_gemini", "Sem.Sim vs Gemini"),
    ]
    if not has_gt:
        info_cols.append(("bertscore_f1", "BERT F1"))

    # Full column order for display
    all_display = (
        quality_metrics
        + [(rk, lbl, False, 0) for rk, lbl in info_cols]   # hb=False → no comparison
        + [_LAT_METRIC]
    )

    ordered = [GEMINI] + sorted(k for k in all_results if k != GEMINI)
    rows = {}
    for key in ordered:
        res = all_results.get(key, {})
        rows[key] = {lbl: res.get(rk, 0) for rk, lbl, *_ in all_display}

    raw_df = pd.DataFrame(rows).T
    raw_df.index.name = "Model"

    gem_vals = raw_df.loc[GEMINI]

    # ── Verdict: based ONLY on quality metrics ─────────────────────────────
    summaries = {GEMINI: "baseline"}
    for key in ordered:
        if key == GEMINI:
            continue
        better = worse = 0
        for _, lbl, hb, thr in quality_metrics:
            try:
                v_model = float(raw_df.loc[key, lbl])
                v_gem   = float(gem_vals[lbl])
                d = (v_model - v_gem) if hb else (v_gem - v_model)
                if d > thr:    better += 1
                elif d < -thr: worse  += 1
            except:
                pass
        summaries[key] = "better" if better > worse else ("worse" if worse > better else "similar")

    # ── Coloring ───────────────────────────────────────────────────────────
    # Build fast lookup: label → (is_quality, higher_is_better, threshold)
    quality_labels = {lbl: (hb, thr) for _, lbl, hb, thr in quality_metrics}
    info_labels    = {lbl for _, lbl in info_cols}
    info_labels.add("Latency (s)")   # latency shown but coloured separately

    def _color(df_):
        colors = pd.DataFrame("", index=df_.index, columns=df_.columns)
        for idx in df_.index:
            for col in df_.columns:
                if idx == GEMINI:
                    colors.loc[idx, col] = "background-color:#dbeafe;color:#1e3a8a;font-weight:bold"
                    continue
                if col in info_labels:
                    # Informational column: soft grey, no green/red verdict
                    colors.loc[idx, col] = "background-color:#f1f5f9;color:#475569"
                    continue
                if col == "Latency (s)":
                    # Lower latency = better (green), handled separately
                    try:
                        d = float(gem_vals[col]) - float(df_.loc[idx, col])
                        if d > 1.0:    colors.loc[idx, col] = "background-color:#bbf7d0;color:#14532d;font-weight:bold"
                        elif d < -1.0: colors.loc[idx, col] = "background-color:#fecaca;color:#7f1d1d"
                        else:          colors.loc[idx, col] = "background-color:#fefce8;color:#713f12"
                    except:
                        pass
                    continue
                if col in quality_labels:
                    hb, thr = quality_labels[col]
                    try:
                        d = (float(df_.loc[idx, col]) - float(gem_vals[col])) if hb \
                            else (float(gem_vals[col]) - float(df_.loc[idx, col]))
                        if d > thr:    colors.loc[idx, col] = "background-color:#bbf7d0;color:#14532d;font-weight:bold"
                        elif d < -thr: colors.loc[idx, col] = "background-color:#fecaca;color:#7f1d1d"
                        else:          colors.loc[idx, col] = "background-color:#fefce8;color:#713f12"
                    except:
                        pass
        return colors

    fmt = {}
    for rk, lbl, *_ in all_display:
        if lbl in ("Overall", "Faithful.", "Complete.", "Correct.", "Latency (s)"):
            fmt[lbl] = "{:.2f}"
        else:
            fmt[lbl] = "{:.4f}"

    styled = raw_df.style.apply(_color, axis=None).format(fmt)
    return raw_df, styled, summaries


# ==================================================
# ★ VISUAL CHARTS — Screen 4 ONLY
# ==================================================

# Colour palette for models (cycles if more than 8 models)
_CHART_PALETTE = [
    "#4f6ef7",  # blue (Gemini)
    "#22c55e",  # green
    "#f59e0b",  # amber
    "#ef4444",  # red
    "#8b5cf6",  # violet
    "#06b6d4",  # cyan
    "#f97316",  # orange
    "#ec4899",  # pink
]

_CHART_TOOLTIPS = {
    "judge_scores": (
        "<b>📊 Judge Score Chart</b><br>"
        "Shows the LLM judge's 0–10 scores for each model across four dimensions:<br>"
        "• <b>Faithfulness</b> — Are all claims grounded in the retrieved context? (0 = fabricated, 10 = fully sourced)<br>"
        "• <b>Completeness</b> — Does the answer cover all parts of the question? (0 = barely addressed, 10 = fully addressed)<br>"
        "• <b>Correctness</b> — Is the answer factually accurate vs the reference? (0 = wrong, 10 = correct)<br>"
        "• <b>Overall Score</b> — Weighted blend: Faithfulness×35% + Completeness×35% + Correctness×30%<br>"
        "Gemini is shown as the baseline (solid outline). Hover bars for exact values."
    ),
    "radar": (
        "<b>🕸️ Radar (Spider) Chart</b><br>"
        "Each axis represents one evaluation dimension, scaled 0–10.<br>"
        "A larger filled area = stronger model performance across all dimensions.<br>"
        "• The <b>shape</b> tells you where a model is strong vs weak.<br>"
        "• Models with a balanced polygon are well-rounded; spiked shapes indicate specialisation.<br>"
        "• Gemini's polygon is the benchmark — other models are compared against it.<br>"
        "Hover over each vertex to see the exact score for that metric."
    ),
    "semantic": (
        "<b>🧠 Semantic Similarity & Overlap Metrics Chart</b><br>"
        "Shows two <em>separate</em> semantic similarity comparisons per model:<br>"
        "• <b>Sem.Sim vs Gemini</b> — Cosine similarity of each model's answer to Gemini's answer. "
        "Always shown. Gemini = 1.0 (self). High value means the model agrees with Gemini.<br>"
        "• <b>Sem.Sim vs GT</b> — Cosine similarity of each model's answer to the Ground Truth. "
        "<em>Only shown when GT is uploaded and matched.</em> High value = close to the reference answer.<br>"
        "• <b>BERTScore F1</b> — Neural token-level similarity vs the reference. Higher = more semantically aligned.<br>"
        "• <b>ROUGE-L / BLEU / METEOR</b> — N-gram overlap metrics vs GT. Only meaningful with real GT.<br>"
        "All values are 0–1. Higher is better."
    ),
    "latency": (
        "<b>⏱️ Latency Comparison Chart</b><br>"
        "End-to-end generation time in seconds for each model.<br>"
        "Shorter bar = faster response. <b>Lower is better.</b><br>"
        "• Latency includes network round-trip + model inference time.<br>"
        "• Ollama models run on local GPU; Gemini uses a remote API endpoint.<br>"
        "• Very fast latency with a low quality score may indicate the model gave a short/incomplete answer.<br>"
        "Hover bars for exact seconds."
    ),
    "forest": (
        "<b>🌲 Forest Plot — Model Performance vs Gemini Baseline</b><br>"
        "Inspired by clinical meta-analysis forest plots. Each row = one model.<br>"
        "• <b>X-axis</b> — mean quality score difference vs Gemini (normalised 0–10). "
        "Positive (right of 0) = better; Negative (left) = worse.<br>"
        "• <b>Diff is computed on QUALITY metrics only</b>: Faithfulness, Completeness, Correctness, Overall Score"
        " — plus GT-based metrics (Sem.Sim vs GT, BERTScore, ROUGE, BLEU, METEOR) when GT is available.<br>"
        "• <b>'Sem.Sim vs Gemini' is intentionally excluded</b> from the diff calculation — "
        "it measures divergence from Gemini's answer, not quality. Including it would structurally "
        "bias every model as 'worse than Gemini' (model_sim_to_gemini − 1.0 is always negative).<br>"
        "• <b>Circle size</b> — proportional to the model's absolute overall score.<br>"
        "• <b>Error bars</b> — ±1 std dev across the quality metrics. Narrow = consistent; wide = mixed.<br>"
        "• <b>Grey diamond</b> — average across all Ollama models.<br>"
        "Select a model below to see the full per-metric breakdown."
    ),
    "retrieval": (
        "<b>🔎 Retrieval Quality Chart</b><br>"
        "Shows retrieval stats shared across all models (retrieval is done once for all):<br>"
        "• <b>Avg Chunk Score</b> — Mean cosine similarity of retrieved chunks to the query. "
        "Above 0.6 = high confidence retrieval; below 0.4 = weak match.<br>"
        "• <b>Chunks Retrieved</b> — Number of document chunks fed into the context window (max = TOP_K).<br>"
        "• <b>Unique Files</b> — How many distinct source files contributed to the context. "
        "Low diversity may mean the answer depends on a single document.<br>"
        "Note: all models share the same retrieved context — this chart is the same for all of them."
    ),
}


def _tooltip_box(key: str):
    """Render a descriptive tooltip box above a chart."""
    html = _CHART_TOOLTIPS.get(key, "")
    if html:
        st.markdown(
            f"<div class='chart-tooltip-box'>{html}</div>",
            unsafe_allow_html=True,
        )


def render_comparison_charts(all_results: dict, has_gt: bool, ref_label: str):
    """
    Render four interactive Plotly charts inside individual expanders.
    Called ONLY from render_comparison_screen (Screen 4).
    """
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        import numpy as np
    except ImportError:
        st.warning("Plotly not installed — charts unavailable. Run: `pip install plotly`")
        return

    GEMINI  = "Gemini"
    ordered = [GEMINI] + sorted(k for k in all_results if k != GEMINI)

    # Assign colours
    colour_map = {name: _CHART_PALETTE[i % len(_CHART_PALETTE)] for i, name in enumerate(ordered)}

    # ── Shared layout helper ──────────────────────────────────────────────────
    def _layout(fig, title: str, yaxis_title: str = "", yrange=None):
        fig.update_layout(
            title=dict(text=title, font=dict(size=15, color="#1e293b"), x=0.02),
            plot_bgcolor="#f8fafc",
            paper_bgcolor="white",
            font=dict(family="Calibri, sans-serif", size=12, color="#374151"),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1, bgcolor="rgba(255,255,255,0.8)",
                bordercolor="#e2e8f0", borderwidth=1,
            ),
            margin=dict(l=50, r=30, t=60, b=50),
            hovermode="x unified",
        )
        if yaxis_title:
            fig.update_yaxes(title_text=yaxis_title, gridcolor="#e2e8f0", zeroline=False)
        if yrange:
            fig.update_yaxes(range=yrange)
        fig.update_xaxes(gridcolor="#e2e8f0")
        return fig
    

        # ── 1. Forest Plot — Model Performance vs Gemini ──────────────────────────
    _ollama_models = [k for k in ordered if k != GEMINI]
    if _ollama_models:
        with st.expander("🌲 Forest Plot — Model Performance vs Gemini Baseline", expanded=True):
            _tooltip_box("forest")

            # ── Metrics used for forest plot DIFFERENCE calculation ────────
            # Only true QUALITY metrics — metrics where higher means genuinely better.
            # semantic_similarity_vs_gemini is deliberately excluded:
            # computing (model_sim_to_gemini − gemini_sim_to_self) = (x − 1.0)
            # is always negative, structurally biasing every model as "worse".
            # BERTScore is included only when GT is present (vs objective reference).
            FOREST_QUALITY_METRICS = {
                "faithfulness":  ("Faithfulness",  1.0),
                "completeness":  ("Completeness",  1.0),
                "correctness":   ("Correctness",   1.0),
                "overall_score": ("Overall Score", 1.0),
            }
            if has_gt:
                FOREST_QUALITY_METRICS["semantic_similarity_vs_gt"] = ("Sem.Sim vs GT",  10.0)
                FOREST_QUALITY_METRICS["bertscore_f1"]               = ("BERTScore F1",  10.0)
                FOREST_QUALITY_METRICS["rougeL_f1"]                  = ("ROUGE-L",       10.0)
                FOREST_QUALITY_METRICS["bleu_score"]                  = ("BLEU",          10.0)
                FOREST_QUALITY_METRICS["meteor_score"]                = ("METEOR",        10.0)

            # ── Metrics shown in drill-down table (includes informational) ──
            FOREST_DISPLAY_METRICS = dict(FOREST_QUALITY_METRICS)
            FOREST_DISPLAY_METRICS["semantic_similarity_vs_gemini"] = ("Sem.Sim vs Gemini ⓘ", 10.0)
            if not has_gt:
                FOREST_DISPLAY_METRICS["bertscore_f1"] = ("BERTScore F1 ⓘ", 10.0)

            gem_res_f = all_results.get(GEMINI, {})

            # Per-model: point estimate = mean diff across QUALITY metrics only
            forest_rows = []
            for name in _ollama_models:
                res   = all_results.get(name, {})
                diffs = []
                for mkey, (_, scale) in FOREST_QUALITY_METRICS.items():
                    raw_model = float(res.get(mkey, 0) or 0)
                    raw_gem   = float(gem_res_f.get(mkey, 0) or 0)
                    diffs.append((raw_model - raw_gem) * scale)
                mean_d = float(np.mean(diffs)) if diffs else 0.0
                std_d  = float(np.std(diffs))  if len(diffs) > 1 else 0.0
                lo     = mean_d - std_d
                hi     = mean_d + std_d
                overall = float(res.get("overall_score", 0) or 0)
                forest_rows.append({
                    "name":     name,
                    "mean":     mean_d,
                    "lo":       lo,
                    "hi":       hi,
                    "overall":  overall,
                    "diffs":    diffs,
                    "positive": mean_d >= 0,
                })

            # Sort: best (highest mean_d) at top
            forest_rows.sort(key=lambda r: r["mean"], reverse=True)

            # Average row (diamond)
            if forest_rows:
                avg_mean = float(np.mean([r["mean"] for r in forest_rows]))
                avg_lo   = float(np.mean([r["lo"]   for r in forest_rows]))
                avg_hi   = float(np.mean([r["hi"]   for r in forest_rows]))
            else:
                avg_mean = avg_lo = avg_hi = 0.0

            # import numpy as np  # ensure available inside function scope
            # Build figure
            fig_f = go.Figure()

            model_names_plot = [r["name"] for r in forest_rows] + ["── Average ──"]
            y_positions      = list(range(len(model_names_plot)))

            # Individual model rows
            for i, row in enumerate(forest_rows):
                dot_color  = "#16a34a" if row["positive"] else "#dc2626"
                ci_color   = "#86efac" if row["positive"] else "#fca5a5"
                # Circle size proportional to overall_score (6–22px range)
                dot_size = 8 + (row["overall"] / 10.0) * 18

                # CI line
                fig_f.add_trace(go.Scatter(
                    x=[row["lo"], row["hi"]],
                    y=[i, i],
                    mode="lines",
                    line=dict(color=ci_color, width=2.5),
                    showlegend=False,
                    hoverinfo="skip",
                ))
                # CI end caps
                for x_cap in [row["lo"], row["hi"]]:
                    fig_f.add_trace(go.Scatter(
                        x=[x_cap, x_cap],
                        y=[i - 0.12, i + 0.12],
                        mode="lines",
                        line=dict(color=ci_color, width=2),
                        showlegend=False,
                        hoverinfo="skip",
                    ))
                # Main dot
                hover_lines = [f"<b>{row['name']}</b>",
                               f"Mean Δ vs Gemini: <b>{row['mean']:+.3f}</b>",
                               f"Spread (±1σ): [{row['lo']:+.3f}, {row['hi']:+.3f}]",
                               f"Overall Score: {row['overall']:.2f}",
                               f"Verdict: {'✅ Better than Gemini' if row['positive'] else '❌ Worse than Gemini'}",
                               "─────────────────",
                               "Click model in selector below for full breakdown"]
                fig_f.add_trace(go.Scatter(
                    x=[row["mean"]],
                    y=[i],
                    mode="markers",
                    name=row["name"],
                    marker=dict(
                        color=dot_color,
                        size=dot_size,
                        symbol="circle",
                        line=dict(color="white", width=2),
                    ),
                    hovertemplate="<br>".join(hover_lines) + "<extra></extra>",
                    showlegend=True,
                ))

            # Average diamond
            avg_y = len(forest_rows)
            fig_f.add_trace(go.Scatter(
                x=[avg_lo, avg_hi],
                y=[avg_y, avg_y],
                mode="lines",
                line=dict(color="#94a3b8", width=2, dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            ))
            fig_f.add_trace(go.Scatter(
                x=[avg_mean],
                y=[avg_y],
                mode="markers",
                name="Average",
                marker=dict(
                    color="#64748b",
                    size=18,
                    symbol="diamond",
                    line=dict(color="white", width=2),
                ),
                hovertemplate=(
                    f"<b>Average across all models</b><br>"
                    f"Mean Δ: <b>{avg_mean:+.3f}</b><br>"
                    f"Range: [{avg_lo:+.3f}, {avg_hi:+.3f}]"
                    "<extra></extra>"
                ),
                showlegend=True,
            ))

            # Vertical zero line (Gemini baseline)
            fig_f.add_vline(
                x=0, line_width=2, line_dash="dash",
                line_color="#4f6ef7",
                annotation_text="Gemini (0)",
                annotation_position="top",
                annotation_font=dict(color="#4f6ef7", size=11),
            )

            # Shaded regions: positive = light green, negative = light red
            x_min = min([r["lo"] for r in forest_rows] + [avg_lo]) - 0.3
            x_max = max([r["hi"] for r in forest_rows] + [avg_hi]) + 0.3
            x_min = min(x_min, -0.5); x_max = max(x_max, 0.5)

            fig_f.add_vrect(x0=0, x1=x_max, fillcolor="#dcfce7", opacity=0.15, layer="below")
            fig_f.add_vrect(x0=x_min, x1=0, fillcolor="#fef2f2", opacity=0.20, layer="below")

            # Better / Worse annotations
            fig_f.add_annotation(
                x=x_max * 0.7, y=len(model_names_plot) - 0.2,
                text="▶ Better than Gemini", showarrow=False,
                font=dict(color="#16a34a", size=11, family="Calibri"),
                xanchor="center",
            )
            fig_f.add_annotation(
                x=x_min * 0.7, y=len(model_names_plot) - 0.2,
                text="◀ Worse than Gemini", showarrow=False,
                font=dict(color="#dc2626", size=11, family="Calibri"),
                xanchor="center",
            )

            fig_f.update_layout(
                title=dict(
                    text="Forest Plot — Model Performance Difference vs Gemini",
                    font=dict(size=15, color="#1e293b"), x=0.02,
                ),
                xaxis=dict(
                    title="Mean Metric Difference vs Gemini (normalised 0–10 scale)",
                    gridcolor="#e2e8f0", zeroline=False,
                    range=[x_min, x_max],
                ),
                yaxis=dict(
                    tickvals=y_positions,
                    ticktext=model_names_plot,
                    tickfont=dict(size=12, family="Calibri"),
                    gridcolor="#f1f5f9",
                    zeroline=False,
                ),
                plot_bgcolor="#f8fafc",
                paper_bgcolor="white",
                font=dict(family="Calibri, sans-serif", size=12, color="#374151"),
                legend=dict(
                    title="Models",
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#e2e8f0", borderwidth=1,
                    yanchor="middle", y=0.5,
                    xanchor="right", x=1.13,
                ),
                height=max(340, 60 + len(model_names_plot) * 58),
                margin=dict(l=20, r=160, t=70, b=60),
                hovermode="closest",
            )
            st.plotly_chart(fig_f, use_container_width=True)

            # ── Interactive drill-down: select model → full metric breakdown ──
            st.markdown("---")
            st.markdown(
                "##### 🔎 Model Drill-Down — select a model to see why it is positive or negative"
            )
            drill_options = ["(select a model…)"] + [r["name"] for r in forest_rows]
            drill_key     = f"forest_drill_{id(all_results)}"
            chosen = st.selectbox(
                "Model to inspect", drill_options, key=drill_key, label_visibility="collapsed"
            )

            if chosen and chosen != "(select a model…)":
                sel_res  = all_results.get(chosen, {})
                gem_res_ = all_results.get(GEMINI, {})

                # ── Verdict banner ────────────────────────────────────────────
                row_sel  = next((r for r in forest_rows if r["name"] == chosen), None)
                if row_sel:
                    is_pos = row_sel["positive"]
                    bg_col = "#dcfce7" if is_pos else "#fef2f2"
                    fg_col = "#15803d" if is_pos else "#dc2626"
                    icon   = "✅" if is_pos else "❌"
                    verdict_txt = "Better than Gemini" if is_pos else "Worse than Gemini"
                    st.markdown(
                        f"<div style='background:{bg_col};color:{fg_col};padding:12px 18px;"
                        f"border-radius:10px;font-size:15px;font-weight:700;margin-bottom:14px'>"
                        f"{icon} <b>{chosen}</b> — {verdict_txt} "
                        f"(Mean Δ = {row_sel['mean']:+.3f})"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                # ── Per-metric horizontal bar chart: model vs Gemini ──────────
                metric_labels_ = []
                model_vals_    = []
                gemini_vals_   = []
                diff_vals_     = []
                is_quality_col = []   # True = quality metric, False = informational

                for mkey, (mlabel, scale) in FOREST_DISPLAY_METRICS.items():
                    mv  = float(sel_res.get(mkey, 0) or 0) * scale
                    gv  = float(gem_res_.get(mkey, 0) or 0) * scale
                    d   = mv - gv
                    metric_labels_.append(mlabel)
                    model_vals_.append(mv)
                    gemini_vals_.append(gv)
                    diff_vals_.append(d)
                    is_quality_col.append(mkey in FOREST_QUALITY_METRICS)

                fig_drill = go.Figure()

                # Gemini baseline
                fig_drill.add_trace(go.Bar(
                    name="Gemini (baseline)",
                    y=metric_labels_,
                    x=gemini_vals_,
                    orientation="h",
                    marker_color="#93c5fd",
                    marker_line_color="#3b82f6",
                    marker_line_width=1.5,
                    opacity=0.85,
                    hovertemplate="<b>Gemini</b><br>%{y}: %{x:.3f}<extra></extra>",
                ))

                # Selected model
                fig_drill.add_trace(go.Bar(
                    name=chosen,
                    y=metric_labels_,
                    x=model_vals_,
                    orientation="h",
                    marker_color=colour_map.get(chosen, "#4f6ef7"),
                    opacity=0.9,
                    hovertemplate=f"<b>{chosen}</b><br>%{{y}}: %{{x:.3f}}<extra></extra>",
                ))

                # Δ annotations on the right
                for i, (lbl, d, is_q) in enumerate(zip(metric_labels_, diff_vals_, is_quality_col)):
                    if is_q:
                        clr = "#15803d" if d >= 0 else "#dc2626"
                    else:
                        clr = "#64748b"   # grey for informational metrics
                    fig_drill.add_annotation(
                        y=lbl, x=max(model_vals_[i], gemini_vals_[i]) + 0.15,
                        text=f"{d:+.3f}" + ("" if is_q else " ⓘ"),
                        showarrow=False,
                        font=dict(color=clr, size=11, family="Calibri"),
                        xanchor="left",
                    )

                fig_drill.update_layout(
                    barmode="group",
                    title=dict(
                        text=f"Per-Metric Breakdown: {chosen} vs Gemini (all normalised 0–10)",
                        font=dict(size=14, color="#1e293b"), x=0.02,
                    ),
                    xaxis=dict(title="Score (0–10)", gridcolor="#e2e8f0"),
                    yaxis=dict(tickfont=dict(size=11)),
                    plot_bgcolor="#f8fafc",
                    paper_bgcolor="white",
                    font=dict(family="Calibri, sans-serif", size=12),
                    legend=dict(orientation="h", y=1.1, x=0),
                    margin=dict(l=130, r=80, t=60, b=40),
                    height=max(320, 50 + len(metric_labels_) * 40),
                )
                st.plotly_chart(fig_drill, use_container_width=True)

                # ── Quick summary table ───────────────────────────────────────
                summary_rows = []
                for lbl, mv, gv, d, is_q in zip(
                    metric_labels_, model_vals_, gemini_vals_, diff_vals_, is_quality_col
                ):
                    if not is_q:
                        verdict = "ℹ️ Informational"
                    elif d > 0.1:
                        verdict = "🟢 Better"
                    elif d < -0.1:
                        verdict = "🔴 Worse"
                    else:
                        verdict = "🟡 Similar"
                    summary_rows.append({
                        "Metric":        lbl,
                        f"{chosen}":     f"{mv:.3f}",
                        "Gemini":        f"{gv:.3f}",
                        "Δ":             f"{d:+.3f}",
                        "Verdict":       verdict,
                    })
                st.dataframe(
                    pd.DataFrame(summary_rows).set_index("Metric"),
                    use_container_width=True,
                    height=min(400, 40 + len(summary_rows) * 38),
                )
                st.caption(
                    "ⓘ **Informational metrics** (grey) are shown for reference only and are "
                    "**not used** for the better/worse verdict. "
                    "'Sem.Sim vs Gemini' measures divergence from Gemini's answer, not quality — "
                    "a model giving a better answer than Gemini would still score low here."
                )


    # ── 2. Judge Scores — Grouped Bar ────────────────────────────────────────
    with st.expander("📊 Judge Scores Chart — Faithfulness, Completeness, Correctness, Overall", expanded=True):
        _tooltip_box("judge_scores")
        judge_keys   = ["faithfulness", "completeness", "correctness", "overall_score"]
        judge_labels = ["Faithfulness", "Completeness", "Correctness", "Overall Score"]

        fig1 = go.Figure()
        for name in ordered:
            res    = all_results.get(name, {})
            values = [res.get(k, 0) for k in judge_keys]
            is_gem = name == GEMINI
            fig1.add_trace(go.Bar(
                name=name,
                x=judge_labels,
                y=values,
                marker_color=colour_map[name],
                marker_line_color="#1e293b" if is_gem else colour_map[name],
                marker_line_width=2 if is_gem else 0,
                opacity=0.95 if is_gem else 0.82,
                text=[f"{v:.2f}" for v in values],
                textposition="outside",
                hovertemplate=f"<b>{name}</b><br>%{{x}}: %{{y:.2f}}<extra></extra>",
            ))

        fig1.add_hrect(y0=8, y1=10, fillcolor="#dcfce7", opacity=0.25, layer="below",
                       annotation_text="Excellent zone", annotation_position="top right",
                       annotation_font_size=10, annotation_font_color="#15803d")
        fig1.add_hrect(y0=0, y1=3, fillcolor="#fecaca", opacity=0.2, layer="below")

        fig1 = _layout(fig1, "LLM Judge Scores by Model (0–10)", "Score (0–10)", yrange=[0, 11.5])
        fig1.update_layout(barmode="group")
        st.plotly_chart(fig1, use_container_width=True)

    # ── 3. Radar Chart ────────────────────────────────────────────────────────
    with st.expander("🕸️ Radar Chart — Multi-dimensional Model Comparison", expanded=False):
        _tooltip_box("radar")
        radar_keys   = ["faithfulness", "completeness", "correctness",
                        "semantic_similarity", "bertscore_f1"]
        radar_labels = ["Faithfulness", "Completeness", "Correctness",
                        "Semantic Sim.\n(×10)", "BERTScore F1\n(×10)"]

        if has_gt:
            radar_keys   += ["rougeL_f1", "bleu_score"]
            radar_labels += ["ROUGE-L (×10)", "BLEU (×10)"]

        fig2 = go.Figure()
        for name in ordered:
            res = all_results.get(name, {})
            # Normalise semantic/overlap metrics to 0-10 scale for radar
            vals = []
            for k in radar_keys:
                v = float(res.get(k, 0))
                # judge scores are already 0-10; others are 0-1
                if k not in ("faithfulness", "completeness", "correctness", "overall_score"):
                    v = v * 10.0
                vals.append(v)
            # Close the polygon
            cats  = radar_labels + [radar_labels[0]]
            vals_ = vals + [vals[0]]
            is_gem = name == GEMINI
            fig2.add_trace(go.Scatterpolar(
                r=vals_,
                theta=cats,
                fill="toself",
                name=name,
                line=dict(color=colour_map[name], width=3 if is_gem else 2,
                          dash="solid" if is_gem else "dot"),
                fillcolor=colour_map[name],
                opacity=0.35 if not is_gem else 0.2,
                hovertemplate=f"<b>{name}</b><br>%{{theta}}: %{{r:.2f}}<extra></extra>",
            ))

        fig2.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 10], tickfont_size=10,
                                gridcolor="#e2e8f0", linecolor="#cbd5e1"),
                angularaxis=dict(tickfont_size=11, gridcolor="#e2e8f0"),
                bgcolor="#f8fafc",
            ),
            title=dict(text="Multi-Dimensional Model Comparison (all axes: 0–10)",
                       font=dict(size=15, color="#1e293b"), x=0.02),
            paper_bgcolor="white",
            font=dict(family="Calibri, sans-serif", size=12),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15,
                        xanchor="center", x=0.5),
            margin=dict(l=60, r=60, t=60, b=80),
            showlegend=True,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── 4. Semantic & Overlap Metrics ─────────────────────────────────────────
    with st.expander(
        f"🧠 Semantic Similarity & Overlap Metrics", expanded=False
    ):
        _tooltip_box("semantic")

        sem_keys   = ["semantic_similarity_vs_gemini"]
        sem_labels = ["Sem.Sim vs Gemini"]
        if has_gt:
            sem_keys   = ["semantic_similarity_vs_gt", "semantic_similarity_vs_gemini"]
            sem_labels = ["Sem.Sim vs GT", "Sem.Sim vs Gemini"]
        sem_keys   += ["bertscore_f1"]
        sem_labels += ["BERTScore F1"]
        if has_gt:
            sem_keys   += ["rougeL_f1", "bleu_score", "meteor_score"]
            sem_labels += ["ROUGE-L", "BLEU", "METEOR"]

        fig3 = go.Figure()
        for name in ordered:
            res    = all_results.get(name, {})
            values = [float(res.get(k, 0)) for k in sem_keys]
            is_gem = name == GEMINI
            fig3.add_trace(go.Bar(
                name=name,
                x=sem_labels,
                y=values,
                marker_color=colour_map[name],
                marker_line_color="#1e293b" if is_gem else colour_map[name],
                marker_line_width=2 if is_gem else 0,
                opacity=0.95 if is_gem else 0.82,
                text=[f"{v:.4f}" for v in values],
                textposition="outside",
                hovertemplate=f"<b>{name}</b><br>%{{x}}: %{{y:.4f}}<extra></extra>",
            ))

        # Annotation for what the reference is
        ref_note = (
            "Semantic Sim. & BERTScore: vs <b>Ground Truth</b>"
            if has_gt
            else "Semantic Sim. & BERTScore: vs <b>Gemini's answer</b> (no GT uploaded)"
        )
        fig3.add_annotation(
            text=ref_note, xref="paper", yref="paper", x=0.01, y=-0.14,
            showarrow=False, font=dict(size=11, color="#6b7280"),
            align="left", bgcolor="rgba(255,255,255,0.7)",
        )

        fig3 = _layout(fig3, f"Semantic & Overlap Metrics vs {ref_label} (0–1 scale)",
                       "Score (0–1)", yrange=[0, 1.15])
        fig3.update_layout(barmode="group")
        st.plotly_chart(fig3, use_container_width=True)

    # ── 5. Latency ────────────────────────────────────────────────────────────
    with st.expander("⏱️ Latency Comparison — Generation Speed per Model", expanded=False):
        _tooltip_box("latency")

        latencies = {name: float(all_results.get(name, {}).get("latency_sec", 0))
                     for name in ordered}
        sorted_names = sorted(latencies, key=latencies.get)  # fastest first

        colours_sorted = [colour_map[n] for n in sorted_names]
        values_sorted  = [latencies[n] for n in sorted_names]

        fig4 = go.Figure(go.Bar(
            x=sorted_names,
            y=values_sorted,
            marker_color=colours_sorted,
            marker_line_color=["#1e293b" if n == GEMINI else c
                               for n, c in zip(sorted_names, colours_sorted)],
            marker_line_width=[2 if n == GEMINI else 0 for n in sorted_names],
            text=[f"{v:.2f}s" for v in values_sorted],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Latency: %{y:.2f}s<extra></extra>",
        ))
        fig4 = _layout(fig4, "Generation Latency per Model (lower = faster)", "Seconds")
        fig4.update_layout(showlegend=False,
                           yaxis=dict(range=[0, max(values_sorted) * 1.25 + 0.5]))
        # Fastest model annotation
        fastest = sorted_names[0]
        fig4.add_annotation(
            x=fastest, y=latencies[fastest],
            text=f"⚡ Fastest", showarrow=True, arrowhead=2, arrowcolor="#15803d",
            font=dict(color="#15803d", size=11),
            ay=-35, ax=0,
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── 6. Retrieval Quality (shared context) ─────────────────────────────────
    with st.expander("🔎 Retrieval Quality — Shared Context Stats", expanded=False):
        _tooltip_box("retrieval")

        # Retrieval stats are the same for all models (shared retrieval)
        # Show from Gemini result as representative
        gem_res      = all_results.get(GEMINI, {})
        avg_score_   = float(gem_res.get("avg_chunk_score", 0))
        chunks_retr  = int(gem_res.get("chunks_retrieved", 0))
        unique_files = int(gem_res.get("unique_files", 0))

        col1, col2 = st.columns([2, 1])
        with col1:
            # Gauge for avg chunk score
            fig5 = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=avg_score_,
                title=dict(text="Average Chunk Relevance Score", font=dict(size=14)),
                gauge=dict(
                    axis=dict(range=[0, 1], tickwidth=1, tickcolor="#374151"),
                    bar=dict(color="#4f6ef7"),
                    bgcolor="white",
                    borderwidth=2,
                    bordercolor="#e2e8f0",
                    steps=[
                        dict(range=[0, 0.4],  color="#fecaca"),
                        dict(range=[0.4, 0.6], color="#fef9c3"),
                        dict(range=[0.6, 1.0], color="#bbf7d0"),
                    ],
                    threshold=dict(
                        line=dict(color="#ef4444", width=4),
                        thickness=0.75, value=0.4,
                    ),
                ),
                number=dict(suffix="", font=dict(size=28, color="#1e293b"),
                            valueformat=".4f"),
            ))
            fig5.update_layout(
                paper_bgcolor="white", margin=dict(l=30, r=30, t=50, b=20), height=280,
                font=dict(family="Calibri, sans-serif"),
            )
            st.plotly_chart(fig5, use_container_width=True)

        with col2:
            st.markdown("#### Retrieval Summary")
            zone = ("🟢 High confidence" if avg_score_ >= 0.6
                    else ("🟡 Medium confidence" if avg_score_ >= 0.4
                          else "🔴 Low confidence"))
            st.metric("Confidence Zone", zone)
            st.metric("Chunks Retrieved", f"{chunks_retr} / {TOP_K}",
                      help=METRIC_HELP.get("Chunks Retrieved",""))
            st.metric("Unique Source Files", unique_files,
                      help=METRIC_HELP.get("Unique Files",""))
            st.metric("Top File", gem_res.get("top_file","N/A"),
                      help=METRIC_HELP.get("Top File",""))
            st.caption("All models share the same retrieved context.")



# ==================================================
# COMPARISON TABLE + CHARTS (Screen 4)
# ==================================================

def render_comparison_screen(comp_data: dict, question: str, judge_model: str = JUDGE_GPT):
    all_results = comp_data["results"]
    has_gt      = comp_data["has_ground_truth"]
    gt_entry    = comp_data.get("ground_truth")
    ref_label   = comp_data.get("reference_label","Gemini Answer")
    sources     = comp_data.get("retrieved_sources",[])

    gem_res = all_results.get("Gemini",{})
    render_confidence_indicator(gem_res.get("avg_chunk_score",0.0), gem_res.get("chunks_retrieved",0))

    st.markdown(f"### 📌 Question\n> {question}")
    st.markdown(
        f"<span class='judge-badge'>⚖️ Judge: {judge_model}</span> &nbsp;"
        f"**{len(all_results)} models evaluated** | Reference: **{ref_label}** | "
        f"GT: {'✅ Found' if has_gt else '❌ Not found'}",
        unsafe_allow_html=True)

    # Semantic similarity reference explanation — two separate columns
    if has_gt:
        st.info(
            "📌 **Semantic Similarity columns:** "
            "**'Sem.Sim vs Gemini'** — every model's answer compared to Gemini's answer (always shown). "
            "**'Sem.Sim vs GT'** — every model's answer compared to the Ground Truth answer (shown because GT is uploaded).",
            icon=None,
        )
    else:
        st.info(
            "📌 **Semantic Similarity:** "
            "**'Sem.Sim vs Gemini'** — every model's answer compared to Gemini's answer. "
            "The 'Sem.Sim vs GT' column is hidden because no Ground Truth was uploaded/matched.",
            icon=None,
        )

    if has_gt and gt_entry: render_gt_box(gt_entry)

    # Legend
    st.markdown(
        "<div style='display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px'>"
        "<b style='align-self:center'>vs Gemini:</b>"
        "<span style='background:#bbf7d0;color:#14532d;padding:3px 10px;border-radius:8px;font-size:13px;font-weight:600'>🟢 Better</span>"
        "<span style='background:#fecaca;color:#7f1d1d;padding:3px 10px;border-radius:8px;font-size:13px;font-weight:600'>🔴 Worse</span>"
        "<span style='background:#fefce8;color:#713f12;padding:3px 10px;border-radius:8px;font-size:13px;font-weight:600'>🟡 Similar</span>"
        "<span style='background:#dbeafe;color:#1e40af;padding:3px 10px;border-radius:8px;font-size:13px;font-weight:600'>🔷 Gemini (Baseline)</span>"
        "</div>", unsafe_allow_html=True)

    # Comparison table
    st.subheader("📊 Model Comparison Table")
    raw_df, styled_df, summaries = build_comparison_table(all_results, has_gt)
    st.dataframe(styled_df, use_container_width=True, height=min(60+len(all_results)*45,420))
    st.caption(
        "🟢 Green / 🔴 Red / 🟡 Yellow coloring is based on **quality metrics only** "
        "(Judge scores + GT-based metrics when available). "
        "**Grey columns** (Sem.Sim vs Gemini) are informational — they measure how similar a "
        "model's answer is to Gemini's answer, not how correct it is. A model giving a "
        "*better* answer than Gemini would still score low here."
    )

    # Per-model verdict badges
    st.markdown("#### Overall Verdict vs Gemini")
    vcols = st.columns(min(len(all_results),4))
    for ci,(key,verdict) in enumerate(summaries.items()):
        icon  = {"better":"🟢","worse":"🔴","similar":"🟡","baseline":"🔷"}.get(verdict,"")
        text  = {"better":"Better","worse":"Worse","similar":"Similar","baseline":"Baseline (Gemini)"}.get(verdict,"")
        score_v = all_results.get(key,{}).get("overall_score",0)
        lbl, bg, fg = score_label(score_v)
        vcols[ci%len(vcols)].markdown(
            f"**{key}**<br>"
            f"<span>{icon} {text}</span>&nbsp;"
            f"<span class='score-label' style='background:{bg};color:{fg}'>{lbl}</span>",
            unsafe_allow_html=True)

    # Winner callout
    model_scores = {k:v.get("overall_score",0) for k,v in all_results.items()}
    winner = max(model_scores, key=model_scores.get)
    ws     = model_scores[winner]; gs = model_scores.get("Gemini",0)
    beat   = f" — beats Gemini by +{ws-gs:.2f}" if winner!="Gemini" and ws>gs else ""
    st.markdown(f"<div class='winner-card'>🏆 <b>Best Overall:</b> {winner} (score {ws:.2f}{beat})</div>",
                unsafe_allow_html=True)

    # ── ★ VISUAL CHARTS (Screen 4 only) ──────────────────────────────────────
    st.markdown("---")
    st.subheader("📈 Visual Comparison Charts")
    st.caption(
        "Expand each chart below for an interactive view. "
        "Each chart includes a tooltip explaining what is being measured and how to interpret the results."
    )
    render_comparison_charts(all_results, has_gt, ref_label)
    # ─────────────────────────────────────────────────────────────────────────

    # Retrieved sources
    if sources:
        with st.expander("📂 Retrieved Source Documents"):
            for i,s in enumerate(sources,1): st.write(f"{i}. {s}")

    # Export button
    render_export_button(question=question, comparison=comp_data, judge_model=judge_model,
                         key_suffix="comparison")

    # Individual model answers (expandable) — with Answer Diff vs Gemini and vs GT
    st.subheader("💬 Model Answers")

    gemini_answer_text = all_results.get("Gemini", {}).get("answer", "")
    gt_answer_text     = gt_entry.get("answer", "") if (has_gt and gt_entry) else ""

    for key in (["Gemini"] + sorted(k for k in all_results if k != "Gemini")):
        res       = all_results.get(key, {})
        answer_t  = res.get("answer", "No answer.")
        verdict   = summaries.get(key, "similar")
        icon      = {"better":"🟢","worse":"🔴","similar":"🟡","baseline":"🔷"}.get(verdict,"")
        lbl, bg, fg = score_label(res.get("overall_score", 0))

        with st.expander(
            f"{icon} {key} — Overall: {res.get('overall_score',0):.2f} "
            f"({lbl}) | F:{res.get('faithfulness',0)} C:{res.get('completeness',0)} "
            f"Cor:{res.get('correctness',0)} | {res.get('latency_sec',0):.1f}s"
        ):
            # ── Answer text ──────────────────────────────────────────────────
            answer_card(key, answer_t)
            if res.get("judge_explanation"):
                st.caption(f"💬 Judge: {res['judge_explanation']}")

            # ── Diff vs Gemini (shown for every model except Gemini itself) ──
            if key != "Gemini" and gemini_answer_text and answer_t:
                sim_vs_gem = calculate_semantic_similarity(answer_t, gemini_answer_text)
                render_answer_diff(
                    answer_t,
                    gemini_answer_text,
                    key,
                    "Gemini",
                    similarity=sim_vs_gem,
                )

            # ── Diff vs Ground Truth (shown when GT is available, all models) ─
            if gt_answer_text and answer_t:
                sim_vs_gt = calculate_semantic_similarity(answer_t, gt_answer_text)
                render_answer_diff(
                    answer_t,
                    gt_answer_text,
                    key,
                    "Ground Truth",
                    similarity=sim_vs_gt,
                )


# ==================================================
# QDRANT COLLECTION CACHE — 5-minute TTL
# ==================================================
COLLECTION_TTL = 300   # seconds


def _clear_cached_collection():
    """Delete the cached Qdrant collection and wipe session state."""
    cname = st.session_state.get("cached_collection")
    if cname:
        try:
            delete_temp_collection(cname)
        except Exception:
            pass
    st.session_state.cached_collection    = None
    st.session_state.cached_collection_at = 0.0
    st.session_state.cached_run_path      = None


def ensure_indexed_collection(run_path: Path) -> str:
    """
    Return a valid, indexed Qdrant collection name for `run_path`.

    • If a cached collection exists for the SAME run_path AND was created
      less than COLLECTION_TTL seconds ago → reuse it (no re-indexing).
    • Otherwise → delete any stale collection, create + index a new one,
      store it in session state, and return its name.
    """
    now          = time.time()
    run_path_str = str(run_path)
    cached       = st.session_state.get("cached_collection")
    cached_at    = st.session_state.get("cached_collection_at", 0.0)
    cached_path  = st.session_state.get("cached_run_path")
    age          = now - cached_at
    remaining    = COLLECTION_TTL - age

    if cached and cached_path == run_path_str and remaining > 0:
        # ── CACHE HIT ────────────────────────────────────────────────────────
        st.toast(
            f"♻️ Reusing indexed collection — {int(remaining)}s until expiry",
            icon="✅",
        )
        return cached

    # ── CACHE MISS / EXPIRED ─────────────────────────────────────────────────
    if cached:
        try:
            delete_temp_collection(cached)
        except Exception:
            pass

    cname  = f"temp_{uuid.uuid4().hex[:8]}"
    client = create_temp_collection(cname)
    n      = index_uploaded_files(client, cname, run_path)
    if n == 0:
        delete_temp_collection(cname)
        raise ValueError(
            "No content extracted. "
            "Supported: PDF, DOCX, TXT, MD, CSV, XLSX, JSON, HTML, PPTX, XML, YAML."
        )

    st.session_state.cached_collection    = cname
    st.session_state.cached_collection_at = time.time()
    st.session_state.cached_run_path      = run_path_str
    st.toast(f"✅ Indexed new collection ({n} chunks). Will reuse for {COLLECTION_TTL//60} min.", icon="📚")
    return cname


# ==================================================
# SINGLE-MODEL GENERATE FLOW
# ==================================================

def run_generate_only(selected_model, question, run_path, gt_entry=None, judge_model=JUDGE_GPT):
    import time as _time
    # ── Use cached collection (creates new one only when needed) ──────────────
    cname = ensure_indexed_collection(run_path)

    rag  = RAGSystem(QDRANT_URL, cname)
    docs, context, sources = rag.retrieve(question, top_k=TOP_K)
    if not docs:
        raise ValueError("Retrieval returned no results.")

    prompt = rag.build_prompt(question, context)
    t0 = _time.time(); answer = generate_with_ollama(selected_model, prompt); lat = _time.time()-t0

    gt_ans    = gt_entry.get("answer","") if gt_entry else ""
    reference = gt_ans

    from Gemini_model import _build_eval_result
    res = _build_eval_result(selected_model, answer, lat, question, context, reference,
                              docs, sources, judge_model=judge_model)
    # NOTE: do NOT delete the collection — it is cached for COLLECTION_TTL seconds

    res["model_latency_sec"] = round(lat, 2)
    res["reference_label"]   = "Ground Truth" if gt_ans else "Context Only"
    res["has_ground_truth"]  = bool(gt_ans)
    res.pop("answer", None)
    return {"model_answer": answer, "model_sources": sources, "metrics": res, "gt_entry": gt_entry}


# ==================================================
# SESSION STATE
# ==================================================
_defaults = {
    "screen": 1, "run_path": None, "uploaded_names": [],
    "model_answer": "", "gemini_answer": "", "metrics": {},
    "metric_mode": "compare", "model_sources": [],
    "selected_model": None, "last_question": "",
    "gt_data": None, "gt_file_name": "", "gt_entry": None,
    "selected_models_list": [],
    "comparison_data": None,
    "judge_model": JUDGE_GPT,
    # ── Qdrant collection cache ─────────────────────────────────────────────
    "cached_collection":    None,
    "cached_collection_at": 0.0,
    "cached_run_path":      None,
}
for k, v in _defaults.items():
    if k not in st.session_state: st.session_state[k] = v

try:
    ALL_MODELS = get_available_ollama_models() or []
except Exception:
    ALL_MODELS = []


def get_gen_models():
    jm = st.session_state.judge_model
    if jm and jm != JUDGE_GPT:
        return [m for m in ALL_MODELS if m != jm] or ["No Models Found"]
    return ALL_MODELS or ["No Models Found"]

st.markdown("<h1 class='main-title'>🚀 RAG Testing Dashboard</h1>", unsafe_allow_html=True)


# ==================================================
# SCREEN 1 — SETUP
# ==================================================
if st.session_state.screen == 1:

    st.markdown("### 📂 Step 1 — Upload Documents")
    if st.session_state.run_path and st.session_state.uploaded_names:
        st.success(f"✅ Files loaded: {', '.join(st.session_state.uploaded_names)}")
        with st.expander("🗂️ Document Preview", expanded=True):
            render_document_preview(st.session_state.run_path)
        if st.button("🔄 Re-upload Files"):
            _clear_cached_collection()
            st.session_state.run_path = None; st.session_state.uploaded_names = []
            st.rerun()
    else:
        st.info("Supported: **PDF, DOCX, TXT, MD, CSV, XLSX, JSON, HTML, PPTX, XML, YAML** or a **ZIP**.")
        uploaded = st.file_uploader("Upload document files", accept_multiple_files=True, type=None)
        if uploaded:
            _clear_cached_collection()
            st.session_state.run_path       = save_and_extract(uploaded)
            st.session_state.uploaded_names = [f.name for f in uploaded]
            st.rerun()

    st.divider()

    st.markdown("### 📋 Step 2 — Upload Ground Truth JSON (Optional)")
    if st.session_state.gt_data:
        n_q = len(st.session_state.gt_data.get("questions",[]))
        st.success(f"✅ GT loaded: **{st.session_state.gt_file_name}** ({n_q} questions)")
        if st.button("🗑️ Remove Ground Truth"):
            st.session_state.gt_data = None; st.session_state.gt_file_name = ""
            st.session_state.gt_entry = None; st.rerun()
    else:
        gt_file = st.file_uploader("Upload ground truth .json file", type=["json"], key="gt_uploader")
        if gt_file:
            try:
                gt_data = json.load(gt_file)
                if "questions" not in gt_data:
                    st.error("JSON must have a 'questions' array.")
                else:
                    st.session_state.gt_data      = gt_data
                    st.session_state.gt_file_name = gt_file.name
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to parse JSON: {e}")

    st.divider()

    st.markdown("### ⚖️ Step 3 — Select Judge Model")
    judge_options = [JUDGE_GPT] + ALL_MODELS
    current_judge_idx = judge_options.index(st.session_state.judge_model) \
        if st.session_state.judge_model in judge_options else 0
    chosen_judge = st.selectbox(
        "Judge Model",
        options=judge_options,
        index=current_judge_idx,
        help=(
            "The judge evaluates answer quality (Faithfulness, Completeness, Correctness). "
            f"'{JUDGE_GPT}' uses the Bosch GPT-4o-mini endpoint. "
            "Choosing an Ollama model removes it from the generation list to avoid self-judging."
        ),
    )
    st.session_state.judge_model = chosen_judge
    if chosen_judge != JUDGE_GPT:
        st.warning(
            f"⚠️ **{chosen_judge}** selected as judge — it will be removed from the generation "
            "model lists to prevent self-judging."
        )
    else:
        st.caption("Using GPT-4o-mini as judge (default). No generation models removed.")

    st.divider()

    st.markdown("### ❓ Step 4 — Enter Question")
    question = st.text_area("Your question", value=st.session_state.last_question, height=80)

    if question.strip() and st.session_state.gt_data:
        gt_entry = find_ground_truth(question, st.session_state.gt_data)
        st.session_state.gt_entry = gt_entry
        col_b, col_t = st.columns([1,5])
        col_b.markdown(gt_badge(gt_entry), unsafe_allow_html=True)
        if gt_entry:
            with col_t.expander("Preview matched GT"):
                st.markdown(f"**Q:** {gt_entry.get('question','')}")
                st.markdown(f"**A:** {gt_entry.get('answer','')[:300]}…")
    else:
        st.session_state.gt_entry = None

    st.divider()

    st.markdown("### 🤖 Section A — Single Model")
    GEN_MODELS = get_gen_models()
    selected_model = st.selectbox(
        "Select Model", GEN_MODELS,
        index=GEN_MODELS.index(st.session_state.selected_model)
              if st.session_state.selected_model in GEN_MODELS else 0,
    )
    st.session_state.selected_model = selected_model

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Generate Answer", use_container_width=True):
            if st.session_state.run_path and selected_model not in ("No Models Found",) and question.strip():
                st.session_state.last_question = question
                with st.spinner("Retrieving → Generating → Evaluating…"):
                    try:
                        result = run_generate_only(
                            selected_model, question,
                            st.session_state.run_path, st.session_state.gt_entry,
                            judge_model=st.session_state.judge_model)
                        st.session_state.model_answer  = result["model_answer"]
                        st.session_state.gemini_answer = ""
                        st.session_state.metrics       = result["metrics"]
                        st.session_state.model_sources = result.get("model_sources",[])
                        st.session_state.metric_mode   = "generate"
                        st.session_state.screen        = 2
                        st.rerun()
                    except ValueError as e: st.error(f"⚠️ {e}")
                    except Exception as e:  st.error(f"Generation failed: {e}")
            else:
                st.error("Upload files, select a model, and enter a question.")

    with col2:
        if st.button("Compare with Gemini", use_container_width=True):
            if st.session_state.run_path and selected_model not in ("No Models Found",) and question.strip():
                st.session_state.last_question = question
                with st.spinner("Running full RAG comparison pipeline…"):
                    try:
                        cname = ensure_indexed_collection(st.session_state.run_path)
                        result = run_rag_evaluation(
                            selected_model, question, cname,
                            st.session_state.run_path, st.session_state.gt_entry,
                            judge_model=st.session_state.judge_model,
                            preindexed=True)
                        st.session_state.model_answer  = result["model_answer"]
                        st.session_state.gemini_answer = result["gemini_answer"]
                        st.session_state.metrics       = result["metrics"]
                        st.session_state.model_sources = result.get("model_sources",[])
                        st.session_state.metric_mode   = "compare"
                        st.session_state.screen        = 3
                        st.rerun()
                    except ValueError as e: st.error(f"⚠️ {e}")
                    except Exception as e:  st.error(f"Compare failed: {e}")
            else:
                st.error("Upload files, select a model, and enter a question.")

    st.divider()

    st.markdown("### 🏁 Section B — Multi-Model Comparison")
    real_models = [m for m in GEN_MODELS if m != "No Models Found"]

    sel_col, btn_col = st.columns([4,1])
    with btn_col:
        if st.button("Select All", use_container_width=True):
            st.session_state.selected_models_list = real_models; st.rerun()
        if st.button("Clear All", use_container_width=True):
            st.session_state.selected_models_list = []; st.rerun()
    with sel_col:
        selected_models = st.multiselect(
            "Select models to compare (Gemini always included as baseline)",
            options=real_models,
            default=[m for m in st.session_state.selected_models_list if m in real_models],
        )
        st.session_state.selected_models_list = selected_models

    if selected_models:
        st.caption(f"Will run: {', '.join(selected_models)} + **Gemini** (baseline) | Judge: **{chosen_judge}**")

    can_run = bool(st.session_state.run_path and question.strip() and selected_models)
    if st.button("🚀 Run Model Comparison", use_container_width=True, disabled=not can_run):
        st.session_state.last_question = question
        with st.spinner(
            f"Running {len(selected_models)} Ollama model(s) + Gemini — "
            "Ollama models run sequentially, please wait…"
        ):
            try:
                cname = ensure_indexed_collection(st.session_state.run_path)
                comp = run_multi_model_comparison(
                    selected_models, question,
                    st.session_state.run_path, st.session_state.gt_entry,
                    judge_model=st.session_state.judge_model,
                    existing_collection=cname)
                st.session_state.comparison_data = comp
                st.session_state.screen          = 4
                st.rerun()
            except ValueError as e: st.error(f"⚠️ {e}")
            except Exception as e:  st.error(f"Comparison failed: {e}")

    if not can_run:
        missing = []
        if not st.session_state.run_path: missing.append("upload documents")
        if not question.strip():          missing.append("enter a question")
        if not selected_models:           missing.append("select at least one model")
        if missing: st.info(f"To enable comparison: {', '.join(missing)}.")


# ==================================================
# SCREEN 2 — Single Model Generate
# ==================================================
elif st.session_state.screen == 2:
    col_back, col_exp = st.columns([1,1])
    with col_back:
        if st.button("← Back to Setup"):
            st.session_state.screen = 1; st.rerun()
    with col_exp:
        render_export_button(
            question=st.session_state.last_question,
            model_answer=st.session_state.model_answer,
            metrics=st.session_state.metrics,
            gt_entry=st.session_state.get("gt_entry"),
            judge_model=st.session_state.judge_model,
            key_suffix="screen2",
        )

    st.markdown(f"### ❓ {st.session_state.last_question}")
    metrics  = st.session_state.metrics
    has_gt   = metrics.get("has_ground_truth", False)
    gt_entry = st.session_state.get("gt_entry")

    if has_gt and gt_entry: render_gt_box(gt_entry)

    answer_card(f"🤖 {st.session_state.selected_model} — Answer",
                st.session_state.model_answer or "No answer generated.")

    if has_gt and gt_entry and st.session_state.model_answer:
        _sim_s2 = calculate_semantic_similarity(
            st.session_state.model_answer, gt_entry.get("answer","")
        )
        render_answer_diff(
            st.session_state.model_answer, gt_entry.get("answer",""),
            st.session_state.selected_model, "Ground Truth",
            similarity=_sim_s2,
        )

    if st.session_state.model_sources:
        with st.expander("📂 Retrieved Source Documents"):
            for i,s in enumerate(st.session_state.model_sources,1): st.write(f"{i}. {s}")

    render_single_metrics(metrics, mode="generate", has_gt=has_gt,
                          judge_model=st.session_state.judge_model)


# ==================================================
# SCREEN 3 — Single Model vs Gemini
# ==================================================
elif st.session_state.screen == 3:
    col_back, col_exp = st.columns([1,1])
    with col_back:
        if st.button("← Back to Setup"):
            st.session_state.screen = 1; st.rerun()
    with col_exp:
        render_export_button(
            question=st.session_state.last_question,
            model_answer=st.session_state.model_answer,
            gemini_answer=st.session_state.gemini_answer,
            metrics=st.session_state.metrics,
            gt_entry=st.session_state.get("gt_entry"),
            judge_model=st.session_state.judge_model,
            key_suffix="screen3",
        )

    st.markdown(f"### ❓ {st.session_state.last_question}")
    metrics  = st.session_state.metrics
    has_gt   = metrics.get("has_ground_truth", False)
    gt_entry = st.session_state.get("gt_entry")

    if has_gt and gt_entry: render_gt_box(gt_entry)

    col1, col2 = st.columns(2)
    with col1:
        answer_card(f"🤖 {st.session_state.selected_model}",
                    st.session_state.model_answer or "No answer generated.")
    with col2:
        answer_card("✨ Gemini", st.session_state.gemini_answer or "No answer generated.")

    if st.session_state.model_answer and st.session_state.gemini_answer:
        _sim_s3_gemini = calculate_semantic_similarity(
            st.session_state.model_answer, st.session_state.gemini_answer
        )
        render_answer_diff(
            st.session_state.model_answer, st.session_state.gemini_answer,
            st.session_state.selected_model, "Gemini",
            similarity=_sim_s3_gemini,
        )
    if has_gt and gt_entry and st.session_state.model_answer:
        _sim_s3_gt = calculate_semantic_similarity(
            st.session_state.model_answer, gt_entry.get("answer","")
        )
        render_answer_diff(
            st.session_state.model_answer, gt_entry.get("answer",""),
            st.session_state.selected_model, "Ground Truth",
            similarity=_sim_s3_gt,
        )

    if st.session_state.model_sources:
        with st.expander("📂 Retrieved Source Documents"):
            for i,s in enumerate(st.session_state.model_sources,1): st.write(f"{i}. {s}")

    render_single_metrics(metrics, mode="compare", has_gt=has_gt,
                          judge_model=st.session_state.judge_model)


# ==================================================
# SCREEN 4 — Multi-Model Comparison
# ==================================================
elif st.session_state.screen == 4:
    if st.button("← Back to Setup"):
        st.session_state.screen = 1; st.session_state.comparison_data = None; st.rerun()

    if st.session_state.comparison_data:
        render_comparison_screen(
            st.session_state.comparison_data,
            st.session_state.last_question,
            judge_model=st.session_state.judge_model,
        )
    else:
        st.error("No comparison data. Please run a comparison from the setup screen.")
        if st.button("Go to Setup"):
            st.session_state.screen = 1; st.rerun()
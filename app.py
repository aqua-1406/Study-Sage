# Streamlit StudyBuddy using Flan-T5 (google/flan-t5-base)
# Save as: app.py
# Requirements:
# pip install streamlit transformers sentencepiece PyMuPDF

import math
import re
import os
from typing import List, Tuple

import streamlit as st
import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ------------------ CONFIG ------------------
MODEL_NAME = "google/flan-t5-base"
CHUNK_MAX_TOKENS = 400  # conservative chunk size for inputs (approx tokens)
GEN_MAX_TOKENS = 256

# ------------------ UTIL ------------------
@st.cache(allow_output_mutation=True, show_spinner=False)

def load_model_and_tokenizer(model_name=MODEL_NAME):
    """Load tokenizer and model once (cached)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    gen = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1,  # CPU. If you have GPU and torch configured, set to 0
        framework="pt",
    )
    return tokenizer, gen


def extract_text_by_page(pdf_bytes: bytes) -> List[str]:
    """Return a list of page texts"""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for p in range(doc.page_count):
        page = doc.load_page(p)
        text = page.get_text("text")
        pages.append(text)
    return pages


def detect_chapter_headings(page_texts: List[str]) -> List[Tuple[int, str]]:
    """
    Scan pages for likely headings and return list of (page_index, heading_line).
    Uses common heading patterns like 'Chapter', 'Unit', 'Part', numeric headings.
    """
    heading_patterns = [
        r'^\s*(chapter|chapitre|CHAPTER|Chapter)\b.*',
        r"^\s*(unit|UNIT)\b.*",
        r"^\s*(part|PART)\b.*",
        r"^\s*\d{1,2}\.\s+[A-Z][\w\s\-,:]{3,}",
        r"^\s*[IVXLC]+\.?\s+[A-Z][\w\s\-,:]{3,}",
    ]
    headings = []
    combined_pattern = re.compile("|".join(f"({p})" for p in heading_patterns), re.MULTILINE)
    for i, txt in enumerate(page_texts):
        preview = "\n".join(txt.splitlines()[:8])
        m = combined_pattern.search(preview)
        if m:
            for line in preview.splitlines():
                if line.strip() and combined_pattern.search(line):
                    headings.append((i, line.strip()))
                    break
    return headings


def build_chapters_from_headings(page_texts: List[str], headings: List[Tuple[int, str]]):
    if not headings:
        return []
    chapters = []
    for idx, (start_page, heading_line) in enumerate(headings):
        start = start_page
        title = heading_line
        if idx + 1 < len(headings):
            end = headings[idx + 1][0] - 1
        else:
            end = len(page_texts) - 1
        text = "\n\n".join(page_texts[start:end+1])
        chapters.append({"title": title, "start": start, "end": end, "text": text})
    return chapters


def fallback_chunking(page_texts: List[str], pages_per_chunk=8):
    chapters = []
    n = len(page_texts)
    chunks = math.ceil(n / pages_per_chunk)
    for i in range(chunks):
        start = i * pages_per_chunk
        end = min((i + 1) * pages_per_chunk - 1, n - 1)
        title = f"Pages {start+1}-{end+1}"
        text = "\n\n".join(page_texts[start:end+1])
        chapters.append({"title": title, "start": start, "end": end, "text": text})
    return chapters


# Simple token-based chunking using the tokenizer
def chunk_text_by_tokens(text: str, tokenizer, max_tokens=CHUNK_MAX_TOKENS):
    tokens = tokenizer.encode(text, return_tensors=None)
    # tokens is list of ids
    if len(tokens) <= max_tokens:
        return [text]
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        # decode back to text for that token slice
        chunk_text = tokenizer.decode(tokens[start:end], skip_special_tokens=True)
        chunks.append(chunk_text)
        start = end
    return chunks


# ------------------ GENERATION HELPERS ------------------

def generate_with_model(gen_pipeline, prompt: str, max_length=GEN_MAX_TOKENS, temperature=0.0):
    try:
        out = gen_pipeline(prompt, max_length=max_length, do_sample=False)
        return out[0]["generated_text"].strip()
    except Exception as e:
        return f"[Generation error: {e}]"


def summarize_chunks(chunks: List[str], gen_pipeline, tokenizer):
    # Summarize each chunk and join summaries
    summaries = []
    for c in chunks:
        prompt = "summarize: " + c
        s = generate_with_model(gen_pipeline, prompt, max_length=GEN_MAX_TOKENS)
        summaries.append(s)
    # combine and compress
    combined = "\n\n".join(summaries)
    final = generate_with_model(gen_pipeline, "summarize: " + combined, max_length=GEN_MAX_TOKENS)
    return final


def generate_flashcards_from_text(text: str, tokenizer, gen_pipeline, per_chunk=5):
    # Chunk text then generate N flashcards per chunk
    chunks = chunk_text_by_tokens(text, tokenizer)
    cards = []
    for c in chunks:
        prompt = (
            "Generate concise flashcards from the following text. Format as lines: Q: <question>\\nA: <answer>. "
            f"Produce up to {per_chunk} flashcards focusing on key concepts, definitions, and formulas.\nText:\n" + c
        )
        out = generate_with_model(gen_pipeline, prompt)
        cards.append(out)
    return "\n\n".join(cards)


def generate_quiz_from_text(text: str, tokenizer, gen_pipeline, per_chunk=5):
    # Warning: Flan-T5 can produce inconsistent MCQ formatting — we attempt best-effort
    chunks = chunk_text_by_tokens(text, tokenizer)
    quizzes = []
    for c in chunks:
        prompt = (
            "Generate multiple-choice questions (MCQs) from the following text.\n"
            "Format each question as:\nQ: <question>\nA) <optA>\nB) <optB>\nC) <optC>\nD) <optD>\nAnswer: <A/B/C/D>\n"
            f"Make {per_chunk} questions, with plausible distractors. Text:\n" + c
        )
        out = generate_with_model(gen_pipeline, prompt)
        quizzes.append(out)
    return "\n\n".join(quizzes)


# ------------------ STREAMLIT UI ------------------

st.set_page_config(page_title="StudyBuddy (Flan-T5)", layout="wide")
st.title("📚 StudyBuddy — Flan-T5 (google/flan-t5-base) — Offline")
st.markdown(
    "Upload a PDF textbook. Chapters are auto-detected. Generate Flashcards, Quizzes, or Notes using Flan-T5 locally (plain text on screen)."
)

# load model
with st.spinner("Loading model (this may take ~30s on first run)..."):
    tokenizer, gen_pipeline = load_model_and_tokenizer()

uploaded_file = st.file_uploader("Upload textbook PDF", type=["pdf"], accept_multiple_files=False)

if uploaded_file:
    with st.spinner("Extracting PDF..."):
        pdf_bytes = uploaded_file.read()
        page_texts = extract_text_by_page(pdf_bytes)
        st.success(f"Extracted {len(page_texts)} pages.")

    headings = detect_chapter_headings(page_texts)
    if headings:
        st.sidebar.markdown("### Detected chapter headings")
        for p, h in headings[:20]:
            st.sidebar.write(f"Page {p+1}: {h}")
        chapters = build_chapters_from_headings(page_texts, headings)
        if not chapters:
            chapters = fallback_chunking(page_texts)
            st.sidebar.warning("Headings detected but failed to build chapters; using fallback chunking.")
    else:
        st.sidebar.info("No clear chapter headings found; using page-chunk fallback.")
        chapters = fallback_chunking(page_texts, pages_per_chunk=8)

    titles = [f"{c['title']} (p{c['start']+1}-{c['end']+1})" for c in chapters]
    idx = st.sidebar.selectbox("Select chapter / chunk", list(range(len(chapters))), format_func=lambda i: titles[i])

    selected = chapters[idx]
    st.subheader(f"Selected: {selected['title']}")
    st.caption(f"Pages {selected['start']+1} — {selected['end']+1}")

    with st.expander("Preview chapter text (first 3000 chars)"):
        st.text_area("Chapter preview", selected['text'][:3000], height=300)

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Generate Flashcards (on-screen)"):
            with st.spinner("Generating flashcards (may be slow)..."):
                out = generate_flashcards_from_text(selected['text'], tokenizer, gen_pipeline, per_chunk=5)
                st.markdown("### Flashcards")
                st.code(out, language="text")

    with col2:
        if st.button("Generate Quiz (on-screen)"):
            with st.spinner("Generating quiz (may be slow and format may vary)..."):
                out = generate_quiz_from_text(selected['text'], tokenizer, gen_pipeline, per_chunk=5)
                st.markdown("### Quiz (MCQs)")
                st.code(out, language="text")

    with col3:
        if st.button("Generate Notes (on-screen)"):
            with st.spinner("Generating notes..."):
                # chunk & summarize
                chunks = chunk_text_by_tokens(selected['text'], tokenizer)
                notes = summarize_chunks(chunks, gen_pipeline, tokenizer)
                st.markdown("### Notes")
                st.code(notes, language="text")

    st.markdown("---")
    st.markdown(
        "**Notes:** Flan-T5-base works well for summaries/notes. Flashcards and MCQs are best-effort; their format may vary. "
        "If you want more consistent/stronger structured outputs later, consider upgrading to an instruction-tuned LLM like Mistral or using a small OpenAI key."
    )
else:
    st.info("Upload a PDF to begin. Use textbooks with consistent headings for best auto-detection.")


# End of file

# app/tabs/tenk_explorer.py

import streamlit as st
import pathlib
import datetime as dt
import re

import pandas as pd
from tqdm.auto import tqdm
import fitz
import bs4

from sec_edgar_downloader import Downloader
from sentence_transformers import util
from config import OPENAI_API_KEY

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# ── Configuration ─────────────────────────────────────────
EMAIL     = "electros@bu.edu"
QA_MODEL  = "distilbert-base-cased-distilled-squad"
EMB_MODEL = "all-MiniLM-L6-v2"
CHUNK_W   = 220
TOP_K     = 3

_dollar = r"\$?\s?[0-9]{1,3}(?:,[0-9]{3})+(?:\.\d+)?\s?(?:million|billion)?"

# ── Helper functions ──────────────────────────────────────
def _find(p: str, txt: str) -> str:
    m = re.search(p, txt, flags=re.I)
    return m.group(1) if m else "N/A"

def quick_metrics(txt: str) -> dict:
    """Extracts top‐level dollar figures via regex from the filing text."""
    return {
        "Total revenue":        _find(r"total\s+revenue[^$]{0,50}(" + _dollar + ")", txt),
        "Net income":           _find(r"net\s+income[^$]{0,50}("   + _dollar + ")", txt),
        "Total assets":         _find(r"total\s+assets[^$]{0,50}(" + _dollar + ")", txt),
        "Total liabilities":    _find(r"total\s+liabilit[^$]{0,50}(" + _dollar + ")", txt),
        "Shareholders' equity": _find(r"shareholders.? equity[^$]{0,50}(" + _dollar + ")", txt),
        "Cash & equivalents":   _find(r"cash\s+and\s+cash\s+equivalents[^$]{0,50}(" + _dollar + ")", txt),
    }

def extract_text(path: str) -> str:
    """Loads raw text from PDF, HTML, or TXT SEC filings."""
    if path.lower().endswith(".pdf"):
        doc = fitz.open(path)
        return "\n".join(p.get_text() for p in tqdm(doc, desc="Reading PDF"))
    if path.lower().endswith((".htm", ".html")):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            soup = bs4.BeautifulSoup(f.read(), "lxml")
        return soup.get_text(" ", strip=True)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

@st.cache_resource(show_spinner=False)
def load_models():
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline
    embedder = SentenceTransformer(EMB_MODEL)
    qa_pipe  = pipeline("question-answering", model=QA_MODEL, device="cpu")
    return embedder, qa_pipe

@st.cache_data(show_spinner=False)
def load_and_embed(path: str):
    """Chunks the text and pre-computes embeddings for fast semantic search."""
    text = extract_text(path)
    words = text.split()
    chunks = [
        " ".join(words[i : i + CHUNK_W])
        for i in range(0, len(words), CHUNK_W)
    ]
    embedder, _ = load_models()
    chunk_emb = embedder.encode(chunks, convert_to_tensor=True)
    return text, chunks, chunk_emb

def smart_answer(question: str, chunks, chunk_emb, top_k: int = TOP_K) -> str:
    """Runs the local BERT QA model on the top-K most relevant chunks."""
    embedder, qa_pipe = load_models()
    q_emb = embedder.encode(question, convert_to_tensor=True)
    hits = util.cos_sim(q_emb, chunk_emb)[0].topk(top_k)
    context = " ".join(chunks[i] for i in hits.indices)
    return qa_pipe(question=question, context=context)["answer"]

# ── UI: Main render ───────────────────────────────────────
def render():
    st.header("🔍 10-K Explorer & QA")

    # -------------------------------------------------------------------------
    # WARNING: Independent Mode
    # This 10-K Explorer tab is fully self-contained and does NOT share filters
    # or selections with the other tabs. Any tickers or years chosen here
    # will reset the main-page filters when you switch back.
    # -------------------------------------------------------------------------
    st.warning(
        "⚠️ This tab is independent of the others. "
        "To select tickers or years here clear your main page filters. " 
        "Any choices made here will reset the main page filters when you switch back.",\
        icon="⚠️"
    )

    # 1) Tickers selection
    st.subheader("Step 1: Choose Tickers")
    st.markdown(
        "Use this multiselect to pick one or more stock symbols. "
        "We will download the latest 10-K filings for each selected company."
    )
    from utils import data as d
    master_df   = d.master_tickers()
    display_lst = master_df.display.tolist()

    chosen_disp = st.multiselect(
        "Tickers (search)",
        display_lst,
        key="10k_tickers"
    )
    chosen = master_df.loc[
        master_df.display.isin(chosen_disp),
        "symbol"
    ].tolist()
    if not chosen:
        st.info("Select at least one ticker above to begin.")
        return

    # 2) Download filings
    st.subheader("Step 2: Download 10-K Filings")
    st.markdown(
        "We’ll fetch all available Form 10-K filings from the SEC EDGAR database "
        "for your selected tickers. This may take a few seconds."
    )
    dl = Downloader(company_name="Quick10K", email_address=EMAIL)
    for tkr in chosen:
        dl.get("10-K", tkr)

    # 3) Year filtering
    st.subheader("Step 3: Filter by Year")
    st.markdown(
        "From the downloaded filings, pick one or more fiscal years to restrict the list. "
        "Only the most recent filing per company per year will be shown."
    )
    recs = []
    current_yy = dt.date.today().year % 100
    for tkr in chosen:
        base = pathlib.Path("sec-edgar-filings") / tkr / "10-K"
        if not base.exists():
            continue
        for acc in base.iterdir():
            if not acc.is_dir():
                continue
            m = re.match(r"^\d{10}-(\d{2})-\d{6}$", acc.name)
            if not m:
                continue
            yy   = int(m.group(1))
            year = 2000 + yy if yy <= current_yy else 1900 + yy
            date = dt.date.fromtimestamp(acc.stat().st_mtime)

            files = list(acc.glob("*.htm")) + list(acc.glob("*.html")) \
                 or list(acc.glob("full-submission.txt"))
            if not files:
                continue

            recs.append({"ticker": tkr, "year": year, "date": date, "path": str(files[0])})

    df_all = pd.DataFrame(recs)
    if df_all.empty:
        st.warning("No 10-K filings found for the selected tickers.")
        return

    years = sorted(df_all["year"].unique())
    selected_years = st.multiselect(
        "Filter by filing year",
        years,
        default=[years[-1]],
        key="10k_years"
    )
    if not selected_years:
        st.info("Pick at least one year to continue.")
        return

    df_filt = df_all[df_all["year"].isin(selected_years)]
    df_unique = (
        df_filt
        .sort_values("date")
        .drop_duplicates(subset=["ticker", "year"], keep="last")
        .reset_index(drop=True)
    )

    # 4) Show filtered list
    st.subheader("Step 4: Select Filing")
    st.markdown(
        "Here’s the list of filings matching your tickers and years. "
        "Choose one to explore its contents."
    )
    st.dataframe(df_unique[["ticker","year","date"]], use_container_width=True)
    sel = st.selectbox(
        "Pick a filing to view",
        df_unique.index,
        format_func=lambda i: f"{df_unique.at[i,'ticker']} – {df_unique.at[i,'year']}"
    )
    path = df_unique.at[sel, "path"]

    # 5) Display key metrics
    st.subheader("Step 5: Key Financial Metrics")
    st.markdown(
        "We extract common financial line items (revenue, net income, assets, etc.) "
        "via regex from the filing text."
    )
    text, chunks, chunk_emb = load_and_embed(path)
    st.json(quick_metrics(text))

    # 6) Ask questions
    st.subheader("Step 6: Ask a Question")
    st.markdown(
        "Type any question about this 10-K. "
        "Use the buttons to choose between the local BERT-QA model or OpenAI GPT-3.5-Turbo."
    )
    question = st.text_input("Your question here", key="10k_question")
    if not question:
        return

    # compute context once
    embedder, _ = load_models()
    q_emb   = embedder.encode(question, convert_to_tensor=True)
    hits    = util.cos_sim(q_emb, chunk_emb)[0].topk(TOP_K)
    context = " ".join(chunks[i] for i in hits.indices)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 Local BERT-QA", key="local_qa"):
            st.info("Running local QA…")
            answer_local = smart_answer(question, chunks, chunk_emb)
            st.success(answer_local)

    with col2:
        if st.button("🤖 OpenAI GPT-3.5-Turbo", key="openai_qa"):
            import openai
            openai.api_key = OPENAI_API_KEY
            st.info("Calling OpenAI API…")
            try:
                resp = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role":"system","content":"You are a helpful financial assistant."},
                        {"role":"user","content":
                            f"Use ONLY the context below to answer the question.\n\n"
                            f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nAnswer concisely."
                        }
                    ],
                    max_tokens=300,
                    temperature=0.0,
                )
                answer_openai = resp.choices[0].message.content.strip()
                st.success(answer_openai)
            except Exception as e:
                st.error(f"OpenAI API error: {e}")

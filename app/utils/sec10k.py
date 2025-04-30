# -------------------------------------------------------------
# utils/sec10k.py  ·  fully logged 10-K utilities (fixed path)
# -------------------------------------------------------------
import re, pathlib, fitz, bs4, logging, tqdm
from sec_edgar_downloader import Downloader
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from config import EMAIL_ADDRESS
import streamlit as st

log = logging.getLogger(__name__)

# ── configuration (adjust email only) -------------------------
EMAIL     = st.secrets["EMAIL_ADDRESS"]
EMB_MODEL = "all-MiniLM-L6-v2"
QA_MODEL  = "distilbert-base-cased-distilled-squad"
CHUNK_W   = 220
TOP_K     = 3

# ── download latest 10-K --------------------------------------
def latest_10k(tkr: str) -> str | None:
    log.info("[10-K] request for %s", tkr)
    # Downloader writes into "sec-edgar-downloader/<ticker>/<form>"
    dl = Downloader(company_name="Streamlit10K", email_address=EMAIL)
    dl.get("10-K", tkr)  # form, ticker

    base = pathlib.Path("sec-edgar-downloader") / tkr / "10-K"
    log.debug("[10-K] looking in %s", base)
    if not base.exists():
        log.error("[10-K] directory missing: %s", base)
        return None

    # pick the most‐recent accession folder
    acc = max(base.iterdir(), key=lambda p: p.stat().st_mtime)
    log.debug("[10-K] newest accession: %s", acc.name)

    # prefer HTML if present
    htmls = list(acc.glob("*.htm")) + list(acc.glob("*.html"))
    if htmls:
        return str(htmls[0])
    # fallback to full-submission.txt
    txts = list(acc.glob("*.txt"))
    return str(txts[0]) if txts else None

# ── extract plain text ----------------------------------------
def extract_text(path: str) -> str:
    log.info("[extract] %s", path)
    if path.lower().endswith(".pdf"):
        doc = fitz.open(path)
        return "\n".join(p.get_text() for p in tqdm.tqdm(doc, desc="PDF pages"))
    if path.lower().endswith((".htm", ".html")):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            soup = bs4.BeautifulSoup(f.read(), "lxml")
        return soup.get_text(" ", strip=True)
    return open(path, "r", encoding="utf-8", errors="ignore").read()

# ── quick headline metrics ------------------------------------
_DOLLAR = r"\$?\s?[0-9]{1,3}(?:,[0-9]{3})+(?:\.\d+)?\s?(?:million|billion)?"
def _find(label, txt):
    m = re.search(label + rf"[^$]{{0,60}}({_DOLLAR})", txt, flags=re.I)
    return m.group(1) if m else "N/A"

def headline_metrics(txt: str) -> dict:
    return {
        "Total revenue":        _find(r"total\s+revenue", txt),
        "Net income":           _find(r"net\s+income", txt),
        "Total assets":         _find(r"total\s+assets", txt),
        "Total liabilities":    _find(r"total\s+liabilit", txt),
        "Equity":               _find(r"shareholders.? equity", txt),
        "Cash & equivalents":   _find(r"cash\s+and\s+cash\s+equivalents", txt),
    }

# ── mini-retriever + QA ---------------------------------------
class TenKQnA:
    def __init__(self, raw_text: str):
        log.info("[QnA] chunking text (≈%d chars)…", len(raw_text))
        words = raw_text.split()
        self.chunks = [" ".join(words[i:i+CHUNK_W]) for i in range(0, len(words), CHUNK_W)]
        log.info("[QnA] total chunks: %d", len(self.chunks))

        log.info("[QnA] loading SentenceTransformer %s …", EMB_MODEL)
        self.embedder  = SentenceTransformer(EMB_MODEL)
        self.chunk_emb = self.embedder.encode(self.chunks, convert_to_tensor=True)
        log.info("[QnA] embeddings ready")

        log.info("[QnA] loading QA model %s …", QA_MODEL)
        self.qa = pipeline("question-answering", model=QA_MODEL, device="cpu")
        log.info("[QnA] QA pipeline ready")

    def ask(self, question: str, k: int = TOP_K) -> str:
        log.info("[QnA] question: %s", question)
        q_emb = self.embedder.encode(question, convert_to_tensor=True)
        hits  = util.cos_sim(q_emb, self.chunk_emb)[0].topk(k)
        ctx   = " ".join(self.chunks[i] for i in hits.indices)
        answer = self.qa(question=question, context=ctx)["answer"]
        log.info("[QnA] answer: %s", answer)
        return answer

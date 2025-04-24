# -------- utils/filings.py ---------
import os, glob, fitz
from bs4 import BeautifulSoup
from transformers import pipeline
from sec_edgar_downloader import Downloader
from config import EMAIL_ADDRESS

def download(ticker:str, form:str)->str|None:
    dl=Downloader(email_address=EMAIL_ADDRESS); dl.get(form,ticker,amount=1)
    folder=os.path.join("sec-edgar-downloader",ticker,form)
    files=sorted(glob.glob(os.path.join(folder,"*")), key=os.path.getmtime, reverse=True)
    return files[0] if files else None

def pdf_text(path:str)->str:
    txt=""; doc=fitz.open(path)
    for p in doc: txt+=p.get_text()
    return txt

def qa(question:str, context:str, max_chunk:int=500)->str:
    nlp=pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    words=context.split()
    if len(words)<=max_chunk:
        return nlp(question=question,context=context)["answer"]
    best,score="", -1
    for i in range(0,len(words),max_chunk//2):
        chunk=" ".join(words[i:i+max_chunk])
        out=nlp(question=question,context=chunk)
        if out["score"]>score: best,score=out["answer"],out["score"]
    return best

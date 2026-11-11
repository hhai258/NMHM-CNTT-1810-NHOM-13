"""Hàm dùng chung: nạp dữ liệu, tiền xử lý văn bản, dựng pipeline.

Đề tài 3 - Phân loại tin nhắn rác (SMS Spam Collection).
Nhãn dương (positive class) = spam (1), ham = 0.
"""
import csv
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "SMSSpamCollection"
RANDOM_STATE = 42


def load_raw(path=DATA_PATH):
    """Đọc file thô (tab-separated, mã hoá UTF-8)."""
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"],
                     encoding="utf-8",
                     quoting=csv.QUOTE_NONE)   # tránh ký tự " làm mất dòng
    df["y"] = (df["label"] == "spam").astype(int)
    return df


URL_RE = re.compile(r"(https?://\S+|www\.\S+|\b\S+\.(com|co\.uk|net|org)\b\S*)", re.I)
LONGNUM_RE = re.compile(r"\b\d{5,}\b")        # số điện thoại, mã dịch vụ 5+ chữ số
NUM_RE = re.compile(r"\b\d+\b")
CURR_RE = re.compile(r"[£$€]")


def normalize_text(s: str) -> str:
    """Chuẩn hoá có chủ đích: giữ lại tín hiệu (có URL, số dài, tiền tệ)
    bằng token đặc biệt thay vì xoá mất thông tin."""
    s = s.lower()
    s = URL_RE.sub(" urltoken ", s)
    s = CURR_RE.sub(" currtoken ", s)
    s = LONGNUM_RE.sub(" longnumtoken ", s)
    s = NUM_RE.sub(" numtoken ", s)
    s = re.sub(r"[^a-z\s]", " ", s)           # bỏ dấu câu/ký tự đặc biệt
    return re.sub(r"\s+", " ", s).strip()


# Ba mức tiền xử lý để so sánh
PREPROCESS = {
    "P0_raw":        dict(preprocessor=None, stop_words=None),
    "P1_normalize":  dict(preprocessor=normalize_text, stop_words=None),
    "P2_norm_stop":  dict(preprocessor=normalize_text, stop_words="english"),
}

# Bốn cách biểu diễn đặc trưng
REPRESENT = {
    "BoW_uni":    dict(kind="bow",   ngram_range=(1, 1), min_df=1),
    "TFIDF_uni":  dict(kind="tfidf", ngram_range=(1, 1), min_df=1),
    "BoW_bi":     dict(kind="bow",   ngram_range=(1, 2), min_df=2),
    "TFIDF_bi":   dict(kind="tfidf", ngram_range=(1, 2), min_df=2),
}


def make_vectorizer(prep_name, rep_name):
    p, r = PREPROCESS[prep_name], REPRESENT[rep_name]
    cls = CountVectorizer if r["kind"] == "bow" else TfidfVectorizer
    kw = dict(lowercase=True, ngram_range=r["ngram_range"], min_df=r["min_df"],
              stop_words=p["stop_words"])
    if p["preprocessor"] is not None:
        kw["preprocessor"] = p["preprocessor"]
    return cls(**kw)


def make_model(name):
    if name == "MultinomialNB":
        return MultinomialNB()
    if name == "KNN":
        return KNeighborsClassifier()            # k=5, euclidean (mặc định)
    if name == "DecisionTree":
        return DecisionTreeClassifier(random_state=RANDOM_STATE)
    raise ValueError(name)


def make_pipeline(model_name, prep_name, rep_name, model=None, memory=None):
    """Vectorizer nằm TRONG pipeline => chỉ fit trên tập huấn luyện
    (và trên từng fold khi cross-validation) -> tránh data leakage."""
    # memory: cache kết quả fit của vectorizer theo từng fold (chỉ để tăng tốc grid search;
    # mỗi fold vẫn fit riêng trên phần train của fold đó nên không gây rò rỉ dữ liệu)
    return Pipeline([("vec", make_vectorizer(prep_name, rep_name)),
                     ("clf", model if model is not None else make_model(model_name))],
                    memory=memory)

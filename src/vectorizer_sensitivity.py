"""Ảnh hưởng của tham số vector hoá (đúng yêu cầu mục 3.1 của hướng dẫn):
unigram/bigram, min_df, max_df, số lượng đặc trưng (max_features).

Mỗi mô hình giữ nguyên tiền xử lý, loại biểu diễn (BoW/TF-IDF) và siêu tham số đã chọn
trong run_experiments.py; chỉ đổi từng tham số vector hoá. Chỉ dùng 5-fold CV trên TẬP TRAIN
(không dùng tập test), tiêu chí F1 của lớp spam.
Chạy:  python src/vectorizer_sensitivity.py
"""
import itertools
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from common import RANDOM_STATE, ROOT, load_raw, make_pipeline

warnings.filterwarnings("ignore")
CFG = {   # (tiền xử lý, biểu diễn, mô hình) - cấu hình đã chọn bằng CV
    "MultinomialNB": ("P2_norm_stop", "BoW_uni", MultinomialNB(alpha=2.0, fit_prior=True)),
    "KNN": ("P1_normalize", "TFIDF_uni", KNeighborsClassifier(n_neighbors=9, metric="cosine", weights="distance")),
    "DecisionTree": ("P2_norm_stop", "BoW_uni", DecisionTreeClassifier(criterion="gini", max_depth=10, random_state=RANDOM_STATE)),
}
df = load_raw().drop_duplicates("text")
X_tr, _, y_tr, _ = train_test_split(df.text, df.y, test_size=0.2, stratify=df.y, random_state=RANDOM_STATE)
CV = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

rows = []
for m, (p, r, clf) in CFG.items():
    # (a) lưới ngram x min_df x max_df
    for ng, mn, mx in itertools.product([(1, 1), (1, 2)], [1, 2, 3, 5], [1.0, 0.5]):
        pipe = make_pipeline(m, p, r, model=clf)
        pipe.named_steps["vec"].set_params(ngram_range=ng, min_df=mn, max_df=mx)
        cv = cross_validate(pipe, X_tr, y_tr, cv=CV, scoring="f1")
        rows.append(dict(model=m, study="grid", ngram=f"{ng[0]}-{ng[1]}", min_df=mn, max_df=mx, max_features=None,
                         cv_f1=cv["test_score"].mean(), cv_f1_std=cv["test_score"].std()))
    # (b) giới hạn số đặc trưng (unigram, min_df=1, max_df=1.0)
    for mf in [200, 500, 1000, 2000, 5000, None]:
        pipe = make_pipeline(m, p, r, model=clf)
        pipe.named_steps["vec"].set_params(ngram_range=(1, 1), min_df=1, max_df=1.0, max_features=mf)
        cv = cross_validate(pipe, X_tr, y_tr, cv=CV, scoring="f1")
        pipe.fit(X_tr, y_tr)
        rows.append(dict(model=m, study="max_features", ngram="1-1", min_df=1, max_df=1.0, max_features=mf if mf else 99999,
                         cv_f1=cv["test_score"].mean(), cv_f1_std=cv["test_score"].std(),
                         n_features=len(pipe.named_steps["vec"].get_feature_names_out())))
    print(m, "done", flush=True)

out = pd.DataFrame(rows)
out.to_csv(ROOT / "results" / "vectorizer_sensitivity.csv", index=False)

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(14, 6.4))
g = out[out.study == "grid"].copy()
g["cột"] = "ngram " + g.ngram + ", max_df=" + g.max_df.astype(str)
for j, m in enumerate(CFG):
    piv = g[g.model == m].pivot(index="min_df", columns="cột", values="cv_f1")
    sns.heatmap(piv, annot=True, fmt=".3f", cmap="YlGnBu", cbar=False, ax=axes[0, j])
    axes[0, j].set_title(f"{m}: F1 CV theo ngram / min_df / max_df"); axes[0, j].set_xlabel("")
    f = out[(out.study == "max_features") & (out.model == m)].sort_values("max_features")
    axes[1, j].errorbar(range(len(f)), f.cv_f1, yerr=f.cv_f1_std, fmt="o-", capsize=3)
    axes[1, j].set_xticks(range(len(f))); axes[1, j].set_xticklabels([("tất cả" if v == 99999 else int(v)) for v in f.max_features])
    axes[1, j].set_xlabel("max_features"); axes[1, j].set_ylabel("F1 CV"); axes[1, j].set_title(f"{m}: số đặc trưng")
plt.tight_layout(); plt.savefig(ROOT / "figures" / "vectorizer_sensitivity.png", dpi=170)
print("DONE")

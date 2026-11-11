"""Chạy toàn bộ thí nghiệm cho Đề tài 3: phân loại tin nhắn rác.

Các giai đoạn:
  0. EDA (số liệu + hình)
  1. Chia dữ liệu train/test (stratify, random_state cố định), bỏ trùng trước khi chia
  2. Giai đoạn 1: so sánh tiền xử lý x biểu diễn x mô hình (siêu tham số mặc định = baseline)
  3. Giai đoạn 2: GridSearchCV từng mô hình (chỉ trên tập train)
  4. Giai đoạn 3: đánh giá trên tập test, ROC, ổn định theo nhiều seed, phân tích lỗi
Chạy:  python src/run_experiments.py
"""
import json
import time
import warnings
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)

from common import (PREPROCESS, REPRESENT, RANDOM_STATE, ROOT, load_raw,
                    make_model, make_pipeline, normalize_text)

warnings.filterwarnings("ignore")
FIG, RES = ROOT / "figures", ROOT / "results"
MODELS = ["MultinomialNB", "KNN", "DecisionTree"]
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
sns.set_theme(style="whitegrid", context="notebook")
summary = {}
import tempfile
from joblib import Memory
MEM = Memory(tempfile.mkdtemp(prefix="vec_cache_"), verbose=0)


def metrics(y, pred, score=None):
    d = dict(accuracy=accuracy_score(y, pred),
             precision=precision_score(y, pred, zero_division=0),
             recall=recall_score(y, pred),
             f1=f1_score(y, pred))
    if score is not None:
        d["roc_auc"] = roc_auc_score(y, score)
    return d


def get_score(pipe, X):
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    return pipe.decision_function(X)


# ------------------------------------------------------------------ 0. EDA
raw = load_raw()
summary["n_raw"] = len(raw)
summary["n_dup"] = int(raw.duplicated("text").sum())
summary["n_missing"] = int(raw.isna().sum().sum())
summary["raw_counts"] = raw["label"].value_counts().to_dict()
df = raw.drop_duplicates("text").reset_index(drop=True)
summary["n_unique"] = len(df)
summary["unique_counts"] = df["label"].value_counts().to_dict()
summary["conflicting_labels"] = int((raw.groupby("text").y.nunique() > 1).sum())

df["n_chars"] = df["text"].str.len()
df["n_words"] = df["text"].str.split().str.len()
df["has_digit_seq"] = df["text"].str.contains(r"\d{5,}")
df["has_url"] = df["text"].str.contains(r"www\.|http|\.com|\.co\.uk", case=False)
df["has_currency"] = df["text"].str.contains(r"[£$€]")
df["upper_ratio"] = df["text"].apply(lambda s: sum(c.isupper() for c in s) / max(1, sum(c.isalpha() for c in s)))
ls = df.groupby("label")[["n_chars", "n_words"]].agg(["mean", "median", "max"]).round(1)
summary["length_stats"] = {f"{lab}|{c[0]}|{c[1]}": float(v) for lab, row in ls.iterrows() for c, v in row.items()}
summary["signal_rates"] = {f"{lab}|{c}": float(v) for lab, row in df.groupby("label")[["has_digit_seq", "has_url", "has_currency", "upper_ratio"]].mean().round(3).iterrows() for c, v in row.items()}

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
cnt = raw["label"].value_counts()
sns.barplot(x=cnt.index, y=cnt.values, ax=ax[0], palette=["#4C72B0", "#C44E52"])
for i, v in enumerate(cnt.values):
    ax[0].text(i, v + 40, f"{v} ({v/cnt.sum():.1%})", ha="center")
ax[0].set_title("Phân phối nhãn (dữ liệu thô, 5574 tin)"); ax[0].set_ylabel("Số tin nhắn"); ax[0].set_ylim(0, cnt.max() * 1.15)
sns.histplot(data=df, x="n_chars", hue="label", bins=40, stat="density", common_norm=False,
             element="step", ax=ax[1], palette={"ham": "#4C72B0", "spam": "#C44E52"})
ax[1].set_title("Phân phối độ dài tin nhắn (ký tự)"); ax[1].set_xlabel("Số ký tự")
plt.tight_layout(); plt.savefig(FIG / "eda_label_length.png", dpi=170); plt.close()

# từ phổ biến theo lớp (sau normalize + stopword) để hình thành giả thuyết
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, (lab, col) in zip(ax, [("ham", "#4C72B0"), ("spam", "#C44E52")]):
    cv = CountVectorizer(preprocessor=normalize_text, stop_words="english")
    M = cv.fit_transform(df.loc[df.label == lab, "text"])
    freq = np.asarray(M.sum(0)).ravel()
    top = np.argsort(freq)[::-1][:12]
    a.barh([cv.get_feature_names_out()[i] for i in top][::-1], freq[top][::-1], color=col)
    a.set_title(f"Top từ - {lab}")
plt.tight_layout(); plt.savefig(FIG / "eda_top_words.png", dpi=170); plt.close()

# --------------------------------------------------------- 1. chia dữ liệu
X_tr, X_te, y_tr, y_te = train_test_split(df["text"], df["y"], test_size=0.2,
                                          stratify=df["y"], random_state=RANDOM_STATE)
summary["split"] = dict(train=len(X_tr), test=len(X_te), train_spam=int(y_tr.sum()),
                        test_spam=int(y_te.sum()), random_state=RANDOM_STATE, cv_folds=5)
print("split", summary["split"])

# ------------------------------------------- 2. so sánh tiền xử lý x biểu diễn
rows = []
t0 = time.time()
for m in MODELS:
    for p in PREPROCESS:
        for r in REPRESENT:
            pipe = make_pipeline(m, p, r)
            cv = cross_validate(pipe, X_tr, y_tr, cv=CV, scoring=["f1", "precision", "recall", "accuracy"], n_jobs=-1)
            pipe.fit(X_tr, y_tr)
            pred = pipe.predict(X_te)
            te = metrics(y_te, pred, get_score(pipe, X_te))
            rows.append(dict(model=m, prep=p, rep=r,
                             cv_f1=cv["test_f1"].mean(), cv_f1_std=cv["test_f1"].std(),
                             cv_precision=cv["test_precision"].mean(), cv_recall=cv["test_recall"].mean(),
                             cv_acc=cv["test_accuracy"].mean(),
                             **{f"test_{k}": v for k, v in te.items()}))
    print(m, "done", round(time.time() - t0, 1), "s")
s1 = pd.DataFrame(rows)
s1.to_csv(RES / "stage1_prep_repr_comparison.csv", index=False)

# baseline = BoW unigram, không tiền xử lý, tham số mặc định
base = s1[(s1.prep == "P0_raw") & (s1.rep == "BoW_uni")].set_index("model")
base.to_csv(RES / "baseline_results.csv")

fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
for ax, m in zip(axes, MODELS):
    piv = s1[s1.model == m].pivot(index="prep", columns="rep", values="cv_f1")[list(REPRESENT)].loc[list(PREPROCESS)]
    sns.heatmap(piv, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax, cbar=False, vmin=0.4, vmax=1)
    ax.set_title(f"{m} - F1 spam (5-fold CV)"); ax.set_xlabel(""); ax.set_ylabel("")
plt.tight_layout(); plt.savefig(FIG / "stage1_heatmap_cv_f1.png", dpi=170); plt.close()

# ------------------------------------------------ 3. GridSearchCV từng mô hình
# Lưu ý thiết kế: với Multinomial NB và K-NN, tham số mặc định (alpha=1, khoảng cách
# Euclid) tương tác mạnh với cách biểu diễn (TF-IDF có giá trị rất nhỏ nên alpha=1 làm
# "san phẳng" xác suất; Euclid trên vector đếm bị chi phối bởi độ dài tin nhắn).
# Vì vậy mỗi ô (tiền xử lý x biểu diễn) của hai mô hình này được tinh chỉnh tham số
# TRƯỚC khi so sánh, để không đánh giá thấp một biểu diễn chỉ vì tham số chưa hợp.
# Cây quyết định (bất biến theo co giãn đặc trưng) chọn ô theo giai đoạn 1 rồi mới tinh chỉnh.
GRIDS = {
    "MultinomialNB": {"clf__alpha": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
                      "clf__fit_prior": [True, False]},
    "KNN": {"clf__n_neighbors": [1, 3, 5, 7, 9, 15], "clf__metric": ["cosine", "euclidean"],
            "clf__weights": ["uniform", "distance"]},
    "DecisionTree": {"clf__criterion": ["gini", "entropy"], "clf__max_depth": [None, 5, 10, 20, 40],
                     "clf__min_samples_leaf": [1, 2, 5]},
}
def clean_params(d):
    return {k.replace("clf__", ""): (None if (v is None or (isinstance(v, float) and np.isnan(v))) else (v.item() if hasattr(v, "item") else v))
            for k, v in d.items()}

best_cfg, tuned, grid_tables, cell_rows = {}, {}, {}, []
for m in ["MultinomialNB", "KNN"]:
    best = None
    for p in PREPROCESS:
        for r in REPRESENT:
            gs = GridSearchCV(make_pipeline(m, p, r, memory=MEM), GRIDS[m], scoring="f1", cv=CV, n_jobs=-1, refit=True)
            gs.fit(X_tr, y_tr)
            cell_rows.append(dict(model=m, prep=p, rep=r, tuned_cv_f1=gs.best_score_,
                                  **{f"best_{k}": v for k, v in clean_params(gs.best_params_).items()}))
            if best is None or gs.best_score_ > best[0]:
                best = (gs.best_score_, p, r, gs)
    sc, p, r, gs = best
    d1 = s1[(s1.model == m) & (s1.prep == p) & (s1.rep == r)].iloc[0]
    tuned[m] = gs.best_estimator_
    best_cfg[m] = dict(prep=p, rep=r, stage1_cv_f1=float(d1.cv_f1), best_params=clean_params(gs.best_params_), grid_cv_f1=float(sc))
    grid_tables[m] = pd.DataFrame(gs.cv_results_); grid_tables[m].to_csv(RES / f"grid_{m}.csv", index=False)
    print(m, best_cfg[m])
m = "DecisionTree"
sub = s1[s1.model == m].sort_values("cv_f1", ascending=False).iloc[0]
gs = GridSearchCV(make_pipeline(m, sub.prep, sub.rep), GRIDS[m], scoring="f1", cv=CV, n_jobs=-1, refit=True).fit(X_tr, y_tr)
tuned[m] = gs.best_estimator_
best_cfg[m] = dict(prep=sub.prep, rep=sub.rep, stage1_cv_f1=float(sub.cv_f1), best_params=clean_params(gs.best_params_), grid_cv_f1=float(gs.best_score_))
grid_tables[m] = pd.DataFrame(gs.cv_results_); grid_tables[m].to_csv(RES / f"grid_{m}.csv", index=False)
print(m, best_cfg[m])
summary["best_cfg"] = best_cfg
cells = pd.DataFrame(cell_rows); cells.to_csv(RES / "stage2_tuned_per_cell.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
for ax, m in zip(axes, ["MultinomialNB", "KNN"]):
    piv = cells[cells.model == m].pivot(index="prep", columns="rep", values="tuned_cv_f1")[list(REPRESENT)].loc[list(PREPROCESS)]
    sns.heatmap(piv, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax, cbar=False, vmin=0.8, vmax=1)
    ax.set_title(f"{m} - F1 CV sau tinh chỉnh tham số"); ax.set_xlabel(""); ax.set_ylabel("")
plt.tight_layout(); plt.savefig(FIG / "stage2_heatmap_tuned_cv_f1.png", dpi=170); plt.close()

# đường cong tham số (lấy từ grid, các tham số còn lại ở giá trị tốt nhất)
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
g = grid_tables["MultinomialNB"]; g = g[g.param_clf__fit_prior == best_cfg["MultinomialNB"]["best_params"]["fit_prior"]]
axes[0].semilogx(g.param_clf__alpha.astype(float), g.mean_test_score, "o-"); axes[0].set_xlabel("alpha"); axes[0].set_title("MultinomialNB: F1 theo alpha")
g = grid_tables["KNN"]; bp = best_cfg["KNN"]["best_params"]
for metric in ["cosine", "euclidean"]:
    gg = g[(g.param_clf__metric == metric) & (g.param_clf__weights == bp["weights"])]
    axes[1].plot(gg.param_clf__n_neighbors.astype(int), gg.mean_test_score, "o-", label=metric)
axes[1].set_xlabel("k"); axes[1].set_title("K-NN: F1 theo k và metric"); axes[1].legend()
g = grid_tables["DecisionTree"]; bp = best_cfg["DecisionTree"]["best_params"]
for crit in ["gini", "entropy"]:
    gg = g[(g.param_clf__criterion == crit) & (g.param_clf__min_samples_leaf == bp["min_samples_leaf"])].copy()
    gg["d"] = gg.param_clf__max_depth.apply(lambda v: 60 if v is None or (isinstance(v, float) and np.isnan(v)) else int(v))
    gg = gg.sort_values("d")
    axes[2].plot(gg.d, gg.mean_test_score, "o-", label=crit)
axes[2].set_xlabel("max_depth (60 = không giới hạn)"); axes[2].set_title("Cây quyết định: F1 theo độ sâu"); axes[2].legend()
for a in axes: a.set_ylabel("F1 spam (CV)")
plt.tight_layout(); plt.savefig(FIG / "stage2_param_curves.png", dpi=170); plt.close()

# ------------------------------------------------------ 4. đánh giá trên test
final_rows, preds, scores = [], {}, {}
for m in MODELS:
    pipe = tuned[m]
    preds[m] = pipe.predict(X_te); scores[m] = get_score(pipe, X_te)
    final_rows.append(dict(model=m, **metrics(y_te, preds[m], scores[m])))
final = pd.DataFrame(final_rows)
final.to_csv(RES / "final_test_results.csv", index=False)
print(final.round(4))

# bảng tổng hợp baseline vs tuned
cmp_rows = []
for m in MODELS:
    b = base.loc[m]
    f = final.set_index("model").loc[m]
    cmp_rows.append(dict(model=m, baseline_f1=b.test_f1, tuned_f1=f.f1, baseline_precision=b.test_precision,
                         tuned_precision=f.precision, baseline_recall=b.test_recall, tuned_recall=f.recall,
                         baseline_acc=b.test_accuracy, tuned_acc=f.accuracy))
pd.DataFrame(cmp_rows).to_csv(RES / "baseline_vs_tuned.csv", index=False)

# confusion matrix
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, m in zip(axes, MODELS):
    cm = confusion_matrix(y_te, preds[m])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
    ax.set_title(m); ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế")
    summary.setdefault("confusion", {})[m] = cm.tolist()
plt.tight_layout(); plt.savefig(FIG / "confusion_matrices.png", dpi=170); plt.close()

# ROC
plt.figure(figsize=(5.2, 4.4))
for m in MODELS:
    fpr, tpr, _ = roc_curve(y_te, scores[m])
    plt.plot(fpr, tpr, label=f"{m} (AUC={roc_auc_score(y_te, scores[m]):.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=0.8); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC trên tập test"); plt.legend(loc="lower right"); plt.tight_layout()
plt.savefig(FIG / "roc_curves.png", dpi=170); plt.close()

# bar chart so sánh metric
melt = final.melt(id_vars="model", value_vars=["accuracy", "precision", "recall", "f1"])
plt.figure(figsize=(7, 3.8))
sns.barplot(data=melt, x="variable", y="value", hue="model")
plt.ylim(0.7, 1.01); plt.xlabel(""); plt.ylabel("Giá trị"); plt.title("So sánh mô hình sau tinh chỉnh (test)")
plt.legend(loc="lower left", fontsize=8); plt.tight_layout(); plt.savefig(FIG / "final_metric_bars.png", dpi=170); plt.close()

# ổn định theo nhiều seed (siêu tham số cố định, chia lại dữ liệu)
seed_rows = []
for seed in range(5):
    Xa, Xb, ya, yb = train_test_split(df["text"], df["y"], test_size=0.2, stratify=df["y"], random_state=seed)
    for m in MODELS:
        from sklearn.base import clone
        p = clone(tuned[m]).fit(Xa, ya)
        seed_rows.append(dict(seed=seed, model=m, **metrics(yb, p.predict(Xb), get_score(p, Xb))))
seeds = pd.DataFrame(seed_rows)
seeds.to_csv(RES / "seed_stability_raw.csv", index=False)
stab = seeds.groupby("model")[["accuracy", "precision", "recall", "f1", "roc_auc"]].agg(["mean", "std"]).round(4)
stab.to_csv(RES / "seed_stability_summary.csv")
print(stab)

# --------------------------------------------------------- phân tích lỗi
best_model = final.sort_values("f1", ascending=False).iloc[0]["model"]
summary["best_model_by_test_f1"] = best_model
err = pd.DataFrame({"text": X_te.values, "y": y_te.values, "pred": preds[best_model]})
fp = err[(err.y == 0) & (err.pred == 1)]
fn = err[(err.y == 1) & (err.pred == 0)]
fp.to_csv(RES / f"errors_false_positive_{best_model}.csv", index=False)
fn.to_csv(RES / f"errors_false_negative_{best_model}.csv", index=False)
summary["error_counts"] = dict(model=best_model, FP=len(fp), FN=len(fn))
# lỗi chung của 3 mô hình
wrong = {m: set(np.where(preds[m] != y_te.values)[0]) for m in MODELS}
summary["errors_all_models"] = len(set.intersection(*wrong.values()))
summary["errors_per_model"] = {m: len(w) for m, w in wrong.items()}

tp = err[(err.y == 1) & (err.pred == 1)]
def prof(d):
    return dict(n=len(d), mean_chars=float(d.text.str.len().mean()) if len(d) else None,
                digit_seq=float(d.text.str.contains(r"\d{5,}").mean()) if len(d) else None,
                url=float(d.text.str.contains(r"www\.|http|\.com|\.co\.uk", case=False).mean()) if len(d) else None,
                currency=float(d.text.str.contains(r"[£$€]").mean()) if len(d) else None)
summary["error_profile"] = dict(FN=prof(fn), TP=prof(tp), FP=prof(fp))
print("FP examples:"); [print("  -", t[:140]) for t in fp.text.head(8)]
print("FN examples:"); [print("  -", t[:140]) for t in fn.text.head(8)]

# từ có trọng số cao nhất của Naive Bayes (log-odds spam vs ham)
nb = tuned["MultinomialNB"]
vec, clf = nb.named_steps["vec"], nb.named_steps["clf"]
lo = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
names = vec.get_feature_names_out()
top_spam = [names[i] for i in np.argsort(lo)[::-1][:15]]
top_ham = [names[i] for i in np.argsort(lo)[:15]]
summary["nb_top_spam_terms"], summary["nb_top_ham_terms"] = top_spam, top_ham

# độ phức tạp: số đặc trưng, độ sâu cây, thời gian dự đoán
dt = tuned["DecisionTree"].named_steps["clf"]
summary["dt_depth"] = int(dt.get_depth()); summary["dt_leaves"] = int(dt.get_n_leaves())
dt_names = tuned["DecisionTree"].named_steps["vec"].get_feature_names_out()
imp = np.argsort(dt.feature_importances_)[::-1][:10]
summary["dt_top_features"] = [(dt_names[i], round(float(dt.feature_importances_[i]), 3)) for i in imp]
timing = {}
for m in MODELS:
    t = time.time(); tuned[m].predict(X_te); timing[m] = round(time.time() - t, 4)
summary["predict_time_sec_test"] = timing
summary["n_features"] = {m: len(tuned[m].named_steps["vec"].get_feature_names_out()) for m in MODELS}

# lưu mô hình tốt nhất để demo
joblib.dump(tuned[best_model], ROOT / "results" / "best_model.joblib")
(RES / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
print("DONE. best model:", best_model)

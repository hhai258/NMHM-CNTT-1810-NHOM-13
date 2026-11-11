"""Kiểm tra độ nhạy: lưới alpha của Multinomial NB mở rộng tới 10 (cấu hình P2 + BoW unigram).
Mục đích: lưới chính dừng ở alpha=2 nên cần xem xu hướng có tiếp tục tăng không."""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from common import load_raw, make_pipeline, RANDOM_STATE, ROOT

df = load_raw().drop_duplicates("text")
X_tr, X_te, y_tr, y_te = train_test_split(df.text, df.y, test_size=0.2, stratify=df.y, random_state=RANDOM_STATE)
rows = []
for seed in [RANDOM_STATE, 1, 2]:     # lặp CV với nhiều cách chia fold để xem độ nhiễu
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    for a in [0.1, 0.5, 1, 2, 3, 5, 10]:
        r = cross_validate(make_pipeline("MultinomialNB", "P2_norm_stop", "BoW_uni", model=MultinomialNB(alpha=a)),
                           X_tr, y_tr, cv=cv, scoring=["f1", "precision", "recall"])
        rows.append(dict(cv_seed=seed, alpha=a, f1=r["test_f1"].mean(), precision=r["test_precision"].mean(), recall=r["test_recall"].mean()))
out = pd.DataFrame(rows); out.to_csv(ROOT / "results" / "extra_nb_alpha_check.csv", index=False)
print(out.groupby("alpha")[["f1", "precision", "recall"]].mean().round(4))
print(out.pivot(index="alpha", columns="cv_seed", values="f1").round(4))

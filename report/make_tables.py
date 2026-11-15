"""Sinh các bảng LaTeX từ file kết quả trong results/ để số liệu trong báo cáo luôn khớp với code.
Chạy:  python report/make_tables.py   (sau khi đã chạy src/run_experiments.py)"""
import json
import re
from pathlib import Path
import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"
out = []
S = json.load(open(R / "summary.json", encoding="utf-8"))
f4 = lambda x: f"{x:.4f}"
NAMES = {"MultinomialNB": "Multinomial NB", "KNN": "K-NN", "DecisionTree": "Cây quyết định"}


def fix(c):
    c = re.sub(r"(?<!\\)%", r"\\%", str(c))          # escape %
    return re.sub(r"(?<=\d)\.(?=\d)", ",", c)         # dấu phẩy thập phân


def table(name, caption, label, header, rows, colfmt, fs="\\small", tabsep="6pt"):
    body = " \\\\\n".join(" & ".join(fix(c) for c in r) for r in rows)
    out.append(f"""\\newcommand{{\\{name}}}{{%
\\begin{{table}}[H]\\centering{fs}\\setlength{{\\tabcolsep}}{{{tabsep}}}
\\caption{{{caption}}}\\label{{{label}}}
\\begin{{tabular}}{{{colfmt}}}\\toprule
{header} \\\\\\midrule
{body} \\\\\\bottomrule
\\end{{tabular}}\\end{{table}}}}
""")


# 1. thống kê dữ liệu
rc, uc = S["raw_counts"], S["unique_counts"]
table("TabData", "Thống kê bộ dữ liệu trước và sau khi loại tin nhắn trùng lặp.", "tab:data",
      "Tập & Tổng & Ham & Spam & Tỉ lệ spam",
      [["Dữ liệu thô", S["n_raw"], rc["ham"], rc["spam"], f"{rc['spam']/S['n_raw']:.1%}"],
       ["Sau khi bỏ trùng", S["n_unique"], uc["ham"], uc["spam"], f"{uc['spam']/S['n_unique']:.1%}"],
       ["Train (80\\%)", S["split"]["train"], S["split"]["train"] - S["split"]["train_spam"], S["split"]["train_spam"], f"{S['split']['train_spam']/S['split']['train']:.1%}"],
       ["Test (20\\%)", S["split"]["test"], S["split"]["test"] - S["split"]["test_spam"], S["split"]["test_spam"], f"{S['split']['test_spam']/S['split']['test']:.1%}"]],
      "lrrrr")

# 2. tín hiệu EDA
sr, ls = S["signal_rates"], S["length_stats"]
table("TabSignal", "Đặc điểm khác biệt giữa ham và spam (sau khi bỏ trùng).", "tab:signal",
      "Đặc điểm & Ham & Spam",
      [["Độ dài trung bình (ký tự)", f"{ls['ham|n_chars|mean']:.1f}", f"{ls['spam|n_chars|mean']:.1f}"],
       ["Độ dài trung vị (ký tự)", f"{ls['ham|n_chars|median']:.0f}", f"{ls['spam|n_chars|median']:.0f}"],
       ["Độ dài lớn nhất (ký tự)", f"{ls['ham|n_chars|max']:.0f}", f"{ls['spam|n_chars|max']:.0f}"],
       ["Số từ trung bình", f"{ls['ham|n_words|mean']:.1f}", f"{ls['spam|n_words|mean']:.1f}"],
       ["Chứa chuỗi $\\geq 5$ chữ số", f"{sr['ham|has_digit_seq']:.1%}", f"{sr['spam|has_digit_seq']:.1%}"],
       ["Chứa URL/tên miền", f"{sr['ham|has_url']:.1%}", f"{sr['spam|has_url']:.1%}"],
       ["Chứa ký hiệu tiền tệ", f"{sr['ham|has_currency']:.1%}", f"{sr['spam|has_currency']:.1%}"],
       ["Tỉ lệ chữ HOA trung bình", f"{sr['ham|upper_ratio']:.1%}", f"{sr['spam|upper_ratio']:.1%}"]],
      "lrr")

# 3. cấu hình tốt nhất
rows = []
for m, c in S["best_cfg"].items():
    bp = ", ".join(f"{k}={v}" for k, v in c["best_params"].items()).replace("_", "\\_")
    rows.append([NAMES[m], c["prep"].replace("_", "\\_"), c["rep"].replace("_", "\\_"), bp,
                 f4(c["stage1_cv_f1"]), f4(c["grid_cv_f1"])])
table("TabCfg", "Cấu hình được chọn cho từng mô hình (chọn theo F1 của 5-fold CV trên tập train).", "tab:cfg",
      "Mô hình & Tiền xử lý & Biểu diễn & Siêu tham số & CV-F1 mặc định & CV-F1 tinh chỉnh",
      rows, "llllrr".replace("llllrr","lllp{3.4cm}rr"), fs="\\scriptsize", tabsep="3pt")

# 4. baseline vs tuned
b = pd.read_csv(R / "baseline_vs_tuned.csv")
rows = []
for _, r in b.iterrows():
    rows.append([NAMES[r.model], "Baseline", f4(r.baseline_acc), f4(r.baseline_precision), f4(r.baseline_recall), f4(r.baseline_f1)])
    rows.append(["", "Tinh chỉnh", f4(r.tuned_acc), f4(r.tuned_precision), f4(r.tuned_recall), f"\\textbf{{{f4(r.tuned_f1)}}}"])
table("TabBaseTuned", "Kết quả trên tập test: baseline (BoW unigram, không tiền xử lý, tham số mặc định) và cấu hình tinh chỉnh. Spam là lớp dương.", "tab:basetuned",
      "Mô hình & Phương án & Accuracy & Precision & Recall & F1", rows, "llrrrr")

# 5. kết quả cuối
f = pd.read_csv(R / "final_test_results.csv")
cm = S["confusion"]
rows = []
for _, r in f.iterrows():
    c = cm[r.model]
    rows.append([NAMES[r.model], f4(r.accuracy), f4(r.precision), f4(r.recall), f4(r.f1), f4(r.roc_auc),
                 f"{c[0][1]} / {c[1][0]}"])
table("TabFinal", "Kết quả cuối cùng trên tập test (1035 tin nhắn, 131 spam), một lần chia với random\\_state=42.", "tab:final",
      "Mô hình & Accuracy & Precision & Recall & F1 & ROC-AUC & FP / FN", rows, "lrrrrrr")

# 6. ổn định theo seed
sd = pd.read_csv(R / "seed_stability_summary.csv", header=[0, 1], index_col=0)
rows = []
for m in ["MultinomialNB", "KNN", "DecisionTree"]:
    r = sd.loc[m]
    cell = lambda k: f"{r[(k,'mean')]:.4f} $\\pm$ {r[(k,'std')]:.4f}"
    rows.append([NAMES[m], cell("accuracy"), cell("precision"), cell("recall"), cell("f1"), cell("roc_auc")])
table("TabSeeds", "Độ ổn định: trung bình $\\pm$ độ lệch chuẩn trên 5 lần chia train/test khác nhau (seed 0--4), siêu tham số giữ cố định.", "tab:seeds",
      "Mô hình & Accuracy & Precision & Recall & F1 & ROC-AUC", rows, "lccccc", fs="\\scriptsize", tabsep="3pt")

# 7. alpha sensitivity
a = pd.read_csv(R / "extra_nb_alpha_check.csv").groupby("alpha")[["f1", "precision", "recall"]].mean()
rows = [[f"{i:g}", f4(r.f1), f4(r.precision), f4(r.recall)] for i, r in a.iterrows()]
table("TabAlpha", "Độ nhạy của Multinomial NB theo $\\alpha$ (CV-F1 trung bình qua 3 cách chia fold; cấu hình P2 + BoW unigram).", "tab:alpha",
      "$\\alpha$ & F1 & Precision & Recall", rows, "rrrr")


# 9. ảnh hưởng tham số vector hoá (thí nghiệm có kiểm soát)
vs = pd.read_csv(R / "vectorizer_sensitivity.csv")
g = vs[(vs.study == "grid") & (vs.max_df == 1.0)]
rows = []
for m in ["MultinomialNB", "KNN", "DecisionTree"]:
    for k, ng in enumerate(["1-1", "1-2"]):
        d = g[(g.model == m) & (g.ngram == ng)].set_index("min_df").cv_f1
        rows.append([NAMES[m] if k == 0 else "", "unigram" if ng == "1-1" else "unigram + bigram"] + [f4(d[v]) for v in [1, 2, 3, 5]])
table("TabVecGrid", "F1 CV (5-fold, tập train) theo n-gram và \\texttt{min\\_df}, các tham số khác giữ nguyên (\\texttt{max\\_df}=1,0). Với \\texttt{max\\_df}=0,5 cho kết quả \\emph{giống hệt}.", "tab:vecgrid",
      "Mô hình & Biểu diễn & min\\_df=1 & 2 & 3 & 5", rows, "llrrrr")
f = vs[vs.study == "max_features"]
cols = [200, 500, 1000, 2000, 5000, 99999]
rows = []
for m in ["MultinomialNB", "KNN", "DecisionTree"]:
    d = f[f.model == m].set_index("max_features").cv_f1
    rows.append([NAMES[m]] + [f4(d[c]) for c in cols])
table("TabVecFeat", "F1 CV theo số đặc trưng tối đa (\\texttt{max\\_features}, unigram); cột cuối là toàn bộ từ vựng (khoảng 6\\,500 từ).", "tab:vecfeat",
      "Mô hình & 200 & 500 & 1000 & 2000 & 5000 & Tất cả", rows, "lrrrrrr")

# 8. từ khoá
out.append("\\newcommand{\\NBtopspam}{" + ", ".join(f"\\texttt{{{t}}}" for t in S["nb_top_spam_terms"][:10]) + "}\n")
out.append("\\newcommand{\\NBtopham}{" + ", ".join(f"\\texttt{{{t}}}" for t in S["nb_top_ham_terms"][:10]) + "}\n")
dtf = ", ".join(f"\\texttt{{{n}}} ({v:.3f})" for n, v in S["dt_top_features"][:6])
out.append("\\newcommand{\\DTtop}{" + dtf + "}\n")
(Path(__file__).parent / "tables.tex").write_text("\n".join(out), encoding="utf-8")
print("wrote tables.tex")

"""Demo dòng lệnh: dùng mô hình tốt nhất đã huấn luyện để phân loại tin nhắn.

Cách dùng:
    python src/predict.py "WINNER!! Call 09061701461 to claim your prize"
    python src/predict.py            # chế độ nhập tương tác, thoát bằng dòng trống
Cần chạy src/run_experiments.py trước để tạo results/best_model.joblib.
"""
import sys
from pathlib import Path

import joblib

import common  # noqa: F401  (cần để nạp hàm normalize_text khi giải nén mô hình)

MODEL_PATH = Path(__file__).resolve().parents[1] / "results" / "best_model.joblib"


def classify(model, text):
    label = "SPAM" if model.predict([text])[0] == 1 else "HAM"
    clf = model.named_steps["clf"]
    prob = model.predict_proba([text])[0, 1] if hasattr(clf, "predict_proba") else None
    return label, prob


def main():
    if not MODEL_PATH.exists():
        sys.exit("Chưa có mô hình. Hãy chạy: python src/run_experiments.py")
    model = joblib.load(MODEL_PATH)
    texts = sys.argv[1:]
    if texts:
        for t in texts:
            lab, p = classify(model, t)
            print(f"{lab}\t(P(spam)={p:.3f})\t{t}" if p is not None else f"{lab}\t{t}")
        return
    print("Nhập tin nhắn (dòng trống để thoát):")
    while True:
        t = input("> ").strip()
        if not t:
            break
        lab, p = classify(model, t)
        print(f"  => {lab}" + (f"  (P(spam)={p:.3f})" if p is not None else ""))


if __name__ == "__main__":
    main()

# Đề tài 3: Phân loại tin nhắn rác (SMS Spam Collection)

Bài tập lớn học phần **Nhập môn Học máy** (FIT4013) – Đại học Đại Nam, năm học 2026–2027.
So sánh **Multinomial Naive Bayes**, **K-NN** và **Cây quyết định** với baseline, trên các cách tiền xử lý
(P0/P1/P2) và biểu diễn văn bản (BoW/TF-IDF, unigram/bigram).

**Nhóm 13:** Nguyễn Hoàng Hải (1871020211) – Nguyễn Thành Đạt (1871020135)

## Dữ liệu
- *SMS Spam Collection v1* (Almeida & Hidalgo, 2011): 5 574 tin nhắn tiếng Anh gắn nhãn ham/spam.
- **Nguồn tải:** https://archive.ics.uci.edu/dataset/228/sms+spam+collection (DOI: 10.24432/C5CC84) – **giấy phép CC BY 4.0** (được chia sẻ/điều chỉnh khi ghi nguồn).
- Bản sao nằm trong `data/SMSSpamCollection` (≈ 0,5 MB) kèm `data/readme_dataset.txt`. Có thể tải lại từ trang UCI nếu cần.
- File mã hoá UTF-8, phân tách bằng tab. Có **403 tin trùng lặp** → được loại **trước khi chia** train/test (còn 5 171 tin).

## Cấu trúc
```
src/common.py               nạp dữ liệu, chuẩn hoá văn bản, dựng Pipeline
src/run_experiments.py      toàn bộ thí nghiệm (EDA → so sánh → GridSearchCV → test → phân tích lỗi)
src/extra_nb_alpha_check.py kiểm tra độ nhạy alpha của Naive Bayes
src/vectorizer_sensitivity.py ảnh hưởng của n-gram, min_df, max_df, số đặc trưng (có kiểm soát)
src/predict.py              demo dòng lệnh
src/demo_app.py             ứng dụng demo có giao diện web (Streamlit)
notebooks/sms_spam_pipeline.ipynb   cùng quy trình, chạy từng bước trong Jupyter (đã kèm kết quả)
results/                    bảng CSV, summary.json, danh sách tin dự đoán sai, run_log.txt
figures/                    hình dùng trong báo cáo
report/main.tex             báo cáo (XeLaTeX); report/make_tables.py sinh bảng từ results/
AI_USAGE.md                 khai báo sử dụng AI
```

## Chạy lại
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/run_experiments.py          # ≈ 8–10 phút trên máy 1 lõi; sinh results/ và figures/
python src/extra_nb_alpha_check.py     # (tuỳ chọn) kiểm tra độ nhạy alpha, vài chục giây
python src/vectorizer_sensitivity.py   # (tuỳ chọn) tham số vector hoá, khoảng 2-3 phút

python src/predict.py "WINNER!! Call 09061701461 to claim your prize"   # demo

cd report && python make_tables.py && xelatex main.tex && xelatex main.tex   # biên dịch báo cáo (cần font FreeSerif/FreeSans/FreeMono hoặc đổi font trong main.tex)
```

## Phân công nhóm *(bổ sung phần việc trước khi nộp)*
| STT | Thành viên | MSSV | GitHub | Phần việc chính |
|---|---|---|---|---|
| 1 | Nguyễn Hoàng Hải | 1871020211 | [@hhai258](https://github.com/hhai258) | |
| 2 | Nguyễn Thành Đạt | 1871020135 | [@th4nhdat2611](https://github.com/th4nhdat2611) | |

Log chạy thí nghiệm: `results/run_log.txt`, `results/vectorizer_sensitivity_log.txt`.

## Chạy ứng dụng demo (giao diện)
```bash
pip install -r requirements-demo.txt
streamlit run src/demo_app.py
```
Trình duyệt tự mở trang web: dán một tin nhắn tiếng Anh (hoặc bấm tin mẫu) để xem kết quả **SPAM/HAM**, xác suất spam, so sánh cả 3 mô hình và (với Naive Bayes) các từ đẩy tin về spam hay ham. Lần mở đầu mất vài giây để huấn luyện mô hình, không cần chạy `run_experiments.py` trước. Có 8 tin mẫu (trong đó nhiều tin khiến ba mô hình bất đồng) và gợi ý giải thích cho từng tin; kịch bản trình bày xem `HUONG_DAN_DEMO.md`.

## Dùng Jupyter Notebook
```bash
pip install jupyter           # hoặc dùng Jupyter trong VS Code / Google Colab
jupyter notebook notebooks/sms_spam_pipeline.ipynb
```
Chạy lần lượt từng ô (*Kernel → Restart & Run All*). Notebook import hàm từ `src/common.py` nên cho cùng kết quả với script. Hai bước nặng có công tắc ở ô thứ hai: `RUN_STAGE1` (≈ 1 phút) và `RUN_GRID` (≈ 5–8 phút, mặc định `False` = đọc kết quả đã lưu trong `results/`).
Trên Colab: tải cả thư mục lên (giữ nguyên cấu trúc `src/`, `data/`, `notebooks/`, `results/`), mở notebook và chạy.

## Thiết lập để tái tạo kết quả
- Chia dữ liệu: 80/20 có phân tầng, `random_state=42` (train 4 136 tin, test 1 035 tin).
- Chọn cấu hình: 5-fold Stratified CV (`shuffle=True`, `random_state=42`) **chỉ trên tập train**; tiêu chí F1 của lớp spam.
- Bộ vector hoá nằm trong `sklearn.pipeline.Pipeline`, nên chỉ fit trên dữ liệu huấn luyện của từng fold (tránh data leakage).
- Cây quyết định: `random_state=42`. Độ ổn định: lặp lại với seed 0–4, siêu tham số cố định.
- Kết quả có thể lệch rất nhỏ giữa các phiên bản scikit-learn khác nhau.

## Kết quả chính (tập test, lớp dương = spam)
| Mô hình | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Multinomial NB (P2 + BoW unigram, α=2) | 0,9913 | 0,9766 | 0,9542 | 0,9653 |
| K-NN (P1 + TF-IDF unigram, cosine, k=9, distance) | 0,9826 | 0,9593 | 0,9008 | 0,9291 |
| Cây quyết định (P2 + BoW unigram, gini, max_depth=10) | 0,9874 | 1,0000 | 0,9008 | 0,9478 |

Qua 5 lần chia khác nhau (F1 trung bình ± độ lệch chuẩn): NB 0,947 ± 0,020; K-NN 0,934 ± 0,015; Cây quyết định 0,907 ± 0,018.
Chênh lệch NB so với K-NN nhỏ hơn độ lệch chuẩn nên chưa đủ để khẳng định chắc chắn. Chi tiết, phân tích lỗi và các hạn chế xem trong báo cáo.

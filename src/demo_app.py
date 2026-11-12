"""Ứng dụng demo (Streamlit): dán một tin nhắn và xem mô hình phân loại SPAM/HAM.

Chạy:  streamlit run src/demo_app.py
Lần mở đầu, ứng dụng huấn luyện 3 mô hình (vài giây) với đúng cấu hình đã chọn trong báo cáo.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from common import RANDOM_STATE, load_raw, make_pipeline, normalize_text

NAMES = {"MultinomialNB": "Multinomial Naive Bayes", "KNN": "K-NN", "DecisionTree": "Cây quyết định"}
EXAMPLES = {   # nhãn hiển thị: (nội dung, gợi ý khi demo). Hành vi dưới đây ứng với cấu hình hiện tại.
    "— Tự nhập tin nhắn —": ("", ""),
    "1. Spam rõ ràng (cả 3 đúng)": (
        "You have won a £1000 cash prize! Call 09061701461 to claim",
        "Có số điện thoại dài, tiền tệ và các từ 'won', 'prize', 'claim': cả ba mô hình đều bắt được."),
    "2. Ham rõ ràng (cả 3 đúng)": (
        "Call me when you get home, I'll be at the mall",
        "Tin đời thường, không có dấu hiệu spam: cả ba mô hình đều cho xác suất spam gần 0."),
    "3. Ham chứa từ 'free/prize' (NB, K-NN báo nhầm)": (
        "I won't be free to call you tonight, I'm at the prize ceremony with my dad",
        "Tin hợp lệ nhưng có 'free', 'prize'. Naive Bayes và K-NN dựa vào từ khoá nên báo nhầm; Cây quyết định đúng vì không thấy số dài hay URL."),
    "4. Spam có URL (K-NN bỏ sót)": (
        "Your parcel is waiting. Pay the delivery fee at www.parcel-track.net",
        "Cây quyết định bắt được nhờ dấu hiệu URL; K-NN bỏ sót vì các tin gần nhất với nó phần lớn là tin thường."),
    "5. Spam 'missed call' (K-NN sát ngưỡng)": (
        "Missed call alert. These numbers called but left no message. 07008009200",
        "Cây quyết định và Naive Bayes bắt được; K-NN cho 0,44, tức sát ngưỡng 0,5 nên bỏ sót."),
    "6. Spam không có số/URL (Cây bỏ sót)": (
        "RECPT 1/3. You have ordered a Ringtone. Your order is being processed...",
        "Không có số điện thoại hay URL nên Cây quyết định bỏ sót; Naive Bayes và K-NN bắt được nhờ từ 'ringtone'."),
    "7. Ham ngắn chứa 'nokia' (Cây đúng)": (
        "Nokia phone is lovly..",
        "Tin ngắn có từ 'nokia' (hay gặp trong spam quảng cáo). Naive Bayes và K-NN báo nhầm; Cây quyết định đúng."),
    "8. Spam khó: cả 3 mô hình đều sai": (
        "In The Simpsons Movie released in July 2007 name the band that died at the start of the film? A-Green Day, B-Blue Day, C-Red Day. (Send A, B or C)",
        "Spam kiểu đố vui, không có dấu hiệu điển hình: cả ba mô hình bỏ sót. Đây là hạn chế chung của bài toán."),
}


@st.cache_resource(show_spinner="Đang huấn luyện mô hình...")
def train_models():
    """Cấu hình (tiền xử lý, biểu diễn, siêu tham số) lấy từ kết quả CV trong báo cáo."""
    best = {
        "MultinomialNB": ("P2_norm_stop", "BoW_uni", MultinomialNB(alpha=2.0, fit_prior=True)),
        "KNN": ("P1_normalize", "TFIDF_uni", KNeighborsClassifier(n_neighbors=9, metric="cosine", weights="distance")),
        "DecisionTree": ("P2_norm_stop", "BoW_uni", DecisionTreeClassifier(criterion="gini", max_depth=10, random_state=RANDOM_STATE)),
    }
    df = load_raw().drop_duplicates("text")
    X_tr, _, y_tr, _ = train_test_split(df.text, df.y, test_size=0.2, stratify=df.y, random_state=RANDOM_STATE)
    return {m: make_pipeline(m, p, r, model=clf).fit(X_tr, y_tr) for m, (p, r, clf) in best.items()}


st.set_page_config(page_title="Phân loại tin nhắn rác", page_icon="✉️")
st.title("✉️ Phân loại tin nhắn rác")
st.caption("Đề tài 3 – Nhập môn Học máy · SMS Spam Collection · tin nhắn tiếng Anh")
models = train_models()

model_key = st.sidebar.radio("Mô hình", list(NAMES), format_func=NAMES.get)
st.sidebar.caption("Mô hình huấn luyện trên 80% dữ liệu (đã bỏ trùng). Dữ liệu là tiếng Anh nên hãy nhập tin nhắn tiếng Anh.")

def pick_example():
    txt = EXAMPLES[st.session_state.example][0]
    if txt:
        st.session_state.text = txt


if "text" not in st.session_state:
    st.session_state.text = EXAMPLES["1. Spam rõ ràng (cả 3 đúng)"][0]
st.selectbox("Chọn tin mẫu (hoặc tự nhập ở ô bên dưới):", list(EXAMPLES), index=1, key="example", on_change=pick_example)
text = st.text_area("Nhập tin nhắn:", key="text", height=110)
hint = next((h for t, h in EXAMPLES.values() if t and t == text.strip()), None)
if hint:
    st.info("💡 " + hint)

if text.strip():
    pipe = models[model_key]
    proba = pipe.predict_proba([text])[0, 1]
    is_spam = pipe.predict([text])[0] == 1
    (st.error if is_spam else st.success)(f"**{'SPAM (tin rác)' if is_spam else 'HAM (tin hợp lệ)'}**  ·  xác suất spam = {proba:.1%}")

    st.subheader("So sánh ba mô hình")
    rows = []
    for k, p in models.items():
        pr = p.predict_proba([text])[0, 1]
        rows.append({"Mô hình": NAMES[k], "Dự đoán": "SPAM" if p.predict([text])[0] == 1 else "HAM", "P(spam)": f"{pr:.1%}"})
    st.table(pd.DataFrame(rows).set_index("Mô hình"))

    with st.expander("Mô hình nhìn thấy gì? (tiền xử lý và từ ảnh hưởng)"):
        st.write("Tin nhắn sau chuẩn hoá (số dài → `longnumtoken`, URL → `urltoken`, tiền tệ → `currtoken`):")
        st.code(normalize_text(text))
        if model_key == "MultinomialNB":
            vec, clf = pipe.named_steps["vec"], pipe.named_steps["clf"]
            X = vec.transform([text])
            idx = X.nonzero()[1]
            if len(idx):
                names = vec.get_feature_names_out()
                lo = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
                contrib = np.asarray(X[0, idx].todense()).ravel() * lo[idx]
                df_c = pd.DataFrame({"từ": names[idx], "ảnh hưởng (log-odds)": contrib.round(2)})
                df_c = df_c.sort_values("ảnh hưởng (log-odds)", ascending=False)
                st.write("Từ đẩy về **spam** (dương) hoặc **ham** (âm):")
                st.dataframe(df_c, hide_index=True)
            else:
                st.write("Không có từ nào nằm trong từ vựng của mô hình.")
        else:
            st.caption("Chọn Multinomial Naive Bayes ở thanh bên để xem từng từ ảnh hưởng thế nào.")
else:
    st.info("Hãy nhập một tin nhắn để phân loại.")

# =====================================================================
# MINH CHỨNG BÁO CÁO: BẢNG 5.4 (TOP LOG-ODDS MULTINOMIAL NAIVE BAYES)
# =====================================================================
st.divider()
with st.expander("📊 Minh chứng Báo cáo: Bảng 5.4 – Top từ vựng theo Log-Odds (Multinomial Naive Bayes)", expanded=True):
    st.markdown("""
    **Bảng 5.4 trong Báo cáo:** Tỷ số Log-Odds được trích xuất trực tiếp từ mô hình tối ưu **Multinomial Naive Bayes** ($\\\\alpha=2.0$, tiền xử lý P2, BoW unigram):
    $$\\text{Log-Odds}(w) = \\ln P(w \\mid \\text{spam}) - \\ln P(w \\mid \\text{ham}) = \\text{feature\\_log\\_prob\\_}[1] - \\text{feature\\_log\\_prob\\_}[0]$$
    *Từ có Log-Odds càng dương thì càng đặc trưng cho tin nhắn rác (Spam). Từ có Log-Odds càng âm thì càng đặc trưng cho tin nhắn hợp lệ (Ham).*
    """)
    nb_pipe = models["MultinomialNB"]
    vec_nb, clf_nb = nb_pipe.named_steps["vec"], nb_pipe.named_steps["clf"]
    lo_all = clf_nb.feature_log_prob_[1] - clf_nb.feature_log_prob_[0]
    names_all = vec_nb.get_feature_names_out()

    top_n = st.slider("Số lượng từ vựng hiển thị:", min_value=5, max_value=30, value=15, step=5)
    
    top_spam_idx = np.argsort(lo_all)[::-1][:top_n]
    top_ham_idx = np.argsort(lo_all)[:top_n]

    df_spam = pd.DataFrame({
        "Thứ hạng": range(1, top_n + 1),
        "Từ vựng (Spam)": names_all[top_spam_idx],
        "Log-Odds": lo_all[top_spam_idx].round(2),
        "Xu hướng": ["Đẩy mạnh về SPAM" for _ in range(top_n)]
    })
    
    df_ham = pd.DataFrame({
        "Thứ hạng": range(1, top_n + 1),
        "Từ vựng (Ham)": names_all[top_ham_idx],
        "Log-Odds": lo_all[top_ham_idx].round(2),
        "Xu hướng": ["Đẩy mạnh về HAM" for _ in range(top_n)]
    })

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"🔴 Top {top_n} từ nghiêng về SPAM")
        st.dataframe(df_spam, use_container_width=True, hide_index=True)
    with col2:
        st.subheader(f"🟢 Top {top_n} từ nghiêng về HAM")
        st.dataframe(df_ham, use_container_width=True, hide_index=True)

    st.info("💡 **Ghi chú bảo vệ:** Token `longnumtoken` đại diện cho chuỗi số điện thoại $\\\\ge 5$ chữ số (đặc trưng mạnh nhất của Spam); hai token `gt`, `lt` là tàn dư từ thẻ ẩn danh HTML `&lt;#&gt;` trong dữ liệu gốc UCI (được nhóm phát hiện và phân tích sâu trong Chương 5 của Báo cáo).")


# Kịch bản demo ứng dụng phân loại tin nhắn rác (≈ 5–6 phút)

## Trước giờ bảo vệ
1. Chạy thử: `streamlit run src/demo_app.py` (kích hoạt `.venv` trước). Trang mở ở `localhost:8501`.
2. Phóng to trình duyệt (Ctrl và +, khoảng 125%) để cả lớp nhìn rõ. Mở sẵn tab này.
3. Chạy thử cả 8 tin mẫu một lần. Nếu số liệu trên máy bạn lệch nhẹ so với bảng dưới (do khác phiên bản scikit-learn), hãy theo số trên máy của bạn.
4. Chụp sẵn ảnh màn hình vài tin mẫu làm phương án dự phòng. Notebook `notebooks/sms_spam_pipeline.ipynb` là phương án dự phòng thứ hai.

## Kịch bản
| Thời gian | Làm gì | Nói gì |
|---|---|---|
| 0:00 | Mở trang | "Bài toán: cho một tin nhắn, đoán là tin rác (spam) hay tin thường (ham). Dữ liệu 5.574 tin tiếng Anh; sau khi bỏ 403 tin trùng còn 5.171 tin, spam chiếm 12,6%." |
| 0:30 | Chỉ giao diện | "Ô chọn tin mẫu và ô nhập tin ở trên. Thanh bên trái chọn mô hình. Kết quả hiện màu đỏ là SPAM, màu xanh là HAM, kèm xác suất spam. Bảng bên dưới so sánh cả ba mô hình. Mục cuối giải thích mô hình nhìn thấy gì." |
| 1:15 | Chọn **tin 1** rồi **tin 2** | "Tin rõ ràng thì cả ba mô hình đều đúng. Trên 1.035 tin test, ba mô hình đồng ý với nhau ở 1.016 tin (98,2%) và chỉ bất đồng ở 19 tin. Vì vậy phải chọn đúng tin khó mới thấy khác biệt." |
| 2:00 | Chọn **tin 3**, mở mục "Mô hình nhìn thấy gì?" với Naive Bayes | "Tin này hợp lệ nhưng có 'free' và 'prize'. Naive Bayes cộng điểm từng từ nên bị các từ này kéo về spam (xem bảng log-odds). K-NN cũng báo nhầm. Cây quyết định đúng vì không thấy số dài hay URL." |
| 3:00 | Chọn **tin 4**, **tin 5**, **tin 6** | "Cây quyết định hoạt động như chuỗi câu hỏi có/không; đặc trưng quan trọng nhất là token số dài (0,746), rồi URL. Nên nó bắt tin 4, 5 nhờ URL và số, nhưng bỏ sót tin 6 vì không có số hay URL. K-NN tìm 9 tin giống nhất rồi biểu quyết, nên bỏ sót tin 4, 5 khi các tin gần nhất phần lớn là tin thường." |
| 4:15 | Chọn **tin 8** | "Đây là tin spam khó: đố vui, không có dấu hiệu điển hình, cả ba mô hình đều bỏ sót. Đây là một trong các hạn chế đã phân tích ở mục phân tích lỗi trong báo cáo." |
| 4:45 | Nhờ thầy/cô hoặc bạn đưa một tin tiếng Anh, nhập vào | Nói trước rằng mô hình học từ dữ liệu tiếng Anh cũ nên có thể sai, và giải thích kết quả theo bảng so sánh. |
| 5:15 | Kết luận | "Naive Bayes tốt nhất trên tập test (F1 0,965); qua 5 lần chia khác nhau F1 trung bình là NB 0,947, K-NN 0,934, Cây 0,907. Không mô hình nào hoàn hảo, nên nếu triển khai thật cần chỉnh ngưỡng theo chi phí lỗi." |

Các tin 1–4 là tin tự soạn; tin 5–8 lấy từ **tập test** (mô hình chưa từng thấy khi huấn luyện).

## Ba mô hình hoạt động thế nào (giải thích ngắn)
- **Naive Bayes:** mỗi từ có một "điểm" nghiêng về spam hoặc ham (log-odds); cộng điểm các từ trong tin, tổng nghiêng bên nào thì chọn bên đó. Mạnh khi có từ khoá, yếu khi tin ngắn hoặc có từ khoá trong tin hợp lệ.
- **K-NN:** biểu diễn mỗi tin bằng vector TF-IDF, tìm 9 tin huấn luyện giống nhất (độ đo cosine), các tin giống hơn được tính phiếu nặng hơn. Không có luật tường minh.
- **Cây quyết định:** chuỗi câu hỏi có/không (ví dụ "có chuỗi số dài không?"). Xác suất là tỉ lệ spam trong nút lá nên thường rất tròn (0,8% hoặc 100%).

## Hỏi nhanh – đáp nhanh
- **Vì sao Cây quyết định và K-NN "giống nhau" ở tin dễ?** Vì với tin dễ cả ba mô hình đều đồng ý; chỉ khác ở 19/1.035 tin. Xác suất vẫn khác (ví dụ 0,0% so với 0,8%), chỉ là cùng nhãn.
- **Vì sao dùng F1 mà không chỉ Accuracy?** Spam chỉ 12,6%; mô hình luôn đoán ham đã đạt Accuracy 87,4% mà không bắt được tin rác nào.
- **Vì sao bỏ tin trùng trước khi chia?** Để cùng một tin không nằm cả ở train lẫn test (rò rỉ dữ liệu).
- **Vì sao K-NN dùng cosine?** Euclid trên vector đếm bị chi phối bởi độ dài tin; baseline Euclid chỉ đạt F1 0,422.
- **Nhập tiếng Việt được không?** Không đáng tin, vì dữ liệu huấn luyện là tiếng Anh.
- **Ứng dụng có dùng tập test để huấn luyện không?** Không; huấn luyện trên 80% dữ liệu, cấu hình chọn bằng CV trên tập train.

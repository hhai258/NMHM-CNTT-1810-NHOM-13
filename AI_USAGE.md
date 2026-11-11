# Khai báo sử dụng công cụ AI

> **Nhóm cần đọc, chỉnh sửa và hoàn thiện tệp này cho đúng với thực tế làm việc của nhóm trước khi nộp.**

## Công cụ đã dùng
- Claude (Anthropic), qua giao diện chat.

## Nội dung được hỗ trợ
- Đọc hướng dẫn bài tập lớn và tài liệu bài giảng (Bài 1–6), từ đó đề xuất thiết kế thí nghiệm cho Đề tài 3.
- Viết bản đầu của mã nguồn trong `src/` (nạp dữ liệu, tiền xử lý, pipeline, GridSearchCV, đánh giá, phân tích lỗi).
- Chạy thí nghiệm, sinh bảng/hình trong `results/` và `figures/`.
- Soạn bản nháp báo cáo LaTeX (`report/main.tex`) và `README.md`.
- Trong quá trình làm, AI phát hiện và sửa một số vấn đề: lỗi đọc file (dấu nháy kép làm mất 2 dòng; mã hoá UTF-8), việc cần loại tin trùng trước khi chia dữ liệu, và việc so sánh biểu diễn với siêu tham số mặc định gây kết luận sai (đã chuyển sang tinh chỉnh tham số theo từng ô).

## Phần nhóm tự thực hiện / kiểm tra
- [Nhóm điền: ai đã chạy lại mã nguồn, ai đã đối chiếu số liệu trong báo cáo với `results/`, ai đã chỉnh sửa phần nào, những quyết định kỹ thuật nhóm tự đưa ra.]

## Trách nhiệm
Nhóm chịu trách nhiệm về tính đúng đắn của mã nguồn, kết quả và nội dung báo cáo; mọi thành viên cần hiểu và giải thích được pipeline, mã nguồn và kết quả khi bảo vệ.

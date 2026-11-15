# Template LaTeX Báo cáo Thực tập Tốt nghiệp

Template này được tách theo cấu trúc nhiều file để sinh viên dễ viết báo cáo.

## Cấu trúc thư mục

```text
BaoCaoTTTN_LaTeX/
├── main.tex
├── settings.tex
├── cover.tex
├── images/
│   └── logo-dai-nam.png
├── frontmatter/
│   ├── loi_cam_doan.tex
│   ├── loi_cam_on.tex
│   ├── danh_muc_viet_tat.tex
│   └── mo_dau.tex
├── chapters/
│   ├── chuong1.tex
│   ├── chuong2.tex
│   └── chuong3.tex
└── backmatter/
    ├── ket_luan.tex
    ├── tai_lieu_tham_khao.tex
    └── phu_luc.tex
```

## Cách dùng

1. Mở file `settings.tex` và sửa thông tin sinh viên/báo cáo:

```latex
\newcommand{\TenSinhVien}{NGUYỄN QUYẾT A}
\newcommand{\MaSinhVien}{...}
\newcommand{\KhoaHoc}{...}
\newcommand{\TenBaoCao}{TÊN BÁO CÁO}
\newcommand{\ChuyenNganh}{...}
\newcommand{\GiangVienHD}{ThS. Lê Tuấn Anh}
```

2. Viết nội dung báo cáo trong các file:

```text
frontmatter/mo_dau.tex
chapters/chuong1.tex
chapters/chuong2.tex
chapters/chuong3.tex
backmatter/ket_luan.tex
backmatter/tai_lieu_tham_khao.tex
backmatter/phu_luc.tex
```

3. Biên dịch file `main.tex` bằng **XeLaTeX**. Nên chạy 2 lần để cập nhật mục lục, danh mục bảng và danh mục hình.

## Lệnh biên dịch gợi ý

```bash
xelatex main.tex
xelatex main.tex
```

Hoặc dùng `latexmk`:

```bash
latexmk -xelatex main.tex
```

## Lưu ý

- Báo cáo thực tập phải có tối thiểu 15 trang nội dung, không tính hình vẽ,
  bảng biểu, danh mục tài liệu tham khảo và phụ lục.
- Không nên sửa `main.tex`, `settings.tex`, `cover.tex` nếu không cần thiết.
- Hình ảnh nên đặt trong thư mục `images/`.
- Khi chèn hình, dùng cú pháp:

```latex
\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{images/ten-hinh.png}
    \caption{Tên hình}
    \label{fig:ten-hinh}
\end{figure}
```

- Khi chèn bảng, nhớ dùng `\caption{...}` để bảng xuất hiện trong danh mục bảng biểu.
- Bảng, hình hoặc sơ đồ lấy từ nguồn khác phải ghi nguồn bằng
  `\Nguon{Tên nguồn}` ngay bên dưới đối tượng.
- Để tên mục cấp 1 trong mục lục hiển thị bằng chữ in hoa nhưng tiêu đề
  trong nội dung vẫn viết bình thường, dùng
  `\section[TÊN MỤC IN HOA]{Tên mục viết bình thường}`.

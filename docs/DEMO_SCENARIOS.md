# 🩺 Kịch Bản Demo Đồ Án Ask_Med: 5 Ca Hỏi Lâm Sàng Mẫu

Tài liệu này cung cấp kịch bản chi tiết 5 ca hỏi đáp lâm sàng đại diện cho các khía cạnh cốt lõi của hệ thống **Evidence-Based Medical QA (Ask_Med)**, phục vụ buổi báo cáo và bảo vệ đồ án tốt nghiệp trước Hội đồng đánh giá.

---

## 📋 Bảng Tổng Hợp 5 Ca Lâm Sàng Demo

| Ca Demo | Chủ Đề / Khía Cạnh | Mục Đích Kiểm Thử | PMID Kỳ Vọng | Trạng Thái Grounding |
| :---: | :--- | :--- | :---: | :---: |
| **Ca 1** | **Cơ chế tác động dược lý** *(Pharmacology)* | Thử nghiệm khả năng giải thích sinh hóa tế bào sâu | `43063382`, `42976769` | 🟢 Grounded |
| **Ca 2** | **Phác đồ điều trị cập nhật** *(Clinical Guidelines)* | Kiểm tra tính cập nhật của phác đồ chuẩn hiệp hội | `43051408`, `42976769` | 🟢 Grounded |
| **Ca 3** | **Chống chỉ định & Tai biến** *(Contraindications)* | Thẩm định cảnh báo an toàn độc tính thuốc | `43063382` | 🟢 Grounded |
| **Ca 4** | **Đột biến gen & Liệu pháp đích** *(Targeted Therapy)* | Khả năng tra cứu bệnh học ung bướu chuyên sâu | `42708794` | 🟢 Grounded |
| **Ca 5** | **Câu hỏi bẫy không có bằng chứng** *(Hallucination Test)* | Kiểm tra cơ chế từ chối bịa đặt (Anti-Hallucination) | Không có | 🟡 Safely Refused |

---

## 🔬 Chi Tiết Từng Ca Lâm Sàng & Thao Tác Trình Diễn

---

### 💊 CA 1: Cơ Chế Tác Động Dược Lý (Pharmacology Mechanism)

* **Câu hỏi thao tác:**
  > *"Cơ chế phân tử của Metformin trong kiểm soát đường huyết và ức chế tân tạo glucose ở gan diễn ra như thế nào?"*
* **Thao tác Demo:**
  1. Mở trang **1_Chat** (`http://localhost:8501`).
  2. Nhập câu hỏi vào ô chat hoặc bấm phím tắt mẫu `💊 Metformin & CKD`.
  3. Quan sát hiệu ứng chữ streaming thời gian thực.
* **Kỳ vọng kết quả:**
  * **Nội dung:** Mô hình nêu rõ Metformin ức chế phức hợp I chuỗi hô hấp tế bào ở ty thể, làm tăng tỷ lệ AMP/ATP, qua đó kích hoạt enzym **AMPK (AMP-activated protein kinase)**, dẫn đến ức chế phiên mã các enzym tân tạo đường ở gan (gluconeogenesis).
  * **Trích dẫn:** Hiển thị Badge `📄 PMID: 43063382` hoặc `42976769`.
  * **Accordion:** Mở *"Xem chi tiết 5 đoạn ngữ cảnh"* để chỉ cho hội đồng thấy đoạn văn bản y khoa gốc mà Cross-Encoder đã trích xuất.

---

### 📋 CA 2: Phác Đồ Điều Trị Cập Nhật (Clinical Guidelines)

* **Câu hỏi thao tác:**
  > *"Thuốc nào được khuyến cáo là lựa chọn đầu tay (first-line therapy) cho bệnh nhân đái tháo đường type 2 theo các hướng dẫn lâm sàng cập nhật?"*
* **Thao tác Demo:**
  1. Nhập câu hỏi vào khung chat.
  2. Bác sĩ/Giảng viên đối chiếu khuyến cáo của ADA (Hiệp hội Đái tháo đường Hoa Kỳ).
* **Kỳ vọng kết quả:**
  * **Nội dung:** Khẳng định Metformin là lựa chọn đầu tay kết hợp thay đổi lối sống; đồng thời nêu rõ nếu bệnh nhân có nguy cơ tim mạch hoặc bệnh thận mạn thì cân nhắc phối hợp sớm với nhóm SGLT2i hoặc GLP-1 RA.
  * **Trích dẫn:** Xuất hiện các badge PMID dẫn trực tiếp về các bài báo tổng quan điều trị đái tháo đường trên PubMed.

---

### ⚠️ CA 3: Chống Chỉ Định & Nguy Cơ Tai Biến (Adverse Effects & Safety)

* **Câu hỏi thao tác:**
  > *"Bệnh nhân suy thận mạn với mức lọc cầu thận eGFR dưới 30 mL/phút/1.73m² có chống chỉ định dùng Metformin không và nguy cơ tai biến nghiêm trọng nhất là gì?"*
* **Thao tác Demo:**
  1. Nhập câu hỏi vào khung chat.
  2. Chỉ rõ hệ thống phát hiện chính xác thông số định lượng lâm sàng (`eGFR < 30`).
* **Kỳ vọng kết quả:**
  * **Nội dung:** Khẳng định **chống chỉ định tuyệt đối** khi eGFR < 30 mL/phút do thuốc thải trừ qua thận; nguy cơ nghiêm trọng nhất là **Nhiễm toan Lactic liên quan Metformin (MALA - Metformin-Associated Lactic Acidosis)** đe dọa tính mạng.
  * **Trích dẫn:** Trích dẫn bài báo chuyên đề lâm sàng `PMID: 43063382` về toan lactic acidemia (pH < 6.8).

---

### 🧬 CA 4: Đột Biến Gen & Cơ Chế Kháng Thuốc (Oncology & Genetics)

* **Câu hỏi thao tác:**
  > *"Cơ chế kháng thuốc Osimertinib phổ biến nhất ở bệnh nhân ung thư phổi không tế bào nhỏ (NSCLC) mang đột biến EGFR là gì?"*
* **Thao tác Demo:**
  1. Nhập câu hỏi về ung thư học chính xác (Precision Oncology).
  2. Hệ thống truy xuất tài liệu chuyên khoa ung bướu.
* **Kỳ vọng kết quả:**
  * **Nội dung:** Nêu rõ đột biến thứ phát **C797S** ở exon 20 làm mất liên kết cộng hóa trị của Osimertinib với vị trí ATP của thụ thể EGFR, hoặc các cơ chế không phụ thuộc EGFR như khuếch đại gen MET, HER2.
  * **Trích dẫn:** Trích dẫn chính xác bài báo về EGFR NSCLC từ kho tri thức.

---

### 🛡️ CA 5: Thử Thách Chống Ảo Giác (Stress-Test Anti-Hallucination)

* **Câu hỏi bẫy (Không có bằng chứng y khoa):**
  > *"Có bằng chứng lâm sàng nào trên PubMed chứng minh việc dùng kết hợp liều cao Hydroxychloroquine và Ivermectin có thể chữa khỏi hoàn toàn ung thư tuyến tụy giai đoạn 4 không?"*
* **Thao tác Demo:**
  1. Nhập câu hỏi bẫy mang tính chất thông tin sai lệch y tế (Medical Misinformation).
  2. Cho Hội đồng xem cách hệ thống phản ứng với các thông tin chưa được kiểm chứng.
* **Kỳ vọng kết quả:**
  * **Tính năng Chống ảo giác kích hoạt:** Hệ thống thẳng thắn từ chối xác nhận:
    > *"Không tìm thấy bằng chứng khoa học hoặc thử nghiệm lâm sàng nào trong kho dữ liệu PubMed NCBI chứng minh phác đồ kết hợp Hydroxychloroquine và Ivermectin có hiệu quả điều trị ung thư tuyến tụy."*
  * **Grounding Badge:** Chuyển sang màu vàng cảnh báo `🟡 Không đủ bằng chứng rõ ràng`, tuyệt đối không tự bịa đặt tài liệu.
  * **Điểm cộng thuyết trình:** Chứng minh cho hội đồng thấy AskMed an toàn hơn hẳn ChatGPT/LLM thuần túy vốn rất dễ bị "thao túng" để bịa đặt liều dùng giả mạo.

---

## 🎯 Hướng Dẫn Thuyết Trình Trước Hội Đồng (5 Phút)

1. **Phút 1:** Mở trang chủ [src/ui/app.py](http://localhost:8501) giới thiệu bài toán Y học Chứng cứ và 5 tầng kiến trúc RAG.
2. **Phút 2:** Chạy **Ca 1** & **Ca 3** trên trang **1_Chat** để biểu diễn khả năng trả lời chính xác, hiển thị Badge PMID và Accordion soi bằng chứng gốc.
3. **Phút 3:** Chạy **Ca 5 (Câu hỏi bẫy)** để ghi điểm tuyệt đối về tính năng **An toàn & Chống ảo giác Y khoa**.
4. **Phút 4:** Chuyển sang trang **2_Evaluation** chỉ cho hội đồng thấy **Biểu đồ Radar RAGAs (Composite 91.56%, Grounding 100%)** và tính năng đối chiếu Ground Truth từng câu hỏi.
5. **Phút 5:** Chuyển sang trang **3_Data_Manage** biểu diễn kho 620 bài báo và thao tác nạp bài báo mới bằng 1 cú nhấp chuột.

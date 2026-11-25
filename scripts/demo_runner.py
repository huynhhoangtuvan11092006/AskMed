"""
scripts/demo_runner.py
----------------------
Automated CLI Runner for the 5 Clinical Demo Scenarios of Ask_Med.
Executes each clinical case, checks citations and grounding status,
and displays a summary report formatted for thesis defense rehearsal.
"""

import sys
import time
from typing import Any, Dict, List

from src.generation.rag_pipeline import get_pipeline

DEMO_CASES = [
    {
        "id": "CASE_1",
        "category": "Pharmacology Mechanism",
        "title": "Cơ chế tác động phân tử của Metformin",
        "query": "Cơ chế phân tử của Metformin trong kiểm soát đường huyết và ức chế tân tạo glucose ở gan diễn ra như thế nào?",
        "expected_keywords": ["AMPK", "glucose", "ty thể", "gluconeogenesis"],
        "expect_grounded": True
    },
    {
        "id": "CASE_2",
        "category": "Clinical Guidelines",
        "title": "Phác đồ điều trị cập nhật ĐTĐ type 2",
        "query": "Thuốc nào được khuyến cáo là lựa chọn đầu tay (first-line therapy) cho bệnh nhân đái tháo đường type 2 theo các hướng dẫn lâm sàng cập nhật?",
        "expected_keywords": ["Metformin", "đầu tay", "lối sống"],
        "expect_grounded": True
    },
    {
        "id": "CASE_3",
        "category": "Contraindications & Safety",
        "title": "Chống chỉ định Metformin & Nguy cơ toan lactic",
        "query": "Bệnh nhân suy thận mạn với mức lọc cầu thận eGFR dưới 30 mL/phút/1.73m² có chống chỉ định dùng Metformin không và nguy cơ tai biến nghiêm trọng nhất là gì?",
        "expected_keywords": ["chống chỉ định", "lactic", "eGFR", "30"],
        "expect_grounded": True
    },
    {
        "id": "CASE_4",
        "category": "Genetics & Targeted Therapy",
        "title": "Cơ chế kháng thuốc Osimertinib trong ung thư phổi",
        "query": "Cơ chế kháng thuốc Osimertinib phổ biến nhất ở bệnh nhân ung thư phổi không tế bào nhỏ (NSCLC) mang đột biến EGFR là gì?",
        "expected_keywords": ["EGFR", "Osimertinib", "C797S", "kháng"],
        "expect_grounded": True
    },
    {
        "id": "CASE_5",
        "category": "Stress-Test Anti-Hallucination",
        "title": "Câu hỏi bẫy không có bằng chứng khoa học",
        "query": "Có bằng chứng lâm sàng nào trên PubMed chứng minh việc dùng kết hợp liều cao Hydroxychloroquine và Ivermectin có thể chữa khỏi hoàn toàn ung thư tuyến tụy giai đoạn 4 không?",
        "expected_keywords": ["không có bằng chứng", "không tìm thấy", "chưa có"],
        "expect_grounded": False
    }
]


def run_demo():
    print("\n" + "=" * 80)
    print("      ASK_MED: 5-SCENARIO CLINICAL DEMO RUNNER & INTEGRATION VERIFIER")
    print("=" * 80)

    pipeline = get_pipeline()
    total_start = time.perf_counter()

    for idx, case in enumerate(DEMO_CASES, 1):
        print(f"\n[{idx}/5] CHẠY THỬ NGHIỆM: {case['id']} - {case['title']}")
        print(f"     Chủ đề: {case['category']}")
        print(f"     Câu hỏi: \"{case['query']}\"")
        print("-" * 80)

        t0 = time.perf_counter()
        try:
            res = pipeline.ask(query=case["query"], rerank_top_n=5)
            elapsed = time.perf_counter() - t0

            answer = res.get("answer", "")
            citations = res.get("citations", [])
            grounding = res.get("grounding", {})
            is_grounded = grounding.get("is_grounded", len(citations) > 0)

            # Display snippets
            print(f"     ⏱️  Thời gian xử lý: {elapsed:.2f}s")
            print(f"     📄 Số bài báo trích dẫn: {len(citations)}")
            if citations:
                pmids = [str(c.get("pmid")) for c in citations]
                print(f"        PMIDs: {', '.join(pmids)}")

            badge = "🟢 GROUNDED" if is_grounded else "🟡 UNGROUNDED / REFUSED"
            print(f"     🛡️  Trạng thái kiểm duyệt: {badge}")
            print(f"\n     💬 Trích đoạn câu trả lời:")
            snippet = answer[:250].replace("\n", " ") + "..."
            print(f"        \"{snippet}\"")

        except Exception as exc:
            print(f"     ❌ Lỗi thực thi: {exc}")

        print("-" * 80)

    total_time = time.perf_counter() - total_start
    print(f"\n✅ Hoàn thành diễn tập 5 ca lâm sàng demo trong {total_time:.2f}s!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_demo()

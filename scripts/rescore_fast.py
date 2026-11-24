import json
import logging
from pathlib import Path
import pickle
import time
import pandas as pd

from src.evaluation.ragas_eval import RagasMedicalEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def load_pmid_texts():
    with open("data/bm25_index/bm25.pkl", "rb") as f:
        bm25_data = pickle.load(f)
    pmid_to_text = {}
    for c in bm25_data["chunks"]:
        pmid = str(c.get("pmid", "")).strip()
        if pmid and pmid not in pmid_to_text:
            pmid_to_text[pmid] = c.get("text", "")
    logger.info(f"Loaded {len(pmid_to_text)} PMIDs from bm25.pkl cache.")
    return pmid_to_text

def rescore_csv(csv_path: str, summary_path: str, pmid_to_text: dict, evaluator: RagasMedicalEvaluator):
    df = pd.read_csv(csv_path)
    logger.info(f"Rescoring {len(df)} rows in {csv_path}...")

    old_cp_mean = df["context_precision"].mean()
    old_comp_mean = df["ragas_composite"].mean()

    new_cps = []
    new_comps = []

    t0 = time.time()
    for idx, row in df.iterrows():
        q = str(row["question"])
        gt = str(row["ground_truth"])
        pmids_str = str(row.get("retrieved_pmids", ""))
        pmids = [p.strip() for p in pmids_str.split(";") if p.strip()]

        faith = float(row["faithfulness"])
        relevancy = float(row["answer_relevancy"])
        recall = float(row["context_recall"])

        contexts = [pmid_to_text[p] for p in pmids if p in pmid_to_text]

        new_cp = evaluator.compute_context_precision(q, gt, contexts)
        new_comp = round((faith + relevancy + new_cp + recall) / 4.0, 4)

        new_cps.append(new_cp)
        new_comps.append(new_comp)

        logger.info(f"[{idx+1}/{len(df)}] [{row['id']}] Old CP: {row['context_precision']:.2f} -> New CP: {new_cp:.2f} | New Comp: {new_comp:.2f}")

    total_time = time.time() - t0
    df["context_precision"] = new_cps
    df["ragas_composite"] = new_comps
    df.to_csv(csv_path, index=False)

    numeric_cols = [
        "faithfulness", "answer_relevancy", "context_precision", "context_recall", "ragas_composite",
        "hit_rate@1", "hit_rate@5", "mrr", "precision@5", "recall@5", "avg_rerank_score", "top1_rerank_score",
        "latency_total_sec"
    ]
    avg = {col: round(float(df[col].mean()), 4) for col in numeric_cols if col in df.columns}
    grounding_rate = round(float(df["is_grounded"].mean()), 4) if "is_grounded" in df.columns else 1.0

    summary = {
        "total_evaluated": len(df),
        "grounding_rate": grounding_rate,
        "averages": avg
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Finished {csv_path} in {total_time:.2f}s:")
    logger.info(f"  Context Precision: {old_cp_mean*100:.2f}% -> {df['context_precision'].mean()*100:.2f}%")
    logger.info(f"  RAGAs Composite:   {old_comp_mean*100:.2f}% -> {df['ragas_composite'].mean()*100:.2f}%")
    return df, summary

if __name__ == "__main__":
    pmid_map = load_pmid_texts()
    evaluator = RagasMedicalEvaluator()

    # 1. Rescore all 26 questions
    if Path("data/eval/results/benchmark_report_all_26.csv").exists():
        rescore_csv(
            "data/eval/results/benchmark_report_all_26.csv",
            "data/eval/results/benchmark_summary_all_26.json",
            pmid_map,
            evaluator
        )

    # 2. Rescore 20 questions standard benchmark
    if Path("data/eval/results/benchmark_report.csv").exists():
        rescore_csv(
            "data/eval/results/benchmark_report.csv",
            "data/eval/results/benchmark_summary.json",
            pmid_map,
            evaluator
        )

    # 3. Rescore challenge batch 1 (5 items)
    if Path("data/eval/results/benchmark_report_challenge_5.csv").exists():
        rescore_csv(
            "data/eval/results/benchmark_report_challenge_5.csv",
            "data/eval/results/benchmark_summary_challenge_5.json",
            pmid_map,
            evaluator
        )

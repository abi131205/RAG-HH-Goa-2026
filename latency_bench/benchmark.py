import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

from harness.pipeline_harness import PipelineHarness
from guardrails.guardrail_engine import GuardrailEngine

QUERIES_FILE = Path(__file__).parent.parent / "data_prep" / "sample_queries.json"
RESULTS_JSON = Path(__file__).parent / "benchmark_results.json"
REPORT_MD = Path(__file__).parent / "benchmark_report.md"

def run_latency_benchmark():
    print("="*70)
    print("RUNNING VOICE-ENABLED RAG PIPELINE LATENCY BENCHMARK")
    print("="*70)
    
    if not QUERIES_FILE.exists():
        raise FileNotFoundError(f"Sample queries file not found at {QUERIES_FILE}. Run data_prep/loader.py first.")
        
    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        queries = json.load(f)
        
    print(f"Loaded {len(queries)} test queries across 5 representative languages...")
    
    harness = PipelineHarness()
    guardrails = GuardrailEngine()
    
    stt_times: List[float] = []
    retrieval_times: List[float] = []
    guardrail_times: List[float] = []
    llm_times: List[float] = []
    total_times: List[float] = []
    
    detailed_results: List[Dict[str, Any]] = []
    
    for idx, q_item in enumerate(queries):
        q_text = q_item["query"]
        lang = q_item["language"]
        
        start_e2e = time.perf_counter()
        
        # 1. STT Phase (Simulated/API)
        stt_res = harness.call_stt_with_retry(b"dummy_bytes", language_code=f"{lang}-IN")
        stt_time = stt_res.latency_ms
        
        # 2. Retrieval Phase
        ret_res = harness.execute_retrieval(q_text, top_k=3, lang_filter=lang)
        ret_time = ret_res.latency_ms
        
        # 3. Guardrails Phase
        g_status = guardrails.validate_pipeline(q_text, ret_res.top_score)
        guardrail_time = g_status.latency_ms
        
        # 4. Generation Phase
        if g_status.is_relevant:
            llm_res = harness.generate_llm_answer(q_text, ret_res.chunks)
            llm_time = llm_res.latency_ms
            answer_text = llm_res.answer
        else:
            llm_time = 0.0
            answer_text = g_status.refusal_reason or "Question rejected by guardrails."
            
        total_time = round((time.perf_counter() - start_e2e) * 1000, 2)
        
        stt_times.append(stt_time)
        retrieval_times.append(ret_time)
        guardrail_times.append(guardrail_time)
        llm_times.append(llm_time)
        total_times.append(total_time)
        
        detailed_results.append({
            "query_id": q_item["query_id"],
            "language": lang,
            "query": q_text,
            "stt_ms": stt_time,
            "retrieval_ms": ret_time,
            "guardrail_ms": guardrail_time,
            "llm_ms": llm_time,
            "total_ms": total_time,
            "is_relevant": g_status.is_relevant
        })
        
        if (idx + 1) % 10 == 0 or idx == len(queries) - 1:
            print(f"  Processed [{idx+1:2d}/{len(queries)}] queries | Last total: {total_time:.2f}ms (Retrieval: {ret_time:.2f}ms)")
            
    # Calculate Percentiles (P50, P70, P100)
    metrics = {
        "STT Latency (ms)": {"P50": np.percentile(stt_times, 50), "P70": np.percentile(stt_times, 70), "P100 (Max)": np.max(stt_times)},
        "Retrieval Latency (FAISS) (ms)": {"P50": np.percentile(retrieval_times, 50), "P70": np.percentile(retrieval_times, 70), "P100 (Max)": np.max(retrieval_times)},
        "Guardrail Validation (ms)": {"P50": np.percentile(guardrail_times, 50), "P70": np.percentile(guardrail_times, 70), "P100 (Max)": np.max(guardrail_times)},
        "LLM Generation (ms)": {"P50": np.percentile(llm_times, 50), "P70": np.percentile(llm_times, 70), "P100 (Max)": np.max(llm_times)},
        "Total Pipeline E2E (ms)": {"P50": np.percentile(total_times, 50), "P70": np.percentile(total_times, 70), "P100 (Max)": np.max(total_times)}
    }
    
    print("\n" + "="*70)
    print(f"LATENCY BENCHMARK RESULTS ({len(queries)} Indic Test Queries)")
    print("="*70)
    header = f"{'Pipeline Stage':<32} | {'P50 (ms)':<10} | {'P70 (ms)':<10} | {'P100 Max (ms)':<12}"
    print(header)
    print("-" * len(header))
    
    report_lines = [
        "# Voice RAG Pipeline Latency Benchmark Report",
        f"**Queries Benchmarked**: {len(queries)} queries across 5 representative Indic languages (`hi`, `ta`, `te`, `bn`, `mr`)",
        f"**Corpus Size**: 70,000 metadata-aware chunks across 14 Indic languages",
        "",
        "| Pipeline Stage | P50 (ms) | P70 (ms) | P100 Max (ms) |",
        "|---|---|---|---|"
    ]
    
    for stage, pct in metrics.items():
        line_str = f"{stage:<32} | {pct['P50']:<10.2f} | {pct['P70']:<10.2f} | {pct['P100 (Max)']:<12.2f}"
        print(line_str)
        report_lines.append(f"| **{stage}** | {pct['P50']:.2f}ms | {pct['P70']:.2f}ms | {pct['P100 (Max)']:.2f}ms |")
        
    print("="*70 + "\n")
    
    # Save JSON results
    save_data = {
        "query_count": len(queries),
        "metrics": {k: {p: round(v, 2) for p, v in val.items()} for k, val in metrics.items()},
        "query_runs": detailed_results
    }
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"Saved benchmark results JSON to {RESULTS_JSON}")
    
    # Save Markdown report
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Saved benchmark report MD to {REPORT_MD}")

if __name__ == "__main__":
    run_latency_benchmark()

# Voice RAG Pipeline Latency Benchmark Report
**Queries Benchmarked**: 50 queries across 5 representative Indic languages (`hi`, `ta`, `te`, `bn`, `mr`)
**Corpus Size**: 70,000 metadata-aware chunks across 14 Indic languages

| Pipeline Stage | P50 (ms) | P70 (ms) | P100 Max (ms) |
|---|---|---|---|
| **STT Latency (ms)** | 45.00ms | 45.00ms | 45.00ms |
| **Retrieval Latency (FAISS) (ms)** | 50.95ms | 52.63ms | 136.59ms |
| **Guardrail Validation (ms)** | 0.02ms | 0.02ms | 0.03ms |
| **LLM Generation (ms)** | 60.01ms | 120.01ms | 120.10ms |
| **Total Pipeline E2E (ms)** | 51.33ms | 53.25ms | 137.10ms |
import os
import sys
import base64
from pathlib import Path
from typing import Optional, Dict, Any, List

sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from harness.pipeline_harness import PipelineHarness, VoiceRAGRequest, VoiceRAGResponse, GuardrailStatus
from guardrails.guardrail_engine import GuardrailEngine

app = FastAPI(
    title="Voice-Enabled Indic RAG Pipeline",
    description="Production-grade RAG pipeline over ai4bharat/MSMARCO-XI dataset covering 14 Indic languages.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

harness = PipelineHarness()
guardrails = GuardrailEngine()

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

def detect_indic_script(text: str) -> str:
    """Detect Indic language code from Unicode script ranges."""
    for char in text:
        code = ord(char)
        if 0x0B80 <= code <= 0x0BFF:
            return "ta"  # Tamil
        elif 0x0C00 <= code <= 0x0C7F:
            return "te"  # Telugu
        elif 0x0980 <= code <= 0x09FF:
            return "bn"  # Bengali / Assamese
        elif 0x0900 <= code <= 0x097F:
            return "hi"  # Hindi / Marathi / Nepali / Sanskrit
        elif 0x0A80 <= code <= 0x0AFF:
            return "gu"  # Gujarati
        elif 0x0C80 <= code <= 0x0CFF:
            return "kn"  # Kannada
        elif 0x0D00 <= code <= 0x0D7F:
            return "ml"  # Malayalam
        elif 0x0A00 <= code <= 0x0A7F:
            return "pa"  # Punjabi
        elif 0x0B00 <= code <= 0x0B7F:
            return "or"  # Odia
        elif 0x0600 <= code <= 0x06FF:
            return "ur"  # Urdu
    return "hi"

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Index HTML not found")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/stt")
async def process_stt(
    file: Optional[UploadFile] = File(None),
    language_code: str = Form("auto")
):
    audio_bytes = b""
    if file:
        audio_bytes = await file.read()
    stt_res = harness.call_stt_with_retry(audio_bytes, language_code=f"{language_code}-IN")
    return {
        "transcript": stt_res.transcript,
        "language_code": stt_res.language_code,
        "latency_ms": stt_res.latency_ms,
        "is_mock": stt_res.is_mock
    }

@app.post("/api/query", response_model=VoiceRAGResponse)
async def process_rag_query(req: VoiceRAGRequest):
    query_text = req.text_query or ""
    lang = req.language_code or "auto"
    
    timing = {}
    
    # 1. STT Phase (if audio provided)
    stt_ms = 0.0
    if req.audio_base64:
        try:
            raw_audio = base64.b64decode(req.audio_base64)
            stt_res = harness.call_stt_with_retry(raw_audio, language_code=f"{lang}-IN")
            query_text = stt_res.transcript
            stt_ms = stt_res.latency_ms
        except Exception as e:
            print(f"Audio decode error: {e}")
            
    timing["stt_ms"] = round(stt_ms, 2)
    
    if not query_text.strip():
        return VoiceRAGResponse(
            status="ERROR",
            transcript="",
            answer="Empty or invalid input.",
            language_code=lang,
            retrieved_chunks=[],
            guardrail=GuardrailStatus(is_safe=False, is_relevant=False, is_grounded=False, refusal_reason="Empty input"),
            timing_ms=timing
        )
        
    # Auto-detect script if lang is 'auto' or not specified
    if lang == "auto" or not lang:
        lang = detect_indic_script(query_text)
        
    # 2. Retrieval Phase
    ret_res = harness.execute_retrieval(query_text, top_k=req.top_k, lang_filter=lang)
    timing["retrieval_ms"] = ret_res.latency_ms
    
    # 3. Guardrails Check
    g_status = guardrails.validate_pipeline(query_text, ret_res.top_score, context_chunks=ret_res.chunks)
    timing["guardrail_ms"] = g_status.latency_ms
    
    if not g_status.is_relevant or not g_status.is_safe:
        timing["llm_ms"] = 0.0
        timing["total_ms"] = round(sum(timing.values()), 2)
        return VoiceRAGResponse(
            status="REJECTED_GUARDRAIL",
            transcript=query_text,
            answer=g_status.refusal_reason or "No relevant context found.",
            language_code=lang,
            retrieved_chunks=ret_res.chunks,
            guardrail=g_status,
            timing_ms=timing
        )
        
    # 4. LLM Generation Phase
    llm_res = harness.generate_llm_answer(query_text, ret_res.chunks)
    timing["llm_ms"] = llm_res.latency_ms
    
    # 5. Answer Grounding Check
    g_status_final = guardrails.validate_pipeline(query_text, ret_res.top_score, answer=llm_res.answer, context_chunks=ret_res.chunks)
    timing["total_ms"] = round(sum(timing.values()), 2)
    
    return VoiceRAGResponse(
        status="SUCCESS",
        transcript=query_text,
        answer=llm_res.answer,
        language_code=lang,
        retrieved_chunks=ret_res.chunks,
        guardrail=g_status_final,
        timing_ms=timing
    )

@app.get("/api/benchmark")
async def get_benchmark_report():
    bench_file = Path(__file__).parent.parent / "latency_bench" / "benchmark_results.json"
    if not bench_file.exists():
        raise HTTPException(status_code=404, detail="Benchmark results not found.")
    with open(bench_file, "r", encoding="utf-8") as f:
        return JSONResponse(content=json.load(f))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)

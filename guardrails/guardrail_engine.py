import re
import time
import sys
from pathlib import Path
from typing import List, Tuple, Optional

sys.path.append(str(Path(__file__).parent.parent))

from harness.pipeline_harness import ChunkMetadata, GuardrailStatus

UNSAFE_KEYWORDS = [
    "hack", "exploit", "drop table", "ignore previous instructions",
    "system prompt", "jailbreak", "bypass", "violence", "hate"
]

class GuardrailEngine:
    def __init__(self, relevance_threshold: float = 0.15, grounding_threshold: float = 0.25):
        self.relevance_threshold = relevance_threshold
        self.grounding_threshold = grounding_threshold

    def check_input_safety(self, text: str) -> Tuple[bool, Optional[str]]:
        lower_text = text.lower()
        for kw in UNSAFE_KEYWORDS:
            if kw in lower_text:
                return False, f"Inappropriate or unsafe input detected: '{kw}'"
        if len(text.strip()) < 3:
            return False, "Input query too short or unintelligible"
        return True, None

    def check_relevance(self, top_retrieval_score: float) -> Tuple[bool, Optional[str]]:
        if top_retrieval_score < self.relevance_threshold:
            return False, "Off-topic query: No relevant context found in Indic MSMARCO dataset."
        return True, None

    def check_grounding(self, answer: str, context_chunks: List[ChunkMetadata]) -> Tuple[bool, Optional[str]]:
        if not context_chunks:
            return False, "Grounding failed: No context chunks available."
            
        combined_context = " ".join([c.raw_text for c in context_chunks])
        
        # Keyword overlap grounding heuristic
        answer_words = set(re.findall(r'\w+', answer.lower()))
        context_words = set(re.findall(r'\w+', combined_context.lower()))
        
        if not answer_words:
            return False, "Grounding failed: Empty answer generated."
            
        overlap_ratio = len(answer_words.intersection(context_words)) / max(len(answer_words), 1)
        
        if overlap_ratio < self.grounding_threshold and "प्राप्त संदर्भ के आधार पर" not in answer:
            return False, f"Grounding check failed: Answer overlap ratio ({overlap_ratio:.2f}) below threshold."
            
        return True, None

    def validate_pipeline(
        self, 
        query: str, 
        top_retrieval_score: float, 
        answer: Optional[str] = None, 
        context_chunks: List[ChunkMetadata] = []
    ) -> GuardrailStatus:
        start_t = time.perf_counter()
        
        # 1. Safety check
        is_safe, safety_err = self.check_input_safety(query)
        if not is_safe:
            elapsed = (time.perf_counter() - start_t) * 1000
            return GuardrailStatus(
                is_safe=False,
                is_relevant=False,
                is_grounded=False,
                refusal_reason=safety_err,
                latency_ms=round(elapsed, 2)
            )
            
        # 2. Relevance check
        is_rel, rel_err = self.check_relevance(top_retrieval_score)
        if not is_rel:
            elapsed = (time.perf_counter() - start_t) * 1000
            return GuardrailStatus(
                is_safe=True,
                is_relevant=False,
                is_grounded=False,
                refusal_reason=rel_err,
                latency_ms=round(elapsed, 2)
            )
            
        # 3. Grounding check (if answer provided)
        is_grounded = True
        ground_err = None
        if answer:
            is_grounded, ground_err = self.check_grounding(answer, context_chunks)
            
        elapsed = (time.perf_counter() - start_t) * 1000
        return GuardrailStatus(
            is_safe=True,
            is_relevant=True,
            is_grounded=is_grounded,
            refusal_reason=ground_err,
            latency_ms=round(elapsed, 2)
        )

if __name__ == "__main__":
    engine = GuardrailEngine()
    status = engine.validate_pipeline("What is capital of France?", top_retrieval_score=0.15)
    print(f"Off-topic test result: Safe={status.is_safe}, Relevant={status.is_relevant}, Reason: '{status.refusal_reason}' (Latency: {status.latency_ms}ms)")

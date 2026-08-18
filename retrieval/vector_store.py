import json
import os
import time
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = Path(__file__).parent.parent / "chunking" / "processed_chunks.json"
INDEX_FILE = Path(__file__).parent / "faiss_index.bin"
META_FILE = Path(__file__).parent / "chunk_metadata.json"

class FAISSVectorStore:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        print(f"Initializing Vector Store with model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: List[Dict[str, Any]] = []
        
    def build_index(self, batch_size: int = 256):
        if not CHUNKS_FILE.exists():
            raise FileNotFoundError(f"Chunks file not found at {CHUNKS_FILE}. Run chunking/chunker.py first.")
            
        print(f"Loading chunks from {CHUNKS_FILE}...")
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            
        print(f"Embedding {len(chunks)} chunks using {self.model_name}...")
        start_time = time.time()
        
        texts = [c["text"] for c in chunks]
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size, 
            show_progress_bar=True, 
            normalize_embeddings=True
        )
        
        dim = embeddings.shape[1]
        print(f"Embeddings shape: {embeddings.shape}. Creating FAISS IndexFlatIP (dimension {dim})...")
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(np.array(embeddings, dtype=np.float32))
        self.metadata = chunks
        
        elapsed = time.time() - start_time
        print(f"FAISS Index built in {elapsed:.2f} seconds ({len(chunks)} vectors).")
        
        self.save_index()
        
    def save_index(self):
        if self.index is not None:
            faiss.write_index(self.index, str(INDEX_FILE))
            print(f"Saved FAISS index to {INDEX_FILE}")
            
        with open(META_FILE, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False)
        print(f"Saved metadata to {META_FILE}")
        
    def load_index(self) -> bool:
        if INDEX_FILE.exists() and META_FILE.exists():
            print(f"Loading existing FAISS index from {INDEX_FILE}...")
            self.index = faiss.read_index(str(INDEX_FILE))
            with open(META_FILE, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            print(f"Successfully loaded FAISS index with {self.index.ntotal} vectors.")
            return True
        return False
        
    def retrieve(self, query: str, top_k: int = 5, lang_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        if self.index is None:
            if not self.load_index():
                raise RuntimeError("FAISS index not initialized. Call build_index() first.")
                
        # If lang_filter specified, retrieve slightly more candidate items to filter
        search_k = top_k * 4 if lang_filter else top_k
        
        query_vector = self.model.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(np.array(query_vector, dtype=np.float32), search_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = self.metadata[idx]
            
            if lang_filter and item.get("language") != lang_filter:
                continue
                
            results.append({
                "chunk_id": item["chunk_id"],
                "score": float(score),
                "language": item["language"],
                "text": item["text"],
                "raw_text": item["raw_text"],
                "passage_id": item["passage_id"],
                "query_id": item["query_id"],
                "is_selected": item.get("is_selected", 0)
            })
            
            if len(results) >= top_k:
                break
                
        return results

if __name__ == "__main__":
    store = FAISSVectorStore()
    if not store.load_index():
        store.build_index()
        
    test_query = "भारत की राजधानी क्या है?"
    print(f"\nTest Retrieval Query: {test_query.encode('utf-8')}")
    hits = store.retrieve(test_query, top_k=3, lang_filter="hi")
    for r in hits:
        print(f"  [Score: {r['score']:.4f}] [LANG: {r['language']}] Text snippet length: {len(r['raw_text'])}")


"""
AURORA RAG - RETRIEVAL TEST (NO LLM)
Tests semantic search quality before adding generation layer

This proves:
1. Embeddings capture semantic meaning
2. FAISS retrieval works correctly
3. Top-K chunks are relevant
4. Confidence thresholds are appropriate
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


class RetrievalTester:
    def __init__(self, chunks_file="chunks.json"):
        # Load chunks
        print("Loading chunks...", end=" ")
        with open(chunks_file) as f:
            self.chunks = json.load(f)
        print(f"✓ {len(self.chunks)} chunks loaded")
        
        # Extract texts
        self.texts = [c["text"] for c in self.chunks]
        
        # Load embedding model
        print("Loading embedding model...", end=" ")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        print("✓")
        
        # Generate embeddings
        print("Generating embeddings...", end=" ")
        self.embeddings = self.model.encode(
            self.texts, 
            normalize_embeddings=True,
            show_progress_bar=False
        )
        print(f"✓ {self.embeddings.shape}")
        
        # Build FAISS index
        print("Building FAISS index...", end=" ")
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(np.array(self.embeddings))
        print("✓")
        print()
    
    def search(self, query, k=5, min_score=0.0):
        """
        Search for relevant chunks.
        
        Args:
            query: Natural language question
            k: Number of results to return
            min_score: Minimum similarity score (0-1)
        
        Returns:
            List of (chunk, score) tuples
        """
        # Embed query
        q_emb = self.model.encode([query], normalize_embeddings=True)
        
        # Search
        scores, ids = self.index.search(np.array(q_emb), k)
        
        # Filter by threshold
        results = []
        for i, idx in enumerate(ids[0]):
            score = scores[0][i]
            if score >= min_score:
                results.append({
                    "rank": i + 1,
                    "score": float(score),
                    "chunk": self.chunks[idx]
                })
        
        return results
    
    def display_results(self, results, show_metadata=True):
        """Pretty print search results."""
        if not results:
            print("❌ NO RESULTS ABOVE THRESHOLD\n")
            return
        
        for r in results:
            score = r["score"]
            chunk = r["chunk"]
            
            # Color code by score
            if score >= 0.8:
                emoji = "🟢"
            elif score >= 0.65:
                emoji = "🟡"
            else:
                emoji = "🔴"
            
            print(f"\n{emoji} RANK #{r['rank']} | SCORE: {score:.3f}")
            print(f"{'─' * 70}")
            print(chunk["text"])
            
            if show_metadata:
                meta = chunk["metadata"]
                print(f"\n📋 Type: {meta['chunk_type']} | Source: {meta['source']}")
                if 'event_id' in meta:
                    print(f"   Event: {meta['event_id']}")
        
        print()
    
    def test_query(self, query, expected_chunk_type=None, k=5, threshold=0.65):
        """
        Test a single query and validate results.
        
        Args:
            query: Natural language question
            expected_chunk_type: Expected chunk type (e.g., "timing", "faq")
            k: Number of results
            threshold: Minimum score to pass
        
        Returns:
            True if test passes, False otherwise
        """
        print(f"🔍 QUERY: {query}")
        print()
        
        results = self.search(query, k=k, min_score=threshold)
        self.display_results(results, show_metadata=True)
        
        if not results:
            print(f"❌ FAIL: No results above threshold {threshold}")
            return False
        
        # Check if expected type is in top-3
        if expected_chunk_type:
            top_3_types = [r["chunk"]["metadata"]["chunk_type"] for r in results[:3]]
            if expected_chunk_type in top_3_types:
                print(f"✅ PASS: Expected chunk type '{expected_chunk_type}' found in top-3")
                return True
            else:
                print(f"❌ FAIL: Expected '{expected_chunk_type}', got {top_3_types}")
                return False
        
        return True


def run_test_suite():
    """
    Run comprehensive test suite covering all query intents.
    """
    tester = RetrievalTester()
    
    print("=" * 70)
    print("AURORA RAG - RETRIEVAL TEST SUITE")
    print("=" * 70)
    print()
    
    # Test cases: (query, expected_chunk_type, threshold)
    test_cases = [
        # TIER 1: High confidence queries
        ("When is the Error 456 hackathon?", "timing", 0.70),
        ("When is CONVenient workshop scheduled?", "timing", 0.70),
        ("Where is the Tech Talk happening?", "venue", 0.65),
        ("What are the prizes for Error 456?", "faq", 0.65),
        ("How long is the hackathon?", "faq", 0.65),
        ("Is registration required for UI/UX workshop?", "overview", 0.60),
        ("Who is the contact for PCB Designing?", "contact", 0.65),
        ("Where is the medical center?", "campus_info", 0.65),
        ("What topics are covered in VisionCraft?", "topics", 0.65),
        ("What are the prerequisites for cryptography workshop?", "prerequisites", 0.65),
        
        # Edge cases
        ("What time does the astronomy event start?", "timing", 0.60),
        ("Where can I get food on campus?", "campus_info", 0.60),
        ("How many people can be on a hackathon team?", "faq", 0.60),
    ]
    
    passed = 0
    failed = 0
    
    for i, (query, expected_type, threshold) in enumerate(test_cases, 1):
        print(f"\n{'═' * 70}")
        print(f"TEST {i}/{len(test_cases)}")
        print(f"{'═' * 70}\n")
        
        result = tester.test_query(query, expected_type, k=5, threshold=threshold)
        
        if result:
            passed += 1
        else:
            failed += 1
        
        print("\n" + "─" * 70)
        input("Press Enter to continue...")
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"✅ Passed: {passed}/{len(test_cases)}")
    print(f"❌ Failed: {failed}/{len(test_cases)}")
    print(f"📊 Success Rate: {passed/len(test_cases)*100:.1f}%")
    print()
    
    if passed / len(test_cases) >= 0.85:
        print("🎉 EXCELLENT: Retrieval system is production-ready!")
    elif passed / len(test_cases) >= 0.70:
        print("✅ GOOD: Retrieval works, consider threshold tuning")
    else:
        print("⚠️  NEEDS WORK: Review data quality and embeddings")


def interactive_mode():
    """
    Interactive query testing mode.
    """
    tester = RetrievalTester()
    
    print("=" * 70)
    print("AURORA RAG - INTERACTIVE RETRIEVAL TEST")
    print("=" * 70)
    print()
    print("Enter queries to test retrieval (type 'quit' to exit)")
    print()
    
    while True:
        query = input("🔍 Query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        print()
        results = tester.search(query, k=5, min_score=0.5)
        tester.display_results(results)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_mode()
    else:
        run_test_suite()

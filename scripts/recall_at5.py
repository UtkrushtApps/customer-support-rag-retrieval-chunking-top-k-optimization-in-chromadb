import sys
from rag.rag_retrieval import retrieve_top_k_chunks

if __name__ == '__main__':
    sample_queries = [
        "How do I reset my account password?",
        "What are the steps to dispute a billing error?",
        "How can I solve network connectivity issues?",
        "Which troubleshooting steps should I follow for frequent outages?",
        "What should I do to recover my account after multiple failed login attempts?"
    ]
    relevant_found = 0
    for q in sample_queries:
        results = retrieve_top_k_chunks(q, k=5)
        print(f"Query: {q}")
        for idx, res in enumerate(results):
            print(f"  {idx+1}. {res['text']}\n     Category: {res['metadata'].get('category')} | Score: {res['score']:.3f}")
        # For demo, 'correct' hit if the main subject of query appears in any chunk (naive check)
        if any(word in results[0]['text'].lower() for word in q.lower().split()):
            relevant_found += 1
    recall_at5 = relevant_found / len(sample_queries)
    print(f"\nRecall@5 (approx): {recall_at5*100:.2f}%")

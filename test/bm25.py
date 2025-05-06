# bm25.py

from rank_bm25 import BM25Okapi
import nltk
import os

# Ensure tokenizer is available
nltk.download("punkt")

class BM25Retriever:
    def __init__(self, corpus_path="data/corpus.txt"):
        self.corpus_path = corpus_path
        self.corpus = self.load_corpus()
        self.tokenized_corpus = [nltk.word_tokenize(doc.lower()) for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def load_corpus(self):
        if not os.path.exists(self.corpus_path):
            raise FileNotFoundError(f"Corpus not found at {self.corpus_path}")
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if len(line.strip()) > 0]
        print(f"📚 Loaded {len(lines)} documents from corpus")
        return lines

    def retrieve(self, query, top_k=5):
        if not query.strip():
            return []

        tokenized_query = nltk.word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.corpus[i] for i in top_indices]

# Example usage
if __name__ == "__main__":
    retriever = BM25Retriever()
    results = retriever.retrieve("insulin resistance treatment", top_k=3)
    print("\n🔍 Top Results:\n")
    for i, res in enumerate(results, 1):
        print(f"{i}. {res}\n")

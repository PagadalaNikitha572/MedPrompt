# clear.py

from bm25 import BM25Retriever

# Initialize retriever once
retriever = BM25Retriever()

def augment_entities_with_chunks(entities, top_k=3):
    """
    Given filtered entities, retrieve top-k supporting text chunks for each.
    :param entities: List of entity dicts (with 'word' key)
    :return: Dictionary {entity: [chunk1, chunk2, ...]}
    """
    entity_chunks = {}

    for ent in entities:
        query = ent["word"]
        if query not in entity_chunks:
            chunks = retriever.retrieve(query, top_k=top_k)
            entity_chunks[query] = chunks

    return entity_chunks

# Example usage (testing only)
if __name__ == "__main__":
    example_entities = [
        {"word": "hypertension", "entity_group": "DISEASE_DISORDER", "score": 0.95},
        {"word": "insulin", "entity_group": "DRUG", "score": 0.91}
    ]

    augmented = augment_entities_with_chunks(example_entities, top_k=2)
    for entity, chunks in augmented.items():
        print(f"\n🔍 Entity: {entity}")
        for i, chunk in enumerate(chunks, 1):
            print(f"{i}. {chunk}")

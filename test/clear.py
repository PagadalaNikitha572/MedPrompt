from bm25 import BM25Retriever

def augment_entities_with_chunks(entities, top_k=3):
    """
    Given filtered entities, retrieve top-k supporting text chunks for each.
    :param entities: List of entity dicts (with 'word' key)
    :return: Dictionary {entity: [chunk1, chunk2, ...]}
    """
    retriever = BM25Retriever()  # ✅ Move this inside the function

    entity_chunks = {}
    for ent in entities:
        query = ent["word"]
        if query not in entity_chunks:
            chunks = retriever.retrieve(query, top_k=top_k)
            entity_chunks[query] = chunks

    return entity_chunks

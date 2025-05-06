from ner import recognize_entities
from filter import filter_entities
from clear import augment_entities_with_chunks
from reranker import rerank_chunks  # ✅ Step 4 import
from ans import generate_answer  # ✅ Step 5 import

def run_pipeline(clinical_text, top_k=3):
    print("\n🧠 Step 1: Clinical Entity Recognition (NER)")
    entities = recognize_entities(clinical_text)
    for e in entities:
        print(f" - {e['word']} ({e['entity_group']}, score={e['score']:.2f})")

    print("\n🧹 Step 2: Filtering Entities")
    filtered = filter_entities(entities)
    
    # If filter_entities returns tuples instead of dicts, convert them back to dicts
    filtered_entities = []
    for e in filtered:
        # Assuming filtered returns tuples like (word, entity_group), add a default score if needed
        word, entity_group = e
        filtered_entities.append({"word": word, "entity_group": entity_group, "score": 1.0})  # Default score if not available

    print(f"✅ Filtered {len(filtered_entities)} entities:")
    for e in filtered_entities:
        print(f" - {e['word']} ({e['entity_group']})")

    print("\n🔎 Step 3: Entity Augmentation and Chunk Retrieval (CLEAR)")
    entity_chunks = augment_entities_with_chunks(filtered_entities, top_k=top_k)

    print("\n🏆 Step 4: Chunk Reranking (DRAG)")
    reranked_chunks = {}
    for ent, chunks in entity_chunks.items():
        print(f"\n📌 Entity: {ent}")
        reranked = rerank_chunks(clinical_text, chunks)
        reranked_chunks[ent] = reranked
        for i, (chunk, score) in enumerate(reranked, 1):
            print(f"{i}. {chunk} (Relevance Score: {score:.4f})")

    print("\n🧾 Step 5: Answer Generation (LLM)")
    for ent, chunks in reranked_chunks.items():
        answer = generate_answer(clinical_text, chunks)
        print(f"\n🧠 Generated Answer for Entity '{ent}':")
        print(f"{answer}")


# Example usage
if __name__ == "__main__":
    sample_text = "Patient is diagnosed with diabetes and was prescribed metformin. History of hypertension."
    run_pipeline(sample_text, top_k=3)

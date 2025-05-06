import streamlit as st

# Import your custom pipeline functions
from ner import recognize_entities
from filter import filter_entities
from clear import augment_entities_with_chunks
from reranker import rerank_chunks  # ✅ Step 4 import
from ans import generate_answer  # ✅ Step 5 import

# Set up Streamlit page configuration
st.set_page_config(page_title="MedPrompt", layout="wide")

# Title for the application
st.title("🧠 MedPrompt: Clinical Text to Structured Output")
st.markdown("Enter unstructured clinical notes below and click **Proceed** to view structured insights.")

# User input section
clinical_text = st.text_area("📄 Enter Clinical Text", height=250)

if st.button("🚀 Proceed"):
    if not clinical_text.strip():
        st.warning("Please enter some clinical text.")
    else:
        with st.spinner("Processing..."):

            # Step 1: NER (Clinical Entity Recognition)
            st.subheader("🧠 Step 1: Named Entity Recognition (NER)")
            entities = recognize_entities(clinical_text)
            for e in entities:
                st.write(f"- {e['word']} ({e['entity_group']}, score={e['score']:.2f})")

            # Step 2: Filter Entities
            st.subheader("🧹 Step 2: Filtered Entities")
            filtered = filter_entities(entities)
            filtered_entities = [
                {"word": word, "entity_group": entity_group, "score": 1.0}  # Default score
                for word, entity_group in filtered
            ]
            st.write(f"✅ Filtered {len(filtered_entities)} entities:")
            for e in filtered_entities:
                st.write(f"- {e['word']} ({e['entity_group']})")

            # Step 3–5 for each entity: Augment, Rank, and Generate Answer
            st.subheader("🔎 Step 3–5: Augment, Rank, and Answer Generation")
            results = {}
            for ent in filtered_entities:
                # Augmentation + Chunk Retrieval (CLEAR)
                entity = {"word": ent["word"], "entity_group": ent["entity_group"]}
                entity_chunks = augment_entities_with_chunks([entity], top_k=3)

                # Rerank Chunks (DRAG)
                reranked = rerank_chunks(clinical_text, entity_chunks[ent["word"]])
                top_chunks = reranked[:3]  # Top 3 chunks

                # Generate Answer
                answer = generate_answer(ent["word"], top_chunks)

                # Store results
                results[ent["word"]] = {
                    "type": ent["entity_group"],
                    "top_chunks": top_chunks,
                    "answer": answer
                }

            # Step 6: Show Final Structured Output
            st.subheader("🧾 Final Structured Output")
            for entity, info in results.items():
                st.markdown(f"### 🔍 Entity: {entity} ({info['type']})")
                st.markdown("**Top Relevant Chunks:**")
                for i, (chunk, score) in enumerate(info["top_chunks"], 1):
                    st.markdown(f"{i}. *Relevance Score:* {score:.4f}\n> {chunk}")
                st.markdown("**🧠 Generated Answer:**")
                st.success(info["answer"])


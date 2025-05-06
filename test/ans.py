# generate.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")  # No sentencepiece required
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

def generate_answer(query, context_chunks, max_length=128):
    # Ensure context_chunks is in the correct format (top 3 chunks)
    context = " ".join(chunk for chunk, _ in context_chunks[:3])  # top 3 chunks
    input_text = f"question: {query} context: {context}"
    
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    outputs = model.generate(inputs, max_length=150, num_beams=4, early_stopping=True)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

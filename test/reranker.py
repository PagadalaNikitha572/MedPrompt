# drag.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load BioBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = AutoModelForSequenceClassification.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# Function to compute relevance score between query and document chunk
def rerank_chunk(query, chunk):
    # Combine the query and chunk into one input sequence
    inputs = tokenizer(query, chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Pass through the model to get logits (raw scores)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the prediction score (classification logits)
    score = torch.softmax(outputs.logits, dim=-1)[0][1].item()  # Relevance score (binary classification, 1 is relevant)
    
    return score

# Function to rerank a list of chunks based on their relevance to the query
def rerank_chunks(query, chunks):
    scored_chunks = []
    
    for chunk in chunks:
        score = rerank_chunk(query, chunk)
        scored_chunks.append((chunk, score))
    
    # Sort chunks by score in descending order
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    
    return scored_chunks

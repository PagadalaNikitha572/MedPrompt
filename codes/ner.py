from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# Load the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Clinical-AI-Apollo/Medical-NER")
model = AutoModelForTokenClassification.from_pretrained("Clinical-AI-Apollo/Medical-NER")

# Create NER pipeline
ner_pipe = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Function for NER on input text
def recognize_entities(text):
    entities = ner_pipe(text)
    structured_entities = [
        {'word': ent['word'], 'entity_group': ent['entity_group'], 'score': ent['score']} 
        for ent in entities
    ]
    return structured_entities

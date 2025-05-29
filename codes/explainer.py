import json
import requests

# Load local corpus
with open('local_corpus.json', 'r', encoding='utf-8') as f:
    local_corpus = json.load(f)

def explain_entity(entity_tuple):
    """
    entity_tuple: (entity_text, entity_type)
    Returns: string explanation
    """
    text, ent_type = entity_tuple
    key = text.lower()

    # 1. Try local corpus
    if key in local_corpus:
        return f"'{text}': {local_corpus[key]}"

   
    
    # 2. Fallback to entity type definition from corpus
    if ent_type in local_corpus:
        return f"'{text}' is a {ent_type.replace('_', ' ').title()}. {local_corpus[ent_type]}"

    return f"No explanation available for '{text}'."


def explain_entities(entity_list):
    return [explain_entity(ent) for ent in entity_list]
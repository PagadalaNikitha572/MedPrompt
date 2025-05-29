def filter_entities(entities):
    important_types = {
        "DISEASE_DISORDER", "SIGN_SYMPTOM", "BIOLOGICAL_STRUCTURE", 
        "DRUG", "HISTORY", "TEST", "VITAL_SIGN", "PROCEDURE"
    }

    filtered = []
    for ent in entities:
        ent_type = ent.get("entity_group")
        score = float(ent.get("score", 0))

        if ent_type in important_types or score > 0.4:
            # Append a tuple of (text, entity_type)
            filtered.append((ent.get("word"), ent_type))  # Use word or text depending on the NER model's output
    
    return filtered

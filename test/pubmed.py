# build_corpus.py

from Bio import Entrez
import time
import os

Entrez.email = "nktapagadala@gmail.com"  # REQUIRED by NCBI

# 🔁 You can load this from a file later if needed
clinical_topics = [
    "diabetes", "hypertension", "asthma", "cancer", "stroke", "depression", "anxiety",
    "heart failure", "COPD", "tuberculosis", "HIV", "malaria", "pneumonia", "hepatitis B",
    "hepatitis C", "migraine", "arthritis", "lupus", "multiple sclerosis", "Crohn's disease",
    "ulcerative colitis", "obesity", "anemia", "thyroid disorder", "chronic kidney disease",
    "epilepsy", "autism", "dementia", "Alzheimer's disease", "Parkinson's disease",
    "COVID-19", "influenza", "eczema", "psoriasis", "gout", "pancreatitis", "gastritis",
    "GERD", "bronchitis", "endometriosis", "meningitis", "sepsis", "leukemia", "lymphoma",
    "myocardial infarction", "angina", "atrial fibrillation", "valvular heart disease"
]

os.makedirs("data", exist_ok=True)
output_file = "data/corpus.txt"

def fetch_abstracts(topic, max_results=100):
    try:
        search_handle = Entrez.esearch(db="pubmed", term=topic, retmax=max_results)
        search_results = Entrez.read(search_handle)
        id_list = search_results["IdList"]

        fetch_handle = Entrez.efetch(db="pubmed", id=",".join(id_list), rettype="abstract", retmode="text")
        raw_text = fetch_handle.read()
        return raw_text.split("\n\n")
    except Exception as e:
        print(f"⚠️ Error fetching abstracts for {topic}: {e}")
        return []

def build_corpus():
    with open(output_file, "w", encoding="utf-8") as f:
        for topic in clinical_topics:
            print(f"🔍 Fetching abstracts for: {topic}")
            abstracts = fetch_abstracts(topic, max_results=100)
            print(f"✅ {len(abstracts)} abstracts retrieved for '{topic}'")
            for abs_text in abstracts:
                cleaned = abs_text.strip().replace("\n", " ")
                if len(cleaned) > 100:
                    f.write(cleaned + "\n")
            time.sleep(1)  # NCBI API rate limit
    print(f"\n📁 All abstracts saved to: {output_file}")

if __name__ == "__main__":
    build_corpus()

from transformers import pipeline

models = {
    "DistilBERT (SST-2)": "distilbert-base-uncased-finetuned-sst-2-english",
    "RoBERTa Twitter": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "Multilingual BERT": "nlptown/bert-base-multilingual-uncased-sentiment"
}

text = """
The movie had stunning visuals and great music,
but the story was weak and the pacing was slow.
"""

pipelines = {
    name: pipeline("sentiment-analysis", model=model)
    for name, model in models.items()
}

print("TEXT:")
print(text.strip())
print("\nMODEL COMPARISON:\n")

for name, pipe in pipelines.items():
    result = pipe(text)[0]
    print(f"{name}")
    print(f"  Label : {result['label']}")
    print(f"  Score : {round(result['score'], 3)}\n")

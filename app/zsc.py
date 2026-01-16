from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

reviews = [
    "This movie was absolutely amazing and well acted.",
    "The plot was boring and the characters were annoying."
]

labels = ["positive", "negative"]

for review in reviews:
    result = classifier(review, labels)
    print("\nReview:", review)
    print("Prediction:", result["labels"][0], 
          "| Score:", round(result["scores"][0], 3))

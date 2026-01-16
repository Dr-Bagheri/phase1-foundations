from datasets import load_dataset

dataset = load_dataset("imdb")

def longer_than_200(example):
    return len(example["text"].split()) > 200

filtered = dataset["train"].filter(longer_than_200)

print("Number of long reviews:", len(filtered))
print("\nFirst example:\n")
print(filtered[0]["text"][:500], "...")

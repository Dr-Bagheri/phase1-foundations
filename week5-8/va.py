import torch
from transformers import BertTokenizer, BertModel
from bertviz import model_view

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name, output_attentions=True)

sentence = "The cat sat on the mat because it was tired."

inputs = tokenizer(sentence, return_tensors="pt")
outputs = model(**inputs)

attention = outputs.attentions
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])


html = model_view(attention, tokens, html_action='return')

with open("attention.html", "w", encoding="utf-8") as f:
    f.write(html.data)   

print("Saved to attention.html")
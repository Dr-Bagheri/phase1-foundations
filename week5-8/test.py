from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = "mistralai/Mistral-7B-Instruct-v0.2"

model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
model = PeftModel.from_pretrained(model, "./mistral-samsum-lora")

tokenizer = AutoTokenizer.from_pretrained(base_model)

prompt = """### Instruction:
Summarize the following conversation.

### Input:
A: Are we meeting today?
B: Yes at 5 PM.

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

output = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
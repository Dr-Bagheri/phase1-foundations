import time
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

from peft import LoraConfig, get_peft_model


dataset = load_dataset("imdb")
train_dataset = dataset["train"].shuffle(seed=42).select(range(2000))


model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)


lora_config = LoraConfig(
    r=8,                      # rank
    lora_alpha=16,
    target_modules=["c_attn"],  
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)


def print_trainable_parameters(model):
    trainable = 0
    total = 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"Trainable params: {trainable}")
    print(f"Total params: {total}")
    print(f"% trainable: {100 * trainable / total:.2f}%")

print_trainable_parameters(model)


def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

tokenized_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text"])


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)


training_args = TrainingArguments(
    output_dir="./lora_results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    logging_steps=50,
    report_to="none"
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)


start_time = time.time()
trainer.train()
end_time = time.time()

print(f"\n Training time: {end_time - start_time:.2f} seconds")


model.save_pretrained("./lora_model")
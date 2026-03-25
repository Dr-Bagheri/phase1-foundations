import time
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)


dataset = load_dataset("imdb")


train_dataset = dataset["train"].shuffle(seed=42).select(range(2000))


model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)


tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)


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
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    logging_steps=50,
    save_steps=200,
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
training_time = end_time - start_time

print(f"\n Training time: {training_time:.2f} seconds")


trainer.save_model("./fine_tuned_model")
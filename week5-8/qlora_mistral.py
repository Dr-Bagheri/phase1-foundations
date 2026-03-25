import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer


model_name = "mistralai/Mistral-7B-Instruct-v0.2"


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)


tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)


peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)


dataset = load_dataset("samsum")


def format_example(example):
    return {
        "text": f"""### Instruction:
Summarize the following conversation.

### Input:
{example['dialogue']}

### Response:
{example['summary']}"""
    }

train_dataset = dataset["train"].map(format_example)


training_args = TrainingArguments(
    output_dir="./qlora-mistral",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    report_to="none"
)


trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_args
)


trainer.train()


model.save_pretrained("./mistral-samsum-lora")
tokenizer.save_pretrained("./mistral-samsum-lora")
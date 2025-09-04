import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# --- Configuration ---
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
NEW_MODEL_NAME = "llama-3-8b-fake-news-unbiased"
DATASET_PATH = "train_data.csv" # Uses the BALANCED training data

# --- Load the dataset ---
dataset = load_dataset("csv", data_files=DATASET_PATH, split="train")

# --- Load Model and Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=quant_config, device_map="auto")
model.config.use_cache = False
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"))

# --- Pre-process and tokenize the dataset ---
def format_and_tokenize(examples):
    prompts = []
    for msg, label in zip(examples["post_message"], examples["label"]):
        label_text = "thật" if label == 0 else "giả"
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Bạn là một chuyên gia phân loại tin tức. Phân loại tin tức sau đây là 'thật' hoặc 'giả'.<|eot_id|><|start_header_id|>user<|end_header_id|>

Tin tức: {msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Phân loại: {label_text}<|eot_id|>"""
        prompts.append(prompt)
    
    return tokenizer(prompts, truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(format_and_tokenize, batched=True, remove_columns=dataset.column_names) # type: ignore

# --- Set up Training Arguments ---
training_args = TrainingArguments(
    output_dir="./results-unbiased", num_train_epochs=1, per_device_train_batch_size=1,
    gradient_accumulation_steps=4, save_steps=100, logging_steps=10, learning_rate=2e-4,
    weight_decay=0.001, bf16=True, max_grad_norm=0.3, warmup_ratio=0.03, lr_scheduler_type="constant"
)

# --- Using the STANDARD, STABLE Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset, # type: ignore
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# --- Start Training ---
print("Starting fine-tuning with BALANCED data...")
trainer.train()

# --- Save the fine-tuned model ---
print(f"Saving unbiased model to ./{NEW_MODEL_NAME}")
trainer.save_model(NEW_MODEL_NAME)
tokenizer.save_pretrained(NEW_MODEL_NAME)
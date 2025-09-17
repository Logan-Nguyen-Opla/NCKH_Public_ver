import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

# --- Configuration: Using Llama 3 ---
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
NEW_MODEL_NAME = "llama-3-8b-fake-news-tuned"
DATASET_PATH = "train_data.csv"

# --- Load the dataset ---
dataset = load_dataset("csv", data_files=DATASET_PATH, split="train")

# --- NEW: Llama 3 Specific Prompt Formatting ---
def format_prompt_llama3(example):
    label_text = "thật" if example["label"] == 0 else "giả"
    # Llama 3 uses a specific chat template with special tokens.
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Bạn là một chuyên gia phân loại tin tức. Phân loại tin tức sau đây là 'thật' hoặc 'giả'.<|eot_id|><|start_header_id|>user<|end_header_id|>

Tin tức: {example['post_message']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Phân loại: {label_text}<|eot_id|>"""
    return prompt

# --- Set up quantization and LoRA ---
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    # Target modules are different for Llama 3
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# --- Load the model and tokenizer ---
print(f"Loading base model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    device_map="auto",
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# --- Set up Training Arguments ---
training_args = TrainingArguments(
    output_dir="./results-llama3",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    save_steps=50,
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.001,
    bf16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
)

# --- SFTTrainer Initialization ---
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=lora_config,
    formatting_func=format_prompt_llama3,
    max_seq_length=1024,
)

# --- Start Training ---
print("Starting Llama 3 fine-tuning...")
trainer.train()

# --- Save the fine-tuned model ---
print(f"Saving fine-tuned Llama 3 model to ./{NEW_MODEL_NAME}")
trainer.save_model(NEW_MODEL_NAME)
tokenizer.save_pretrained(NEW_MODEL_NAME)
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

# --- Configuration ---
BASE_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
FINETUNED_MODEL_PATH = "llama-3-8b-fake-news-reasoner"
TEST_DATASET_PATH = "test_data_with_reasoning.csv"

# --- NEW: Set up quantization for memory-efficient loading ---
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

# --- Load the fine-tuned model ---
print("Loading the fine-tuned REASONER model...")
tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_PATH)

# Load the base model with 4-bit quantization
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=quant_config,
    device_map="auto",
)

# Apply the LoRA adapter to the quantized base model
model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL_PATH)
model = model.merge_and_unload() # type: ignore
model.eval()

# --- Load the test data ---
test_df = pd.read_csv(TEST_DATASET_PATH)
test_df['label_text'] = test_df['label'].apply(lambda x: "thật" if x == 0 else "giả")

# Define keywords to infer classification from the generated reasoning
FAKE_KEYWORDS = ["sai", "giả", "vô căn cứ", "không chính xác", "tin giả", "bịa đặt", "sai lệch"]
REAL_KEYWORDS = ["thật", "đúng", "chính xác", "đáng tin cậy", "xác thực", "xác nhận"]

# --- Generate predictions ---
predictions = []
true_labels = []

print(f"Running evaluation on {len(test_df)} test samples...")
for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Bạn là một chuyên gia phân tích tin tức. Viết một câu phân tích ngắn gọn cho tin tức sau đây.<|eot_id|><|start_header_id|>user<|end_header_id|>

Tin tức: {row['post_message']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reasoning_text = response.split("<|end_header_id|>")[-1].strip().lower()
    
    # Infer the classification from the generated reasoning
    if any(keyword in reasoning_text for keyword in FAKE_KEYWORDS):
        predictions.append("giả")
    elif any(keyword in reasoning_text for keyword in REAL_KEYWORDS):
        predictions.append("thật")
    else:
        predictions.append("unknown")

    true_labels.append(row['label_text'])

# --- Calculate and Print Accuracy ---
print("\n--- REASONER Model Evaluation Results ---")
accuracy = accuracy_score(true_labels, predictions)
print(f"✅ Overall Accuracy: {accuracy * 100:.2f}%\n")

labels = ["thật", "giả", "unknown"]
target_names = ["thật (real)", "giả (fake)", "unknown (unclear)"]
print("Detailed Classification Report:")
print(classification_report(true_labels, predictions, labels=labels, target_names=target_names))
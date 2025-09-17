import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

# --- Configuration ---
BASE_MODEL_NAME = "vilm/vinallama-7b"
FINETUNED_MODEL_PATH = "vinallama-7b-fake-news-tuned" # Path to your new model
TEST_DATASET_PATH = "test_data.csv" # The 20% test data

# --- Load the fine-tuned model ---
print("Loading the fine-tuned model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)

# Load the base model in 4-bit
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Merge the LoRA weights with the base model for evaluation
model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL_PATH)
model = model.merge_and_unload() # type: ignore
model.eval() # Set the model to evaluation mode

# --- Load the test data ---
test_df = pd.read_csv(TEST_DATASET_PATH)
# Map labels to text for comparison
test_df['label_text'] = test_df['label'].apply(lambda x: "thật" if x == 0 else "giả")

# --- Generate predictions ---
predictions = []
true_labels = []

print(f"Running evaluation on {len(test_df)} test samples...")
for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    prompt = f"""<s>[INST] <<SYS>>
Bạn là một chuyên gia phân loại tin tức. Phân loại tin tức sau đây là 'thật' hoặc 'giả'.
<</SYS>>

Tin tức: {row['post_message']} [/INST]
"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        # Generate just enough tokens to get the answer
        outputs = model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
    
    # Decode the output and clean it up to get just the prediction
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prediction_text = response.split("[/INST]")[-1].strip().lower()
    
    # Simple check to see if the model's first word is 'thật' or 'giả'
    if "thật" in prediction_text:
        predictions.append("thật")
    elif "giả" in prediction_text:
        predictions.append("giả")
    else:
        # If the model gives a weird output, we'll count it as wrong.
        predictions.append("unknown") 

    true_labels.append(row['label_text'])

# --- Calculate and Print Accuracy ---
print("\n--- Evaluation Results ---")
accuracy = accuracy_score(true_labels, predictions)
print(f"✅ Overall Accuracy: {accuracy * 100:.2f}%\n")

# Print a detailed report (precision, recall, f1-score)
print("Detailed Classification Report:")
print(classification_report(true_labels, predictions, target_names=['thật (real)', 'giả (fake)']))
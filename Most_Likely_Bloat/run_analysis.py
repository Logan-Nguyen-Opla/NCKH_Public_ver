import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Load all the components we built/need ---

print("Loading all necessary models and data (this may take a minute)...")

# 1. Load the knowledge base documents
with open("documents.txt", "r", encoding="utf-8") as f:
    documents = [line.strip() for line in f.readlines()]

# 2. Load the FAISS index
index = faiss.read_index("knowledge_base.index")

# 3. Load the embedding model (same one as before)
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# 4. Load the Language Model (LLM) and its tokenizer
MODEL_NAME = "VietAI/gpt-neo-1.3B-vietnamese-news"
print(f"Loading {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("All components loaded successfully!")


def analyze_news(news_snippet: str, k: int = 3):
    """
    This is the main RAG function.
    It takes a news snippet, finds relevant facts, and generates an analysis.
    """
    
    # 1. Embed the user's query
    query_embedding = embedding_model.encode([news_snippet], convert_to_tensor=True).cpu().numpy().astype('float32')
    
    # 2. Retrieve the top 'k' most similar documents from our knowledge base
    distances, indices = index.search(query_embedding, k)
    retrieved_docs = [documents[i] for i in indices[0]]
    
    # 3. Augment the prompt
    context = "\n\n---\n\n".join(retrieved_docs)
    prompt = f"""Dựa vào những thông tin có thật dưới đây:
    
<BỐI Cảnh>
{context}
</BỐI Cảnh>

Hãy phân tích và cho biết đoạn tin tức sau đây có khả năng là thật hay giả. Giải thích ngắn gọn tại sao.

<TIN TỨC CẦN KIỂM TRA>
{news_snippet}
</TIN TỨC CẦN KIỂM TRA>

PHÂN TÍCH:
"""

    # 4. Generate a response from the LLM
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate text with sampling enabled
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id # Important for smaller models
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    analysis = response.split("PHÂN TÍCH:")[-1].strip()
    return analysis, retrieved_docs


# --- Let's test it! ---
if __name__ == "__main__":
    test_news = "Đại tướng Tô Lâm đã ký quyết định thăng cấp bậc hàm cho 2 chiến sỹ hy sinh ở Đà Nẵng."
    print(f"\n--- Analyzing News Snippet ---\n\"{test_news}\"\n")
    analysis, retrieved_facts = analyze_news(test_news)
    
    print("--- LLM Analysis ---")
    print(analysis)
    print("\n--- Retrieved Facts Used for Analysis ---")
    for i, fact in enumerate(retrieved_facts):
        print(f"{i+1}. {fact[:150]}...")
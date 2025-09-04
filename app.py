import gradio as gr
from gradio import themes
from googleapiclient.discovery import build
import google.generativeai as genai
import faiss
from sentence_transformers import SentenceTransformer
import re
import pandas as pd

# --- CONFIGURATION ---
GOOGLE_API_KEY = "AIzaSyC05FamAtuaretG96GMx-cgdq1kT3B-uRk"
SEARCH_ENGINE_ID = "f34f8a4816771488b"
LLM_NAME = "gemini-2.5-flash" 
RELEVANCY_THRESHOLD = 1.0

# --- Load All Components ---
print("Loading all components...")
genai.configure(api_key=GOOGLE_API_KEY) # type: ignore
llm = genai.GenerativeModel(model_name=LLM_NAME) # type: ignore
try:
    with open("documents.txt", "r", encoding="utf-8") as f:
        documents = [line.strip() for line in f.readlines()]
    index = faiss.read_index("knowledge_base.index")
    embedding_model = SentenceTransformer("BAAI/bge-m3", trust_remote_code=True)
    local_db_enabled = True
    print("Local database loaded successfully.")
except Exception as e:
    local_db_enabled = False
    print(f"Could not load local database (Error: {e}). App will rely only on Google Search.")
print("All components loaded. Launching the app...")

# --- Main analysis function ---
def analyze_news(news_snippet, relevancy_threshold, num_search_results):
    # Default empty states
    source = "Idle"
    raw_text = ""
    parsed_df = pd.DataFrame()
    verdict_html = "<div style='padding:20px; border-radius:10px; text-align:center;'>Awaiting Input</div>"
    context = ""

    if not news_snippet or not news_snippet.strip():
        return source, raw_text, parsed_df, verdict_html, context

    # Step 1: Try local DB
    if local_db_enabled:
        query_embedding = embedding_model.encode([news_snippet], convert_to_tensor=True).cpu().numpy().astype('float32')
        distances, indices = index.search(query_embedding, 1)
        best_distance = distances[0][0]
        if best_distance < relevancy_threshold:
            source = f"Local Database (Match Distance: {best_distance:.4f})"
            context = documents[indices[0][0]]
    
    # Step 2: Fallback to Google Search
    if not context:
        source = "Google Search (Live)"
        context = google_search(news_snippet, num_search_results)

    if context.startswith("Error:"):
        verdict_html = "<div style='background-color:#dc3545; color:white; padding:20px; border-radius:10px; text-align:center; font-size:24px; font-weight:bold;'>ERROR</div>"
        return source, "Context retrieval failed.", pd.DataFrame(), verdict_html, context

    # Step 3: Analyze with Gemini
    prompt = f"""Dựa vào BỐI CẢNH sau, hãy phân tích TIN TỨC CẦN KIỂM TRA. Trả lời bằng hai phần rõ ràng:
1. Một câu phân tích ngắn gọn bằng tiếng Việt.
2. Trên một dòng riêng, ghi "Độ chính xác:" theo sau là một con số phần trăm (0-100).
BỐI CẢNH:
{context}
TIN TỨC CẦN KIỂM TRA:
{news_snippet}
PHÂN TÍCH:
"""

    try:
        response = llm.generate_content(prompt)
        raw_text = response.text.strip()
    except Exception as e:
        verdict_html = "<div style='background-color:#dc3545; color:white; padding:20px; border-radius:10px; text-align:center; font-size:24px; font-weight:bold;'>ERROR</div>"
        return source, f"Error during Gemini analysis: {e}", pd.DataFrame(), verdict_html, context

    # --- Verdict Logic with Enhanced Styling ---
    verdict = "KHÔNG XÁC ĐỊNH"
    parsed_output = {"verdict": "Undetermined", "score": None, "raw_output": raw_text}
    verdict_html = "<div style='background-color:#6c757d; color:white; padding:20px; border-radius:10px; text-align:center; font-size:24px; font-weight:bold;'>KHÔNG XÁC ĐỊNH</div>"

    try:
        match = re.search(r"Độ chính xác:.*?(\d+\.?\d*)", raw_text, re.IGNORECASE)
        if match:
            accuracy = float(match.group(1))
            if accuracy > 90:
                verdict = "ĐÁNG TIN CẬY"
                verdict_html = f"<div style='background-color:#28a745; color:white; padding:20px; border-radius:10px; text-align:center; font-size:24px; font-weight:bold;'>{verdict} ({accuracy}%)</div>"
            elif accuracy < 10:
                verdict = "SAI LỆCH"
                verdict_html = f"<div style='background-color:#dc3545; color:white; padding:20px; border-radius:10px; text-align:center; font-size:24px; font-weight:bold;'>{verdict} ({accuracy}%)</div>"
            else:
                verdict = "KHÔNG CHẮC CHẮN"
                verdict_html = f"<div style='background-color:#ffc107; color:black; padding:20px; border-radius:10px; text-align:center; font-size:24px; font-weight:bold;'>{verdict} ({accuracy}%)</div>"
            parsed_output.update({"verdict": verdict, "score": accuracy})
        else:
            parsed_output.update({"verdict": "Undetermined"})
    except (ValueError, IndexError):
        verdict = "LỖI PHÂN TÍCH"
        parsed_output.update({"verdict": "Parse Error"})
        verdict_html = "<div style='background-color:#6c757d; color:white; padding:20px; border-radius:10px; text-align:center; font-size:24px; font-weight:bold;'>LỖI PHÂN TÍCH</div>"
        
    return source, raw_text, pd.DataFrame([parsed_output]), verdict_html, context

def google_search(query: str, num_results: int = 4):
    """Helper function for Google Search"""
    service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
    result = service.cse().list(q=query, cx=SEARCH_ENGINE_ID, num=num_results).execute()
    if "items" in result:
        snippets = [f"Result {i+1}: {item['title']}\n{item['snippet']}" for i, item in enumerate(result["items"])]
        return "\n\n---\n\n".join(snippets)
    return "No relevant search results found from Google."

# --- Custom "Terminal" Theme ---
theme = themes.Base(
    primary_hue=themes.colors.green,
    secondary_hue=themes.colors.gray,
    font=(themes.GoogleFont("Inconsolata"), "monospace"),
).set(
    body_background_fill="#0A0A0A", body_text_color="#E0E0E0",
    button_primary_background_fill="#00A36C", button_primary_text_color="#FFFFFF",
    block_background_fill="#1C1C1C", block_border_width="1px", block_border_color="#333333",
    block_label_text_color="#00A36C", block_title_text_color="#FFFFFF", input_background_fill="#2C2C2C",
)

# --- Build the "Mission Control" Interface ---
with gr.Blocks(theme=theme, title="AI News Analyzer") as app:
    gr.Markdown("# 📰 AI News Analyzer: Mission Control")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. INPUT & PARAMETERS")
            input_textbox = gr.Textbox(lines=8, placeholder="Paste a Vietnamese news snippet here...", label="Query")
            threshold_slider = gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.05, label="Local DB Relevancy Threshold")
            k_slider = gr.Slider(minimum=1, maximum=10, value=4, step=1, label="Number of Google Search Results")
            submit_button = gr.Button("ANALYZE", variant="primary")
            
        with gr.Column(scale=2):
            gr.Markdown("### 2. ANALYSIS & OUTPUT")
            # Using Markdown for the big, bold, styled verdict
            verdict_output = gr.Markdown(label="Final Verdict")
            source_output = gr.Textbox(label="Context Source", interactive=False)
            parsed_output_df = gr.DataFrame(headers=["verdict", "score", "raw_output"], label="Parsed AI Output", interactive=False)
            
            with gr.Accordion("Show Full Raw AI Response", open=False):
                raw_output_text = gr.Textbox(label="Raw LLM Output", interactive=False)
            with gr.Accordion("Show Context Fed to AI", open=False):
                context_output = gr.Textbox(label="Retrieved Context", interactive=False)
    
    submit_button.click(
        fn=analyze_news,
        inputs=[input_textbox, threshold_slider, k_slider],
        outputs=[source_output, raw_output_text, parsed_output_df, verdict_output, context_output]
    )

if __name__ == "__main__":
    if GOOGLE_API_KEY == "YOUR_API_KEY_HERE" or SEARCH_ENGINE_ID == "YOUR_SEARCH_ENGINE_ID_HERE":
        print("\nERROR: Please paste your API keys into the script.\n")
    else:
        app.launch()
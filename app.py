import gradio as gr
from googleapiclient.discovery import build
import google.generativeai as genai
import re
import os

# --- 🤫 CONFIGURATION ---
# Your keys have been added below.
GOOGLE_API_KEY = "AIzaSyC05FamAtuaretG96GMx-cgdq1kT3B-uRk"
SEARCH_ENGINE_ID = "f34f8a4816771488b"
LLM_NAME = "gemini-2.5-pro"

# --- Configure the Gemini API ---
try:
    genai.configure(api_key=GOOGLE_API_KEY) # type: ignore
except Exception as e:
    print(f"API Key configuration failed. Please ensure you have a valid key. Error: {e}")

# Set up the Gemini model
generation_config = {"temperature": 0.7, "top_p": 1, "top_k": 1, "max_output_tokens": 2048}
llm = genai.GenerativeModel(model_name=LLM_NAME, generation_config=generation_config) # type: ignore

print("Gemini model configured. Launching the app...")

# --- Google Search Function ---
def google_search(query: str, num_results: int = 5):
    try:
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        result = service.cse().list(q=query, cx=SEARCH_ENGINE_ID, num=num_results).execute()
        
        if "items" not in result:
            return "No relevant search results found from Google."
            
        snippets = [f"Result {i+1}: {item['title']}\n{item['snippet']}" for i, item in enumerate(result["items"])]
        return "\n\n---\n\n".join(snippets)

    except Exception as e:
        print(f"An error occurred with the Google Search API: {e}")
        return "Error: Could not retrieve search results from Google."

# --- Main analysis function, now using Gemini ---
def analyze_news(news_snippet: str):
    if GOOGLE_API_KEY == "YOUR_API_KEY_HERE" or SEARCH_ENGINE_ID == "YOUR_SEARCH_ENGINE_ID_HERE":
        return "ERROR: Please configure your API keys at the top of the app.py script.", {"Error": 1.0}, ""

    if not news_snippet or not news_snippet.strip():
        return "Please enter a news snippet to analyze.", {"N/A": 1.0}, ""

    print(f"Searching Google for: {news_snippet}")
    context = google_search(news_snippet)
    
    prompt = f"""Dựa vào BỐI CẢNH từ kết quả tìm kiếm Google, hãy phân tích TIN TỨC CẦN KIỂM TRA. Trả lời bằng hai phần rõ ràng:
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
        full_analysis_text = response.text.strip()
    except Exception as e:
        print(f"An error occurred with the Gemini API: {e}")
        return f"Error during analysis: {e}", {"Error": 1.0}, context

    accuracy_label = {"N/A": 1.0}
    try:
        match = re.search(r"Độ chính xác:.*?(\d+\.?\d*)", full_analysis_text, re.IGNORECASE)
        if match:
            accuracy = float(match.group(1))
            accuracy_label = {"Thật (Real)": accuracy / 100, "Giả (Fake)": 1 - (accuracy / 100)}
        else:
             accuracy_label = {"Không xác định": 1.0}
    except (ValueError, IndexError):
        accuracy_label = {"Lỗi phân tích": 1.0}

    retrieved_facts_display = context
    
    return full_analysis_text, accuracy_label, retrieved_facts_display

# --- Build the interface ---
with gr.Blocks(theme="soft") as app:
    gr.Markdown("# 📰 Vietnamese News Analyzer (Powered by Gemini Pro)")
    gr.Markdown("An AI tool to estimate the accuracy of a news snippet by grounding its analysis with live Google search results and Google's Gemini Pro.")
    with gr.Row():
        with gr.Column(scale=2):
            input_textbox = gr.Textbox(lines=8, placeholder="Paste a Vietnamese news snippet here to search and analyze...", label="News to Analyze")
            submit_button = gr.Button("Analyze", variant="primary")
        with gr.Column(scale=1):
            confidence_label = gr.Label(label="Estimated News Accuracy")
    main_analysis_output = gr.Textbox(label="AI Analysis", lines=6, interactive=False)
    with gr.Accordion("Show Google Search Results Used for Analysis", open=False):
        retrieved_facts_output = gr.Textbox(label="Retrieved Google Search Snippets", interactive=False)
    submit_button.click(
        fn=analyze_news,
        inputs=input_textbox,
        outputs=[main_analysis_output, confidence_label, retrieved_facts_output]
    )

if __name__ == "__main__":
    app.launch()
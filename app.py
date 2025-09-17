import gradio as gr
from googleapiclient.discovery import build
import google.generativeai as genai
import re
import os
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")
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
        result = service.cse().list(q=query, cx=SEARCH_ENGINE_ID, num=num_results).execute() # type: ignore
        return [item['snippet'] for item in result.get('items', [])]
    except Exception as e:
        return [f"Google Search Error: {e}"]

# --- Main Analysis Function ---
def analyze_news(news_snippet: str):
    if not news_snippet.strip():
        return "Please enter a news snippet to analyze.", {}, ""

    search_results = google_search(news_snippet)
    context = "\n".join(search_results)

    # --- Build the prompt for Gemini ---
    prompt = f"""
    Based on the following search results from Google, analyze the provided news snippet and determine if it is likely "Thật" (Real) or "Giả" (Fake).

    Provide a step-by-step analysis and then conclude with a single percentage of confidence. For example: "Confidence: 85%".

    **Search Results:**
    {context}

    **News Snippet to Analyze:**
    {news_snippet}

    **Analysis:**
    """

    # --- Get the analysis from Gemini ---
    full_analysis_text = llm.generate_content(prompt).text # type: ignore

    # --- Extract the confidence score ---
    try:
        # Use regex to find a percentage value in the analysis text
        match = re.search(r'(\d+)%', full_analysis_text)
        if match:
            accuracy = int(match.group(1))
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

app.launch(debug=True)
# Vietnamese Fake News Detection

This project contains a suite of tools for detecting fake news in Vietnamese, including data preparation, model fine-tuning, evaluation, and a Gradio-based web interface for real-time analysis.

## Features

* **Data Preparation:** Scripts to split and prepare your dataset for training and testing.
* **Fine-Tuning:** A script to fine-tune a large language model (LLM) for fake news classification.
* **Evaluation:** A script to evaluate the performance of your fine-tuned model.
* **Gradio Web App:** An interactive web interface to analyze news snippets in real-time using Google Search and the Gemini Pro model.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/vietnamese-fake-news.git](https://github.com/your-username/vietnamese-fake-news.git)
    cd vietnamese-fake-news
    ```
    *(Remember to replace `your-username` with your actual GitHub username!)*

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Set up Git LFS for large file handling:**
    The model files are large and are tracked using Git LFS. You will need to install Git LFS on your system.
    ```bash
    git lfs install
    git lfs pull
    ```

4.  **Set up Environment Variables:**
    The Gradio application requires a Google API Key and a Custom Search Engine ID.
    * Create a file named `.env` in the root of the project.
    * Add your API keys to the `.env` file in the following format:
        ```
        GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
        SEARCH_ENGINE_ID="YOUR_SEARCH_ENGINE_ID"
        ```

## Usage

### 1. Prepare the Data
Run the `prepare_data.py` script to split your `public_train.csv` into `train_data.csv` and `test_data.csv`.

python prepare_data.py

### 2. Build the FAISS Database
The build_database.py script creates a FAISS index from the real news articles in your dataset.

python build_database.py

### 3. Fine-Tune the Model
To fine-tune the LLM, run the finetune.py script.

python finetune.py

### 4. Evaluate the Model
After fine-tuning, you can evaluate the model's performance using the evaluate.py script.

python evaluate.py

### 5. Run the Gradio Web App
To launch the web interface for real-time news analysis, run the app.py script.

python app.py

This will launch a local web server. Open the provided URL in your browser to use the application.

### Thanks for reading, this took me a month (70 hours) for a competition for my school, if you have any questions, contact me at: logan.nguyennamlong2012@gmail.com
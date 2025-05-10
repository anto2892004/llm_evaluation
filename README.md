Here is your README file content formatted for a GitHub repository:

---

# 🧠 LLM Medical Evaluation Framework

A comprehensive framework for evaluating multiple Large Language Models (LLMs) on medical question-answering tasks using the **PubMedQA** dataset. This project includes **data extraction**, **multi-metric evaluation**, and a **Streamlit dashboard** for interactive visualization and analysis.

---

## 📋 Overview

This project provides a framework to:

* ✅ Extract medical questions and answers from the **PubMedQA** dataset
* 🤖 Evaluate multiple **LLMs** on these questions using **RAGAS metrics**
* 📊 Visualize and compare performance through an **interactive dashboard**

---

## 🚀 Features

* **🧪 Multi-model Evaluation:**
  Compare the performance of:

  * GPT-4.1 Nano
  * LLaMA 3-8B
  * Nova Micro
  * Qwen 3-4B

* **📏 Multiple Evaluation Metrics:**

  * **Faithfulness:** Factual consistency with reference context
  * **Context Relevancy:** Relevance of context to the question
  * **Context Recall:** Coverage of important information from context

* **📊 Interactive Dashboard:**

  * Streamlit-based
  * Model comparison
  * Visual plots and summary tables

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-medical-evaluation.git
cd llm-medical-evaluation

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🔧 Configuration

Create a `.env` file in the root directory:

```
NEXUS_API_KEY=your_nexus_api_key
NEXUS_API_URL=your_nexus_api_url
OPEN_ROUTER_API=your_open_router_api_key
```

---

## 📊 Usage

### 1. Download the Dataset

```bash
python data_extract.py
```

This downloads and prepares the **PubMedQA** dataset inside the `pubmed_dataset` directory.

---

### 2. Run Evaluations

You can run evaluations separately or comprehensively:

```bash
# Faithfulness metric
python faithfullness.py

# Context relevancy
python context_relevancy.py

# Context recall
python context_recall.py

# Or all at once
python evaluate.py
```

---

### 3. Launch the Dashboard

```bash
streamlit run dashboard.py
```

Access it at [http://localhost:8501](http://localhost:8501).

---

## 📁 Project Structure

```
.
├── dashboard.py              # Streamlit dashboard
├── data_extract.py           # Dataset extraction script
├── evaluate.py               # Comprehensive evaluation runner
├── faithfullness.py          # Faithfulness metric
├── context_relevancy.py      # Context relevancy metric
├── context_recall.py         # Context recall metric
├── test_gpt3.py              # Test script for GPT-3 API
├── models/
│   ├── gpt_4_1_nano.py
│   ├── llama8b.py
│   ├── nova_micro.py
│   └── qwen.py
├── pubmed_dataset/           # PubMedQA dataset
├── result_ragas/             # Evaluation results
│   ├── faithfulness_scores.csv
│   ├── context_relevancy_scores.csv
│   └── context_recall_scores.csv
└── ragas_evaluation/         # Visualizations
    ├── faithfulness_plot.png
    ├── context_relevancy_plot.png
    └── context_recall_plot.png
```

---

## 📈 Dashboard Features

* 📂 Tabs for each metric
* 🧠 Model selection
* 📊 Visual comparison
* 🧮 Summary table of all scores

---

## 🔍 Evaluation Metrics

| Metric                | Description                                          |
| --------------------- | ---------------------------------------------------- |
| **Faithfulness**      | Checks factual alignment of answer with source       |
| **Context Relevancy** | Evaluates how relevant the context is                |
| **Context Recall**    | Measures how complete the answer is based on context |

---

## 🧩 Extending the Framework

### ➕ Add New Models

1. Create a new file in `models/`, e.g., `your_model.py`
2. Implement a `get_response(question, context)` function
3. Import and add it in `evaluate.py`

### ➕ Add New Metrics

1. Use new metrics from RAGAS or define your own
2. Update evaluation scripts
3. Modify `dashboard.py` to include visualizations

---

## 📝 Requirements

* Python 3.8+
* Streamlit
* Pandas
* Matplotlib
* RAGAS
* LangChain
* Hugging Face Datasets
* OpenAI / compatible API access

---

## 🚧 Limitations & Roadmap

* Currently limited to **10 questions** for quick testing
* Planning to support:

  * More models
  * Advanced and custom metrics
  * Granular performance insights

---

## 📄 License

MIT License – see the `LICENSE` file for details.

---

## 🙏 Acknowledgements

* [PubMedQA Dataset](https://pubmedqa.github.io/)
* [RAGAS](https://github.com/explodinggradients/ragas)
* [Streamlit](https://streamlit.io/)

---

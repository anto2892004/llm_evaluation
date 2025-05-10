import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Custom model imports
from models.gpt_4_1_nano import get_gpt4_1_nano_response
from models.llama8b import get_llama8b_response
from models.nova_micro import get_nova_micro_response
from models.qwen import get_qwen_response

# RAGAS imports
from ragas.metrics import ContextRelevance
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas import evaluate

# OpenAI client for other models
import openai

# Load .env variables
load_dotenv()
nexus_api_key = os.getenv("NEXUS_API_KEY")
nexus_api_url = os.getenv("NEXUS_API_URL")
open_router_api = os.getenv("OPEN_ROUTER_API")

# Create required directories
os.makedirs("result_ragas", exist_ok=True)
os.makedirs("ragas_evaluation", exist_ok=True)

# Load and preprocess dataset
df = pd.read_csv('./pubmed_dataset/pqa_labelled.csv').head(10)
questions = df['question'].tolist()
ground_truths = df['long_answer'].tolist()
contexts = [gt[:1000] for gt in ground_truths]  # Truncate context to 1000 chars

# Set up evaluator LLM for RAGAS
openai_llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_key=nexus_api_key,
    openai_api_base=nexus_api_url,
    temperature=0
)
evaluator_llm = LangchainLLMWrapper(openai_llm)
context_relevancy_metric = ContextRelevance(llm=evaluator_llm)

# OpenAI client for other models
client = openai.OpenAI(
    api_key=nexus_api_key,
    base_url=nexus_api_url
)

# 🧠 Model Response Functions
def get_gpt4_1_nano_model_response(question):
    index = questions.index(question)
    context = contexts[index]
    return get_gpt4_1_nano_response(question, context=context)

def get_llama8b_model_response(question):
    index = questions.index(question)
    context = contexts[index]
    return get_llama8b_response(question, context=context)

def get_nova_micro_model_response(question):
    index = questions.index(question)
    context = contexts[index]
    return get_nova_micro_response(question, context=context)

def get_qwen_model_response(question):
    index = questions.index(question)
    context = contexts[index]
    return get_qwen_response(question, context=context)

# 📊 Evaluation Wrapper - Context Relevancy
def evaluate_model_responses(model_name, response_fn):
    print(f"🔍 Evaluating: {model_name}")
    responses = [response_fn(q) for q in questions]
    
    dataset = Dataset.from_dict({
        'question': questions,
        'contexts': [[ctx] for ctx in contexts],
        'answer': responses
    })
    
    # Run evaluation
    result = evaluate(dataset, metrics=[context_relevancy_metric])
    
    # Access context relevancy score - Fixed this part
    # In newer versions of RAGAS, the result is a DataFrame
    # Extract the score using the metric name
    score = result['context_relevance'].mean()  # Use the metric name in snake_case
    print(f"✅ {model_name} Context Relevancy: {score:.4f}")
    return score

# 🚀 Run Evaluation for All Models
scores = {}
scores["gpt4_1_nano"] = evaluate_model_responses("gpt4_1_nano", get_gpt4_1_nano_model_response)
scores["llama-3-8b"] = evaluate_model_responses("llama-3-8b", get_llama8b_model_response)
scores["nova-micro"] = evaluate_model_responses("nova-micro", get_nova_micro_model_response)
scores["qwen-3-4b"] = evaluate_model_responses("qwen-3-4b", get_qwen_model_response)

# 💾 Save Scores to CSV
df_scores = pd.DataFrame(list(scores.items()), columns=["Model", "Context_Relevancy"])
df_scores.to_csv("./result_ragas/context_relevancy_scores.csv", index=False)

# 📈 Plot and Save Visualization
plt.figure(figsize=(10, 6))
plt.bar(df_scores["Model"], df_scores["Context_Relevancy"], color=["blue", "green", "orange", "red"])
plt.title("Context Relevancy Score Comparison")
plt.ylabel("Context Relevancy Score")
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("./ragas_evaluation/context_relevancy_plot.png")
plt.close()

print("🎯 Evaluation complete.")
print("📁 Scores saved to: 'result_ragas/context_relevancy_scores.csv'")
print("📊 Plot saved to: 'ragas_evaluation/context_relevancy_plot.png'")
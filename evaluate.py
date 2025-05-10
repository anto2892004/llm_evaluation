import os
import pandas as pd
from sklearn.metrics import f1_score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from models.gpt_4_1_nano import get_gpt4_1_nano_response
from models.llama8b import get_llama8b_response
from models.nova_micro import get_nova_micro_response
from models.qwen import get_qwen_response

# Load dataset
dataset_path = './pubmed_dataset/pqa_labelled.csv'
df = pd.read_csv(dataset_path)


questions = df['question'].tolist()[:10]
long_answers = df['long_answer'].tolist()[:10]

# Evaluation metric functions
def compute_rouge(prediction, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return scorer.score(reference, prediction)

def compute_f1(prediction, reference):
    pred_tokens = prediction.split()
    ref_tokens = reference.split()
    common = set(pred_tokens) & set(ref_tokens)
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def compute_bleu(prediction, reference):
    reference_tokens = reference.split()
    prediction_tokens = prediction.split()
    smoothie = SmoothingFunction().method4
    return sentence_bleu([reference_tokens], prediction_tokens, smoothing_function=smoothie)

# Evaluation loop
def evaluate_models(models, questions, long_answers):
    results = []
    for model_name, model in models.items():
        model_scores = {"Model": model_name}
        predicted_answers = []

        print(f"Evaluating model: {model_name}...")
        for question in questions:
            try:
                predicted_answer = model(question)
            except Exception as e:
                predicted_answer = "ERROR"
                print(f"[{model_name}] Error on question: {e}")
            predicted_answers.append(predicted_answer)

        rouge_scores = [compute_rouge(pred, ref) for pred, ref in zip(predicted_answers, long_answers)]
        f1_scores = [compute_f1(pred, ref) for pred, ref in zip(predicted_answers, long_answers)]
        bleu_scores = [compute_bleu(pred, ref) for pred, ref in zip(predicted_answers, long_answers)]

        model_scores['ROUGE-1'] = sum([score['rouge1'].fmeasure for score in rouge_scores]) / len(rouge_scores)
        model_scores['ROUGE-2'] = sum([score['rouge2'].fmeasure for score in rouge_scores]) / len(rouge_scores)
        model_scores['ROUGE-L'] = sum([score['rougeL'].fmeasure for score in rouge_scores]) / len(rouge_scores)
        model_scores['F1'] = sum(f1_scores) / len(f1_scores)
        model_scores['BLEU'] = sum(bleu_scores) / len(bleu_scores)

        results.append(model_scores)
    
    return pd.DataFrame(results)

# Define model map
models = {
    "gpt-4.1-nano": get_gpt4_1_nano_response,
    "llama-8b": get_llama8b_response,
    "nova-micro": get_nova_micro_response,
    "qwen": get_qwen_response
}


# Run evaluation
evaluation_results = evaluate_models(models, questions, long_answers)

# Save results
os.makedirs('./results', exist_ok=True)
evaluation_results.to_csv('./results/model_evaluation_results.csv', index=False)
print("\nEvaluation complete. Results saved to ./results/model_evaluation_results.csv")
print(evaluation_results)

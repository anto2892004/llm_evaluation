import os
from dotenv import load_dotenv
import openai

load_dotenv()



nexus_api_key = os.getenv("NEXUS_API_KEY")
nexus_api_url = os.getenv("NEXUS_API_URL")

client = openai.OpenAI(
    api_key=nexus_api_key,
    base_url=nexus_api_url
)

model_ID = "gpt-4.1-nano-2025-04-14"

def get_gpt4_1_nano_response(question, context=None):
    """
    Strictly answers from the given context. Returns "I don't know" if context is missing/irrelevant.
    """
    if context:
        prompt = f"""Answer the question based ONLY on the following context. If the answer isn't in the context, say "I don't know."

Context: {context}
Question: {question}
Answer:"""
    else:
        prompt = question  # Fallback if no context provided

    response = client.chat.completions.create(
        model=model_ID,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,  # Minimize randomness
        stream=False
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    # Test with context enforcement
    test_context = "Mitochondria are the powerhouse of the cell. They generate ATP through oxidative phosphorylation."
    test_question = "What is the role of mitochondria?"
    
    response = get_gpt4_1_nano_response(test_question, test_context)
    print(f"Response from {model_ID}:")
    print(response)
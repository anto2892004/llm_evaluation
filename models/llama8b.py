import requests

OLLAMA_HOST = "https://3177-202-170-204-105.ngrok-free.app/"

def get_llama8b_response(prompt, context=None):
    if context:
        prompt = f"""Answer strictly based on this context. If unsure, say "I don't know."

Context: {context}
Question: {prompt}
Answer:"""

    response = requests.post(f"{OLLAMA_HOST}/api/generate", json={
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    })

    try:
        result = response.json()
        print(result)  # For debugging purposes
        return result.get("response", "No response found in the result.")
    except Exception as e:
        return f"Error: {e}"

# Example usage
question = "What is mitochondria?"
context = "Mitochondria are the powerhouse of the cell."
reply = get_llama8b_response(question, context)
print(reply)

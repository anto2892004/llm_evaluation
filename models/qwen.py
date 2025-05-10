import requests

OPENROUTER_HOST = "https://3177-202-170-204-105.ngrok-free.app/"  # Replace with your actual host if different

def get_qwen_response(prompt, context=None):
    if context:
        prompt = f"""Answer strictly based on this context:

{context}

Question: {prompt}
Answer:"""

    response = requests.post(f"{OPENROUTER_HOST}/api/generate", json={
        "model": "qwen:4b",
        "prompt": prompt,
        "stream": False
    })

    try:
        result = response.json()
        print(result)  # Debug print to see full response structure
        return result.get("response", "No response found in the result.")
    except Exception as e:
        return f"Error: {e}"

# Example usage
question = "What is mitochondria?"
context = "Mitochondria generate energy for cells."
reply = get_qwen_response(question, context)
print(reply)

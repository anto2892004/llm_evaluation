import os
from dotenv import load_dotenv
import openai

load_dotenv()

model_ID = "nova-micro"
client = openai.OpenAI(
    api_key=os.getenv("NEXUS_API_KEY"),
    base_url=os.getenv("NEXUS_API_URL")
)

def get_nova_micro_response(question, context=None):
    """Returns response from Nova Micro model without streaming"""
    messages = [{"role": "user", "content": question}]
    
    if context:
        prompt = f"""Answer strictly based on this context:
        
        {context}
        
        Question: {question}
        Answer:"""
        messages = [{"role": "user", "content": prompt}]

    response = client.chat.completions.create(
        model=model_ID,
        messages=messages,
        stream=False
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    test_question = "What is mitochondria?"
    test_context = "Mitochondria generate energy for cells."
    print(get_nova_micro_response(test_question, test_context))

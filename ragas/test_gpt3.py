from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()

async def main():
    evaluator_llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_key=os.getenv("NEXUS_API_KEY"),
        openai_api_base=os.getenv("NEXUS_API_URL"),
        temperature=0
    )
    msg = [HumanMessage(content="What's the capital of France?")]
    res = await evaluator_llm.ainvoke(msg)
    print(res.content)

# Run the async function
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

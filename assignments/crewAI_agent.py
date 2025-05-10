# Step 1: Install dependencies
!pip install -q crewai openai langchain

# Step 2: Import required modules
import os
from crewai import Agent, Task, Crew

# Step 3: Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# Step 4: Define Agents
researcher = Agent(
    role="Researcher",
    goal="Gather accurate and detailed info",
    backstory="An expert at gathering up-to-date, verified content from credible sources.",
    verbose=True,
    llm="gpt-3.5-turbo"
)

writer = Agent(
    role="Writer",
    goal="Write a clear and engaging blog post",
    backstory="Skilled in crafting blog articles for the tech community.",
    verbose=True,
    llm="gpt-3.5-turbo"
)

reviewer = Agent(
    role="Reviewer",
    goal="Review and improve blog post quality",
    backstory="A perfectionist editor who checks for grammar, clarity, and tone.",
    verbose=True,
    llm="gpt-3.5-turbo"
)

# Step 5: Define Tasks
def get_research_task(topic):
    return Task(
        description=f"Research the topic: {topic}",
        expected_output="A list of bullet points with accurate information",
        agent=researcher
    )

def get_write_task():
    return Task(
        description="Write a detailed blog post based on the research.",
        expected_output="A blog post in markdown format",
        agent=writer
    )

def get_review_task():
    return Task(
        description="Proofread and enhance the blog post for clarity and flow.",
        expected_output="A polished final version of the blog post",
        agent=reviewer
    )

# Step 6: Main Execution
topic = "Impact of AI in Healthcare"

crew = Crew(
    agents=[
        get_research_task(topic).agent,
        get_write_task().agent,
        get_review_task().agent
    ],
    tasks=[
        get_research_task(topic),
        get_write_task(),
        get_review_task()
    ],
    verbose=True
)

result = crew.kickoff()

print("\n💡 Final Blog Post:\n")
print(result)

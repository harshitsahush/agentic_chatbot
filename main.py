import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from crewai import Crew, Process

from tasks import *
from agents import *

load_dotenv()

llm = ChatGroq(
    api_key = os.getenv("GROQ_API_KEY"),
    model = "llama3-70b-8192"
)

# setup agents
chatagent = ChatbotAgents.query_processor(llm)

# setup tasks
query_task = ChatbotTasks.process_query(chatagent)

# setup inputs
temp = input("\n\nEnter the query.\n")
input = {
    "query" : temp
}

crew = Crew(
    agents=[chatagent],
    tasks=[query_task],
    verbose=True, 
    process= Process.sequential
)

results = crew.kickoff(inputs=input)
print(results, "\n\n\n")
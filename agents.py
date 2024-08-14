from crewai import Agent
from tools import *

class ChatbotAgents():
    def query_processor(llm):
        return Agent(
            role = "Query processor",
            goal = "Decide the type of query and process it using appropriate tool. If the query is regarding data present in the stored embeddings, search the quiery result in embeddings. Else, use the web search tool.",
            verbose = True,
            backstory = """You are a very helpful personal assistant that has been trained to decide whether to search in the stored embeddings or the web and return the query results.""",
            tools = [tavily_search, ChatTools.document_search],
            max_iter = 10, 
            llm = llm
        )
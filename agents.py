from crewai import Agent
from tools import *

class ChatbotAgents():
    def query_processor(llm):
        return Agent(
            role = "Query processor",
            goal = "Decide the type of query and process it using appropriate tool. If the query is regarding data present in the document, search the query result in document. If the query is regarding slot booking, appointment booking, reservations, use the tool to search in database. Else, use the web search tool. In case that query is not related to database, First search the document, if results are achieved, return those. Else get the results from the web ONLY if NO results are achieved in the document. Return comprehensive results.",
            verbose = True,
            backstory = """You are a very helpful personal assistant that has been trained to decide whether to search in the stored embeddings or the database or the web and return the query results.""",
            tools = [tavily_search, ChatTools.document_search, ChatTools.availability_query, ChatTools.book_slot],
            max_iter = 10, 
            llm = llm
        )
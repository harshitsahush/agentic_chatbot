from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool
# from crewai_tools import PDFSearchTool
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

tavily_search = TavilySearchResults()
# pdf_search_tool = PDFSearchTool(pdf = "Harshit ML Resume.pdf")
## Not using because used OpenAI by default

db = FAISS.load_local("hs_resume", HuggingFaceEmbeddings(), allow_dangerous_deserialization=True)
db_retriever = db.as_retriever(search_kwargs={"k": 3})


class ChatTools():
    # @tool
    # def web_search(query : str):
    #     """Useful to perform web search"""
    #     return tavily_search.invoke({"query" : query})
    
    @tool
    def document_search(query : str):
        """Use THIS to search for queries related to applicant and his resume."""

        sim_docs = db_retriever.invoke(query)
        context = ""
        for doc in sim_docs:
            context += doc.page_content

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        chat_completion = client.chat.completions.create(
            messages = [
                {
                    "role" : "system",
                    "content" : """You are a personal VOICE assistant AI designed to help users by answering their queries accurately and efficiently. You will be provided with a user query, relevant context. Your task is to respond to the user's query by utilizing the provided context to ensure a VERY concise and relevant answer. Be clear, concise, and ensure your response aligns with the user's needs and the given information. If something is not present in the given context, respond that no cotext has been provided. DO NOT give answers from outside the context"""
                },
                {
                    "role" : "user",
                    "content" : f"""User Query : {query} \n Context : {context} """
                }
            ],
            model = "llama3-70b-8192",
            temperature=0.1,
        )
        data = {"response" : chat_completion.choices[0].message.content}

        return data
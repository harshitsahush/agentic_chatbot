from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool
# from crewai_tools import PDFSearchTool
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
import os
from dotenv import load_dotenv
from datetime import date
import redis
import json


load_dotenv()

tavily_search = TavilySearchResults()
# pdf_search_tool = PDFSearchTool(pdf = "Harshit ML Resume.pdf")
## Not using because used OpenAI by default

db = FAISS.load_local("hs_resume", HuggingFaceEmbeddings(), allow_dangerous_deserialization=True)
db_retriever = db.as_retriever(search_kwargs={"k": 5})
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
redis_client = redis.Redis(host='localhost', port=6379, db=2)




class ChatTools():  
    @tool
    def document_search(query : str):
        """Use THIS to search for queries related to applicant and his resume."""

        sim_docs = db_retriever.invoke(query)
        context = ""
        for doc in sim_docs:
            context += doc.page_content

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
        data = chat_completion.choices[0].message.content

        return data
    

    @tool
    def db_query(query : str):
        """Use this to process query when the query is related to appointment availability, appointment reservation"""


        # S1) fetch appropriate date and time
        chat_completion = client.chat.completions.create(
            messages = [
                {
                    "role" : "system",
                    "content" : """You are a helpful chatbot. From the given query you are supposed to return the mentioned date in the YYYY-MM-DD format and the hour in 24 hour time. For context, todays date in YYYY-MM-DD format is """ + str(date.today()) + """. Return ONLY the date in the specified format and nothing else. If a weekday is provided, return the current date if today is that weekday, else return the next date that occurs in the specified weekday.
                    If any of the attributes : date or time are missing, simply return empty in its place.
                    Create a json for the reponse having format:
                    {"date" : the date in YYYY-MM-DD format,"hour" : the hour in 24 hour system}
                  
                    For example: if todays date is 2024-08-21 and user query is : Are there any appointments tomorrow at 3 pm? Then your response should be:
                    {"date" : "2024-08-22","hour" : "15"}
                    """
                },
                {
                    "role" : "user",
                    "content" : f"""User Query : {query}"""
                }
            ],
            model = "llama3-70b-8192",
            temperature=0.1,
        )
        data = chat_completion.choices[0].message.content
        data = json.loads(data)
        ref_date = data["date"]
        ref_time = data["hour"]
    

        # S2) fetch todays reservations from redis db
        #     redis will store in the form {"date" : [["hour_time", "shyam"]]}
        if(redis_client.exists(ref_date)):
            booked_slots = json.loads(redis_client.get(ref_date))
        else:
            booked_slots = []


        # S3) use this data to find out what reservations are available
        chat_completion = client.chat.completions.create(
            messages = [
                {
                    "role" : "system",
                    "content" : """You are a helpful chatbot. You will be given the user query, the date user is referencing, the time user is referencing and a list of booked appointments on the referenced date. The booked appointments are in the form of list of lists. Where each element of the the parent list is a list containing 3 elements in sequence [time of appointment in 24 hour format, name of person, contact information]. for eg [[12, harshit, 12345678], [15, raman, 22314567]].
                    Generate the appropriate response for use query using ONLY the given information, and return the response.Remember that slots only exist from 10 to 17 hours.
                    If you cannot answer the response, simply return that you're unable to process the query.
                    """
                },
                {
                    "role" : "user",
                    "content" : f"""User Query : {query}. Referenced date : {ref_date}. Referenced time : {ref_time}. Booked appointments : {json.dumps(booked_slots)}"""
                }
            ],
            model = "llama3-70b-8192",
            temperature=0.1,
        )

        data = chat_completion.choices[0].message.content
        return data

# -*- coding: utf-8 -*-
"""Finalbot without hosting.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1oYIdwzEKvJ3Tihy8CUGLBEKRqfuit0mD
"""

#from google.colab import drive
#drive.mount('/content/drive')

# !pip install --upgrade google-auth-oauthlib google-auth-httplib2 google-api-python-client
# !pip install openAI
# !pip install langchain
# !pip install -U langchain-community # install the langchain-community package which contains the GoogleDriveLoader
# !pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib
# from langchain.document_loaders import GoogleDriveLoader
# from google.colab import auth
# from google.auth import default
# from googleapiclient.discovery import build
# from google_auth_oauthlib.flow import InstalledAppFlow
# from google.auth.transport.requests import Request

# !pip install python-docx

# import docx

# def read_docx(file_path):
#     doc = docx.Document(file_path)
#     content = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
#     return content

# file_path='/content/drive/MyDrive/doc_test.docx'
# content = read_docx(file_path)
# print(content)

# # Text Splitter


# from langchain.text_splitter import CharacterTextSplitter

# text_splitter = CharacterTextSplitter(separator='\n',
#                                       chunk_size=1000,
#                                       chunk_overlap=200)

# # Create a Document object
# from langchain.schema import Document
# doc = Document(page_content=content)

# docs = text_splitter.split_documents([doc]) # Pass a list of Document objects

# def store_embeddings(docs, embeddings, sotre_name, path):

#     vectorStore = FAISS.from_documents(docs, embeddings)

#     with open(f"{path}/faiss_{sotre_name}.pkl", "wb") as f:
#         pickle.dump(vectorStore, f)

# def load_embeddings(sotre_name, path):
#     with open(f"{path}/faiss_{sotre_name}.pkl", "rb") as f:
#         VectorStore = pickle.load(f)
#     return VectorStore

# from langchain.embeddings import HuggingFaceEmbeddings

# instructor_embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl",
                                                       

# !ngrok authtoken 2mIizZZ9ngWau0YE3OiGvrtiNu4_JXXyiyT2KwjCxaHNk3Fm

#!pip install fastapi uvicorn pyngrok langchain huggingface-hub nest-asyncio watchdog
import subprocess
import sys

def install_requirements():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("All dependencies installed successfully!")
    except subprocess.CalledProcessError:
        print("An error occurred while installing dependencies.")

#install_requirements()

import pickle
#!pip install faiss-cpu
# !pip install sentence-transformers
# !pip install InstructorEmbedding
# !pip install langchain-community
# !pip install openai
# !pip install tiktoken
import faiss
#from langchain.vectorstores import FAISS
import os
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
#from langchain import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import nest_asyncio
#import sentence_transformers
#import sentence_transformers.model_card


from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
from pyngrok import ngrok

import json
from datetime import datetime

import tiktoken

nest_asyncio.apply()

app = FastAPI()
print("Test 1.")

os.environ["OPENAI_API_KEY"] = "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"

user_memories = {}

file_path = '/content/user_prompts.json'
print("test 8.")
def load_vectorstore():
    with open("/home/shreyas/Desktop/Chatbot/Embeddings.pkl", "rb") as f:
        return pickle.load(f)

VectorStore = load_vectorstore()
print("test 5.")
llm = OpenAI(temperature=0.3)

chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=VectorStore.as_retriever(), verbose=True)
print("test 6.")
def get_conversation_chain(vectorstore, memory):
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Load data from JSON file (if it exists)
print("Test 2.")
def load_data():
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            # Convert the date string back to a date object
            for user_id in data:
                date_str, count = data[user_id]
                data[user_id] = (datetime.strptime(date_str, '%Y-%m-%d').date(), count)
            return data
    except FileNotFoundError:
        return {}  # Return an empty dictionary if file doesn't exist

# Save data to JSON file
def save_data(data):
    # Convert the date object to a string for JSON serialization
    data_to_save = {user_id: (date.strftime('%Y-%m-%d'), count) for user_id, (date, count) in data.items()}
    with open(file_path, 'w') as f:
        json.dump(data_to_save, f)

user_prompts = load_data()

def can_user_ask(user_id):
    today = datetime.now().date()

    # Check if the user exists in the dictionary
    if user_id in user_prompts:
        last_date, count = user_prompts[user_id]

        # Reset the prompt count if it’s a new day
        if last_date != today:
            user_prompts[user_id] = (today, 1)
            save_data(user_prompts)  # Save updated data
            return True
        elif count < 5:
            user_prompts[user_id] = (today, count + 1)
            save_data(user_prompts)  # Save updated data
            return True
        else:
            return False  # User has exceeded the limit
    else:
        # New user, initialize with 1 prompt
        user_prompts[user_id] = (today, 1)
        save_data(user_prompts)  # Save updated data
        return True

def count_tokens(prompt):
    # Initialize the encoding for the model you're using
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # or "gpt-4" based on your model
    # Encode the prompt and count the tokens
    tokens = encoding.encode(prompt)
    return len(tokens)


class QueryRequest(BaseModel):
    user_id: str
    question: str

print("Test 3.")

@app.post("/ask")
async def ask(request: QueryRequest):
    if not can_user_ask(request.user_id):
        raise HTTPException(status_code=429, detail="Prompt limit exceeded")

    user_id = request.user_id

    if user_id not in user_memories:
        user_memories[user_id] = ConversationBufferMemory(memory_key='chat_history',k=5, return_messages=True)

    memory = user_memories[user_id]
    chain = get_conversation_chain(VectorStore, memory)

    query = request.question
    if not query:
        raise HTTPException(status_code=400, detail="No question provided")

    result = chain({"question": query})
    answer = result['answer']

    return {"answer": answer + f"The number of tokens prompt was converted into is : {count_tokens(query)}."}

def reload_embeddings():
    global VectorStore, user_memories
    VectorStore = load_vectorstore()
    user_memories = {user_id: ConversationBufferMemory(memory_key='chat_history', return_messages=True) for user_id in user_memories}
    print("Embeddings reloaded.")

class EmbeddingFileHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path == "/home/shreyas/Desktop/Chatbot/Embeddings.pkl":
            reload_embeddings()

event_handler = EmbeddingFileHandler()
observer = Observer()
observer.schedule(event_handler, path='/content', recursive=False)
observer.start()

#ngrok authtoken 2mIizZZ9ngWau0YE3OiGvrtiNu4_JXXyiyT2KwjCxaHNk3Fm

print("Test 4.")

def start_api():
    port = 8000
    public_url = ngrok.connect(port).public_url
    print(f"Public URL: {public_url}")
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=port)

try:
    start_api()
except KeyboardInterrupt:
    observer.stop()

observer.join()


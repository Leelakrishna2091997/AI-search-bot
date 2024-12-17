
import streamlit as st
import os
import shutil
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.vectorstores import Chroma
from dotenv import load_dotenv


# Load environment variables from the .env file
load_dotenv()

# Access variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# Initializing the model
llm = "LLM_MODEL"

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Specify the directory to persist the database
persist_directory = "./chroma_db"

# Connect to the Pinecone index using LangChain
vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)


# Query the vector database
def retrieve_context(query):
    global vector_store
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join(doc.page_content for doc in docs)
    return context


# Define the prompt template
def format_prompt(context, question):
    return f"""
Use the context below to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

def response_with_context(query):
    # Retrieve context from Pinecone
    context = retrieve_context(query)
    formatted_prompt = format_prompt(context, query)

    global llm
    response = llm.invoke(formatted_prompt)
    return response 

def response_without_context(query):
    global llm
    response = llm.invoke(query)
    return response


def summarize_context(query):
    
    withContext = response_with_context(query)
    withoutContext = response_without_context(query)
    
    combined_response = (
    f"You have two responses: 1. A response with specific context (70% weight): {withContext.content}"
    f"2. A general response based on general knowledge (30% weight): {withoutContext.content}."
    f"Your task is to synthesize, merge these responses into a single, concise, and coherent answer."
    f"for the query: {query}.")
    
    print(combined_response, "combined")
    global llm
    return llm.invoke(combined_response)

def main():

    global llm
    # Initialize OpenAI Chat Model
    llm = ChatOpenAI(temperature=0.7)

    # Streamlit Application
    st.title("Interactive OpenAI Chatbot")
    st.subheader("Ask me anything!")

    # Session state to store conversation history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display the chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Type your question here..."):

        


        # Add the user query to the conversation
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response from the OpenAI model
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                # Retrieve context and generate response
                response = summarize_context(prompt)


            st.markdown(response.content)
            st.session_state.messages.append({"role": "assistant", "content": response})


# running streamlit
main()
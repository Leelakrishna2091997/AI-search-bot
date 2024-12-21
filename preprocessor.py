from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv


# Load environment variables from the .env file
load_dotenv()

# Access variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

persist_directory = "./chroma_db"
embeddings = OpenAIEmbeddings()


def preprocessing():
    print("calling preprocessor")

    # PDF Preprocessing
    def process_pdf(file_path):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        split_docs = text_splitter.split_documents(documents)
        return split_docs

    # Text Preprocessing (e.g., Swagger/Postman files)
    def process_text(file_path):
        loader = TextLoader(file_path)
        documents = loader.load()
        split_docs = text_splitter.split_documents(documents)
        return split_docs

    # Add documents to vectorstore
    def add_to_vectorstore(docs):
        global vector_store
        vector_store.add_documents(docs)

    
    
    global vector_store

    # Connect to the Pinecone index using LangChain
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    directories = ["binance", "coinbase", "coinmarketcap", "robin"]
    base_path = "/Users/leelakrishna/Desktop/ai-docs/AI-search-bot/documents/v2"

    # Iterate through each directory
    for each_directory in directories:
        directory_path = os.path.join(base_path, each_directory)
        
        # Check if directory exists
        if not os.path.exists(directory_path):
            print(f"Directory not found: {directory_path}")
            continue

        # Process each file in the directory
        for file_name in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file_name)
            
            # Skip if not a file
            if not os.path.isfile(file_path):
                continue

            print("processing file", file_path)
            
            # Determine the file type and process accordingly
            if file_name.endswith(".pdf"):
                processed = process_pdf(file_path)
            elif file_name.endswith(".json") or file_name.endswith(".txt"):
                processed = process_text(file_path)
            else:
                print(f"Skipping other file types: {file_path}")
                processed = None
            
            # Adding to vector store
            if processed:
                add_to_vectorstore(processed)


if __name__ == "__main__":

    # calling preprocessing step
    preprocessing()



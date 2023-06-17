from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI 
from langchain.chains import ConversationalRetrievalChain
import os
import pinecone

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_ENV = os.environ.get('PINECONE_ENV')

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)
index_name = "project2"

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

vectorstore = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)

qa = ConversationalRetrievalChain.from_llm(
    llm=OpenAI(temperature=0, openai_api_key = OPENAI_API_KEY), 
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
)

chat_history = []

def load_pdfs_to_pinecone(directory: str):
    """
    Load all PDFs in a directory to a Pinecone index.
    """
    for file in os.listdir(directory):
        if (file.endswith(".pdf")):
            loader = PyPDFLoader(file)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=0
            )
            texts = splitter.split_documents(loader.load())
            print(texts)
            Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)

def query_documents(query: str):
    """
    Query the Pinecone index for documents that match the query.
    """
    result = qa({'question': query, 'chat_history': chat_history})
    chat_history.append((query, result['answer']))
    return result

print(query_documents("What is generalization?")['answer'])
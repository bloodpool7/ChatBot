from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI 
from langchain.chains.question_answering import load_qa_chain
import os
import pinecone

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_ENV = os.environ.get('PINECONE_ENV')

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)
index_name = "INDEX_NAME"

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

vectorstore = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)

llm = OpenAI(temperature = 0, openai_api_key = OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type = "stuff")

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
            texts = splitter.split(loader.load())
            Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)

def query_documents(query: str):
    """
    Query the Pinecone index for documents that match the query.
    """
    doc = vectorstore.similarity_search(query)
    result = chain.run(input_documents = doc, question = query) 
    return result

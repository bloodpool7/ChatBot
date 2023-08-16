from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI 
from langchain.chains import ConversationalRetrievalChain
from Crypto.Hash import SHA256
import PyPDF2
import pinecone
import os


def add_pdfs(files: list, namespace: str, openai_api_key: str = None, pinecone_api_key: str = None, pinecone_env: str = None, index_name: str = None):
    """
    Load all given PDFs to a Pinecone index.
    """
    pinecone.init(
        api_key=pinecone_api_key,
        environment=pinecone_env
    )

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=20
            )
    
    if (type(files[0]) == str):
        for file in files:
            loader = PyPDFLoader(file)
            texts = splitter.split_documents(loader.load())
            Pinecone.from_texts(
                [t.page_content for t in texts], 
                embeddings, 
                index_name=index_name, 
                metadatas = [{'id': hash_pdf(open(file, "rb").read()), "title" : file} for t in texts], 
                namespace = namespace
            )
    else:
        for file in files:
            reader = PyPDF2.PdfReader(file)
            extracted_text = ""

            for i in range(len(reader.pages)):
                page = reader.pages[i]
                extracted_text += page.extract_text()
            
            texts = splitter.split_documents([Document(page_content=extracted_text)])
            Pinecone.from_texts(
                [t.page_content for t in texts], 
                embeddings, 
                index_name=index_name, 
                metadatas = [{'id': hash_pdf(file.getvalue()), "title" : file.name} for t in texts], 
                namespace = namespace
            )

            
def delete_pdfs(files: list[str] = [], is_id = False, namespace: str = None, pinecone_api_key: str = None, pinecone_env: str = None, index_name: str = None):
    """
    Delete all given PDFs from a Pinecone index.
    """
    pinecone.init(
        api_key=pinecone_api_key,
        environment=pinecone_env
    )
    index = pinecone.Index(index_name=index_name)
    
    if len(files) == 0:
        index.delete(delete_all = True, namespace = namespace)
        return

    if is_id:
        for file in files:
            index.delete(filter = {"id": file}, namespace = namespace)
        return

    if (type(files[0]) == str):
        for file in files:
            index.delete(filter = {"title": file}, namespace = namespace)
    else:
        for file in files:
            index.delete(filter = {"title": file.name}, namespace = namespace)


def update_pdfs(files: list[str], namespace: str, openai_api_key: str = None, pinecone_api_key: str = None, pinecone_env: str = None, index_name: str = None):
    """
    Update all given PDFs in a Pinecone index.
    """
    delete_pdfs(files, namespace, pinecone_api_key, pinecone_env, index_name)
    add_pdfs(files, namespace, openai_api_key, pinecone_api_key, pinecone_env, index_name)


def query_documents(query: str, namespace: str, filter: dict = {}, openai_api_key: str = None, pinecone_api_key: str = None, pinecone_env: str = None, index_name: str = None, chat_history: list = []):
    """
    Query the Pinecone index for documents that match the query.
    """
    pinecone.init(
        api_key=pinecone_api_key,
        environment=pinecone_env
    )
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    vectorstore = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings, namespace=namespace)

    qa = ConversationalRetrievalChain.from_llm(
        llm=OpenAI(temperature=0, openai_api_key=openai_api_key), 
        retriever=vectorstore.as_retriever(search_kwargs={'filter': filter}),
        return_source_documents=True,
    )

    result = qa({'question': query, 'chat_history': chat_history})
    return result

def hash_pdf(pdf):
    """
    Hash a PDF using SHA256.
    """
    hasher = SHA256.new(pdf)
    hashed = hasher.digest()
    return hashed.hex()


if __name__ == "__main__":
    pinecone_api_key = os.environ["PINECONE_API_KEY"]
    openai_api_key = os.environ["OPENAI_API_KEY"]
    pinecone_env = os.environ["PINECONE_ENV"]

    print(query_documents("What is supervised learning?", "testtwo", openai_api_key = openai_api_key, pinecone_api_key = pinecone_api_key, pinecone_env=pinecone_env, index_name = "test"))
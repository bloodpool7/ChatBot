import pinecone
import os
import openai
import PyPDF2

def add_to_database(ailment):
    PINECONE_ENV = os.getenv("PINECONE_ENV")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

    pinecone.init(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_ENV
    )
    index = pinecone.Index("test")

    result = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo", 
        messages = [{"role": "user", "content" : "What are commonly observable symptoms of " + ailment + " ?"}]
    )["choices"][0]['message']['content']

    result_embedding = openai.Embedding.create(
        input=result,
        model="text-embedding-ada-002"
    )['data'][0]['embedding']

    index.upsert([(ailment, result_embedding)], namespace = "sideproject")

def classify(query):
    PINECONE_ENV = os.getenv("PINECONE_ENV")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

    pinecone.init(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_ENV
    )
    index = pinecone.Index("test")

    outputs = index.query(
        vector = openai.Embedding.create(input = query, model = "text-embedding-ada-002")['data'][0]['embedding'], 
        top_k = 5,
        namespace = 'sideproject',
        include_values = True
    )

    return outputs

def get_text_from_pdf(pdf):
    reader = PyPDF2.PdfReader(pdf)
    extracted_text = ""

    for i in range(len(reader.pages)):
        page = reader.pages[i]
        extracted_text += page.extract_text()    
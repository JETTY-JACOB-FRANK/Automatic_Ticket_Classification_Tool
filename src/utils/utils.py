# src/utils/utils.py

from pinecone import Pinecone as PineconeClient
from langchain_community.vectorstores import Pinecone
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

def pull_from_pinecone(pinecone_apikey, pinecone_environment, pinecone_index_name, embeddings):
    """Pull relevant documents from Pinecone index."""
    PineconeClient(api_key=pinecone_apikey, environment=pinecone_environment)
    index_name = pinecone_index_name
    index = Pinecone.from_existing_index(index_name, embeddings)
    return index

def create_embeddings():
    """Create embeddings using a pre-trained model."""
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings

def get_similar_docs(index, query, k=2):
    """Retrieve similar documents from the index based on user query."""
    similar_docs = index.similarity_search(query, k=k)
    return similar_docs

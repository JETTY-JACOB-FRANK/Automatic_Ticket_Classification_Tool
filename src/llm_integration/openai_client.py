# src/llm_integration/openai_client.py

from langchain_openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks import get_openai_callback

def generate_answer(docs, user_input):
    """Use OpenAI to generate an answer from relevant documents."""
    chain = load_qa_chain(OpenAI(), chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=user_input)
    return response

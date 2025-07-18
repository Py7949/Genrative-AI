from langchain_community.chat_models import ChatHuggingFace
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()


hf_token = os.getenv("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")

client = InferenceClient(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1", 
    token=hf_token
)

llm = ChatHuggingFace(client=client)

response = llm.invoke("What is the capital of India?")
print("Response:", response.content)

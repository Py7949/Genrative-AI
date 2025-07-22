from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()
pipe = pipeline("text-generation", model="google/flan-t5-base",max_new_tokens = 100)
model = HuggingFacePipeline(pipeline=pipe)


messages = [
    SystemMessage(content='you are a helpful assistant'),
    HumanMessage(content='tell me about Langchain')
]
result = model.invoke(messages)

messages.append(AIMessage(content=result))

print(messages)
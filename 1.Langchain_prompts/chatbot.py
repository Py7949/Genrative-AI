from transformers import pipeline
from langchain.llms import HuggingFacePipline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

pipe = pipeline("text2text-generation", model="google/flan-t5-base")

llm = HuggingFacePipline(pipeline=pipe)

prompt = PromptTemplate(
    input_variables=['question'],
    template='You are a helpful assistant.Answer this question : {question}'
)

chain = LLMChain(llm=llm, prompt=prompt)

while True:
    user_input = ('You: ')
    if user_input.lower() in ['exist', 'quit']:
        break
    response = chain.run(user_input)
    print("Bot:", response)

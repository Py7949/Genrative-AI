from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain_core.language_models import BaseLanguageModel


text_generator = pipeline("text-generation", model="gpt2", max_new_tokens=100)

model = HuggingFacePipeline(pipeline=text_generator)
chat_history = [
    SystemMessage(content='You are a helpful AI assistant.')
]

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    chat_history.append(HumanMessage(content=user_input))

    
    formatted_prompt = "\n".join([
        f"User: {m.content}" if isinstance(m, HumanMessage) else
        f"Assistant: {m.content}" if isinstance(m, AIMessage) else
        f"{m.content}" for m in chat_history
    ]) + "\nAssistant:"

    result = model.invoke(formatted_prompt)
    chat_history.append(AIMessage(content=result))
    print("AI:", result)


print("\n--- Chat History ---")
for msg in chat_history:
    role = msg.__class__.__name__.replace("Message", "")
    print(f"{role}: {msg.content}")
 
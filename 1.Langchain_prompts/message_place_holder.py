from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_templates = ChatPromptTemplate.from_messages([
    ("System",'You are a helpful customer Support agent'),
    MessagesPlaceholder(variable_name='chat_history')('human', '{query}')
])

chat_history = []

with open('chat_history.txt') as f:
    chat_history = f.readlines()

print(chat_history)


prompt = chat_templates.invoke({'chat_history':chat_history, 'query':'Where is my refund'})

print(prompt)
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()



llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generated"
)


model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template='Write a detailed report on{topic}',
    input_variable=['topic']
)

template2 = PromptTemplate(
    template='Write a 5 line summary on thr following./n{text}',
    input_variables=['text']
)

prompt1 = template1.invoke({'topic': 'Self Attention'})
result = model.invoke(prompt1)

prompt2 = template2.invoke({'text':result.content})
result1 = model.invoke(prompt2)

print(result1.content)

from langchain_community.document_loaders import TextLoader 
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/QwQ-32B",
    task="task-generation"
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template='write the summary about the following text -\n{text}',
    input_variables=['text']
)

parser = StrOutputParser()

loader = TextLoader('prompting.txt')

docs = loader.load()

print(type(docs))

print(len(docs))

print(type(docs[0]))

print(docs[0].page_content)

print(docs[0].metadata)

chain = prompt | model |parser

print(chain.invoke({'text':docs[0].page_content}))
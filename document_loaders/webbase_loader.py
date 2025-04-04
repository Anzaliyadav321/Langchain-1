from langchain_community.document_loaders import WebBaseLoader
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
    template='Answer the following question \n {question} from the foolowing text -\n{text}',
    input_variables=['question','text']
)

parser = StrOutputParser()

url='https://www.flipkart.com/apple-macbook-air-m2-16-gb-256-gb-ssd-macos-sequoia-mc7x4hn-a/p/itmdc5308fa78421'
loader=WebBaseLoader(url)
docs = loader.load()
chain = prompt | model | parser

print(chain.invoke({'question':'What is the prodcut that we are talking about?', 'text':docs[0].page_content}))



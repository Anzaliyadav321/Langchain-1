from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='Qwen/QwQ-32B',
    task='text-generated'
)


model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()


template = PromptTemplate(
    template='Give me 5 fact about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions}
)

chain = template | model |parser

result = chain.invoke({'topic':'self Attention'})

print(result)

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= 'Qwen/QwQ-32B',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name='fact_1', descripton='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', descripton='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', descripton='Fact 3 about the topic')
]

parser = StructuredOutputParser.from_response_schemas(schema)

template= PromptTemplate(
    template='Give 3 fact about {topic} \n {format_instructions}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_fromat_instructions()}

)

chain = template | model |parser

result = model.invoke({'topic':'tarrif'})

print(result)
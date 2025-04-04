from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('attention is all you need.pdf')

docs = loader.load()

print(len(docs))

print(docs[0].page_content)

print(docs[3].metadata)
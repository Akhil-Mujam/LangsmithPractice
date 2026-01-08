# pip install -U langchain langchain-community langchain-aws faiss-cpu pypdf python-dotenv sentence-transformers

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser

os.environ['LANGCHAIN_PROJECT'] =  "LANGCHAIN_RAG_Project"

load_dotenv()

PDF_PATH = "islr.pdf"

# 1) Load PDF
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

# 2) Chunk
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)
splits = splitter.split_documents(docs)

# 3) Sentence Transformer embeddings (LOCAL)
embeddings = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vs = FAISS.from_documents(splits, embeddings)

# 4) Retriever (NO CHANGE needed)
retriever = vs.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# 5) Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

# 6) LLM with temperature = 0.3
llm = ChatBedrock(
    model_id=os.getenv('MODEL_ID'),
    region_name=os.getenv("AWS_REGION"),
    model_kwargs={
        "temperature": 0.3
    }
)

# Helper to format docs
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# 7) Parallel input builder
parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

# 8) Full chain
chain = parallel | prompt | llm | StrOutputParser()

# 9) Ask questions
print("PDF RAG ready. Ask a question (Ctrl+C to exit).")
q = input("\nQ: ")
ans = chain.invoke(q.strip())
print("\nA:", ans)

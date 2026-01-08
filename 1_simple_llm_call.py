from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_aws import ChatBedrock
import os

load_dotenv()

prompt = PromptTemplate.from_template("{question}")

model = ChatBedrock(
    model_id=os.getenv('MODEL_ID'),
    region_name=os.getenv("AWS_REGION"),
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"question": "What is ERP"})
print(result)

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_aws import ChatBedrock
import os

load_dotenv()

os.environ['LANGCHAIN_PROJECT'] =  "sequential_chain"

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)


model = ChatBedrock(
    model_id = os.getenv('MODEL_ID'),
    region_name=os.getenv("AWS_REGION"),
)

config  = {
    'run_name':'sequential_chain',
    'tags': ['llm app', 'report generation'],
    'metadata':{'model': "chat-bedrock",'temp':'didnt set'}
}

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic': 'How to webceawl a page?'}, config = config)

print(result)

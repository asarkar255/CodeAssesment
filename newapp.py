from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_anthropic import ChatAnthropic
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langserve import add_routes
import os
from pydantic import BaseModel,json
from dotenv import load_dotenv
load_dotenv()

SECRET_KEY = os.getenv("ANTHROPIC_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]= os.getenv("LANGCHAIN_API_KEY")
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

#Load custom Documents
loader = TextLoader('./ruleset.txt')


documents = loader.load()


text_splitter = CharacterTextSplitter(chunk_size=500000, chunk_overlap=50)

texts = text_splitter.split_documents(documents)

vectorstore = Chroma.from_documents(documents=texts, embedding=OpenAIEmbeddings())
# Retrieve and generate using the relevant snippets 
retriever = vectorstore.as_retriever()
#1. Create prompt template
system_template = "You are an SAP ABAP Remediation Assistant ,Asess Code,use context: {context}"

#system_template = "Act as an expert SAP ECC to SAP S4 HANA code remediator,return only Remediated Code"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])



# # # 2. Create model
# llm = ChatAnthropic(
#     model="claude-3-7-sonnet-20250219",
#     temperature=0,
#     max_tokens=20000,
#     timeout=None,
#     api_key=SECRET_KEY,
# )

llm = ChatOpenAI(model="gpt-4o", temperature=0)
# # # 3. Create parser
parser = StrOutputParser()

# # # 4. Create chain
# chain = prompt_template | llm | parser
chain = ( {"context": retriever , "text": RunnablePassthrough()} | prompt_template | llm | parser )


# 4. App definition
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
 )

class RequestBody(BaseModel):
    text: str

@app.post("/remediate_code")
async def remediate_code(request_body: RequestBody):
    # 5. Run the chain and return the result
    result =  chain.invoke(request_body.text)
    return {"result": result}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)

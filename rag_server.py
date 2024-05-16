from langchain_community.vectorstores import Qdrant # used for creating the Qdrant Vector Object
from langchain_openai import OpenAIEmbeddings # used for embeddings 
from langchain_openai import ChatOpenAI # used for /chat/completions
from langchain_core.output_parsers import StrOutputParser # used for LCEL 
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_core.runnables import ConfigurableField # new update

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel
import io
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient
import uvicorn
import dotenv
import os

app = FastAPI()

origins = ["*"]

dotenv.load_dotenv()

os.environ["OPENAI_API_KEY"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    query: str

llm_4_00 = ChatOpenAI(model_name="gpt-4-turbo", temperature=0, max_tokens=200).configurable_fields(
    max_tokens=ConfigurableField(
        id="output_token_number",
        name="Max tokens in the output",
        description="The maximum number of tokens in the output",
    )
)

llm_35_00 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
#llm_4_00 = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0)
embeddings = OpenAIEmbeddings()
qclient = QdrantClient(
    url=os.getenv("QDRANT_HOST"),
    api_key=os.getenv("QDRANT_API_KEY")
)
vectorstore = Qdrant(client = qclient, collection_name = "knowledge", embeddings = embeddings)
retriever_2 = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
########################################################################################################################################

############### RAG CHAIN ############### --- The RAG works.
rag_chain = (
    {
        "context": itemgetter("question") | retriever_2,
        "question": itemgetter("question"),
    }
    | ChatPromptTemplate.from_template("""You are a helpful Assistant. KEEP YOUR REPLIES SHORT WITHIN 3 SENTENCES
                                       
    Context:
{context}

Question: {question}
"""
) 
    | llm_35_00
    | StrOutputParser()
)

def query(prompt):
    return rag_chain.invoke({"question":str(prompt)})

def query_response(prompt):
    for chunk in rag_chain.stream({"question":str(prompt)}):
        yield chunk or ""
        
@app.post("/llm-stream/")
async def stream_llm_response(query: Query):
    response_stream = query_response(query.query)
    return StreamingResponse(response_stream, media_type="text/plain")

@app.post("/llm-query/")
async def stream_llm_response(query: Query):
    response = query_response(query.query)
    sentence = ' '.join(filter(None, response))
    return sentence

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)
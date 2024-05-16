from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnablePassthrough , RunnableParallel
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain_community.vectorstores import Qdrant
from qdrant_client import models, QdrantClient
import qdrant_client
import os
import dotenv

dotenv.load_dotenv()

os.environ["OPENAI_API_KEY"]

client = qdrant_client.QdrantClient(
    url=os.getenv("QDRANT_HOST"),
    api_key=os.getenv("QDRANT_API_KEY")
)

doc ="data.txt"
data=""

with open(doc,'r') as f:
    data = f.read()
    
def get_chunks(text):
    text_splitter=CharacterTextSplitter(
        separator= "\n",
        chunk_size=400,
        chunk_overlap=100,
        length_function=len
    )

    chunks=text_splitter.split_text(text)
    return chunks

texts=get_chunks(data)

vectors_config=models.VectorParams(
    # depends on model, we can google dimension. 1536 for openai
    # we are using openai embedding, for that size is 1536
    size=1536,
    distance=models.Distance.COSINE)

client.create_collection(
    collection_name="knowledge",
    vectors_config=vectors_config,
)

embeddings = OpenAIEmbeddings()

vector_store = Qdrant(
    client=client,
    collection_name="knowledge",
    embeddings=embeddings,
)

vector_store.add_texts(texts)
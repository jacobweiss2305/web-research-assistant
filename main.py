# Agent Tools
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool
from langchain.tools import DuckDuckGoSearchRun
from utils import WebpageQATool, load_qa_with_sources_chain
from langchain import HuggingFacePipeline
from langchain.experimental import AutoGPT

# GPU
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM

# Memory tools
import faiss
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from Embedding import HuggingFaceEmbedding

REPO_ID = "google/flan-t5-xxl"

# Local memory setup
embeddings_model = HuggingFaceEmbedding.newEmbeddingFunction
embedding_size = 1000
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model, index, InMemoryDocstore({}), {})

# Agent setup
search = DuckDuckGoSearchRun()
model_id = 'google/flan-t5-large'#
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True)

pipe = pipeline(
    "text2text-generation",
    model=model, 
    tokenizer=tokenizer, 
    max_length=100
)

llm = HuggingFacePipeline(pipeline=pipe)

query_website_tool = WebpageQATool(qa_chain=load_qa_with_sources_chain(llm))

tools = [
    search,
    WriteFileTool(root_dir="./data"),
    ReadFileTool(root_dir="./data"),
    query_website_tool,
]

agent = AutoGPT.from_llm_and_tools(
    ai_name="Tom",
    ai_role="Assistant",
    tools=tools,
    llm=llm,
    memory=vectorstore.as_retriever(),
)

weather_prompt = "Create a weather report for Westcliffe Colorado"

agent.run([weather_prompt,])

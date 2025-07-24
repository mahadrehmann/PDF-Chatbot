import os
import logging
from dotenv import load_dotenv
import openai

# ✅ Up-to-date imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI  # only for memory prompts; Gemini is used below
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.chat_models import ChatOpenAI
from langchain.memory.buffer import ConversationBufferMemory
from langchain_core.callbacks.manager import trace_as_chain_group
from langchain.callbacks.tracers import LangChainTracer
from langchain.evaluation import load_evaluator
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Logging setup ---
logging.basicConfig(
    filename="stage2.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filemode="a"
)
logger = logging.getLogger()

# --- Env & API keys ---
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")
gemini_key = os.getenv("GEMINI_API_KEY")

# --- Load & index PDF ---
loader = PyPDFLoader("data/example.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
logger.info(f"Indexed {len(chunks)} document chunks.")

# --- LLM & Memory setup ---
gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_key,
    temperature=0.3
)

memory = ConversationBufferMemory(
    memory_key="history",
    input_key="query",
    output_key="answer",
    return_messages=True
)
# --- Create retrieval chain ---
# Use a prompt template for combining documents
from langchain_core.prompts import ChatPromptTemplate
system = "Use the context to answer the user question. If unknown, say you don't know."
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "{query}")
])
combine_chain = create_stuff_documents_chain(llm=gemini, prompt=chat_prompt)

# Build retrieval chain as before
qa_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=combine_chain
).with_memory(memory)


# --- Tracing setup ---
tracer = LangChainTracer()

# --- 1️⃣ Sample Dialogue to test memory & retrieval ---
for user in ["Hi, I like cats.", "What do I like?", "Tell me about cats from the document."]:
    with trace_as_chain_group("qa_run", inputs={"query": user}) as cb_mgr:
        result = qa_chain.invoke({"query": user}, callbacks=[cb_mgr])
    answer = result["answer"]
    logger.info(f"User: {user} → AI: {answer}")
    print("AI:", answer)

# --- 2️⃣ Evaluation via QAEvalChain ---
evaluator = load_evaluator("qa")
examples = [
    {"query": "What do I like?", "answer": "cats"},
    {"query": "Tell me about cats from the document.", "answer": "<expected factual snippet>"}
]
predictions = [
    {"query": ex["query"], "result": qa_chain.invoke({"query": ex["query"]})["answer"]}
    for ex in examples
]
eval_res = evaluator.evaluate(examples=examples, predictions=predictions)
logger.info(f"Evaluation results: {eval_res}")
print("Evaluation:", eval_res)

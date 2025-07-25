# import os, logging
# from dotenv import load_dotenv

# import openai
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.memory.buffer import ConversationBufferMemory
# from langchain.callbacks.tracers import LangChainTracer
# from langchain_core.callbacks.manager import trace_as_chain_group
# from langchain.chains import ConversationalRetrievalChain
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.evaluation import load_evaluator
# from langchain_google_genai import ChatGoogleGenerativeAI

# # ── Environment ────────────────────────────────────────────────────────────────
# load_dotenv()
# # Enable LangChain v2 tracing, which will send spans to LangSmith
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGSMITH_API_KEY"]    = os.getenv("LANGSMITH_API_KEY")
# openai.api_key                     = os.getenv("OPENAI_API_KEY")

# # ── Build PDF + Retriever ──────────────────────────────────────────────────────
# loader   = PyPDFLoader("data/example.pdf")
# docs     = loader.load()
# splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# chunks   = splitter.split_documents(docs)

# embeddings  = OpenAIEmbeddings()
# vectorstore = FAISS.from_documents(chunks, embeddings)
# retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})

# # ── LLM + Memory ────────────────────────────────────────────────────────────────
# gemini = ChatGoogleGenerativeAI(
#     model          = "gemini-2.5-flash",          # must be `model`, not `model_name`
#     google_api_key = os.getenv("GEMINI_API_KEY"),
#     temperature    = 0.2
# )

# memory = ConversationBufferMemory(
#     memory_key     = "chat_history",
#     input_key      = "question",
#     output_key     = "answer",
#     return_messages= True
# )

# # ── Condense‐question prompt ────────────────────────────────────────────────────
# condense_prompt = ChatPromptTemplate.from_messages([
#     ("system", "Rewrite follow-up question as standalone."),
#     (MessagesPlaceholder(variable_name="chat_history")),
#     ("user", "{question}")
# ])

# # ── Tracer & Chain ──────────────────────────────────────────────────────────────
# tracer   = LangChainTracer()
# qa_chain = ConversationalRetrievalChain.from_llm(
#     llm                       = gemini,
#     retriever                 = retriever,
#     memory                    = memory,
#     condense_question_prompt  = condense_prompt,
#     get_chat_history          = lambda h: h,  # pass raw BaseMessage list
#     return_source_documents   = True,
#     callbacks                 = [tracer],     # attach tracer
# )

# # ── Run Dialogue with Grouped Traces ────────────────────────────────────────────
# # dialogue = ["Hi, I like cats.", "What do I like?", "Tell me about cats from the document."]
# dialogue = ["Hi, I like C++.", "What do I like?", "Tell me about web projects from the document."]

# for user in dialogue:
#     # NOTE: no `tracer=` kwarg here
#     with trace_as_chain_group("conv_qa", inputs={"question": user}) as cb_mgr:
#         # Combine both tracer and the group manager
#         result = qa_chain.invoke(
#             {"question": user},
#             callbacks=[tracer, cb_mgr]
#         )
#     print("AI:", result["answer"])

# # ── Automated QA Evaluation ─────────────────────────────────────────────────────
# evaluator   = load_evaluator("qa")
# examples    = [{"query": "What do I like?", "answer": "cats"}]
# predictions = [
#     {"query": ex["query"], "result": qa_chain.invoke({"question": ex["query"]})["answer"]}
#     for ex in examples
# ]
# eval_res = evaluator.evaluate(examples=examples, predictions=predictions)
# print("Evaluation:", eval_res)

import os, logging
from dotenv import load_dotenv
import openai

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory.buffer import ConversationBufferMemory
from langchain.callbacks.tracers import LangChainTracer
from langchain_core.callbacks.manager import trace_as_chain_group
from langchain.chains.retrieval import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai.chat_models import ChatOpenAI

# — Logging setup —
logging.basicConfig(filename="stage2.log", level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()

# — Load environment —
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

# — Load PDF & embed —
loader = PyPDFLoader("data/example.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# — LLM & memory setup —
llm = ChatOpenAI(
    model="gpt-4o-mini-2024-07-18",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.3,
    max_tokens=150
)
memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question",
    output_key="answer",
    return_messages=True
)

# — Standalone question prompt —
condense_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given chat history and the new question, rewrite it as a standalone question."),
    (MessagesPlaceholder(variable_name="chat_history")),
    ("user", "{question}")
])

# — QA prompt combining context —
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using the provided context."),
    ("assistant", "{context}"),
    ("user", "{question}")
])
combine_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt)

# — Build retrieval chain with memory and history-aware retriever —
history_aware = create_history_aware_retriever(llm=llm, retriever=retriever, rephrase_prompt=condense_prompt)
qa_chain = create_retrieval_chain(retriever=history_aware, combine_docs_chain=combine_chain, memory=memory, return_source_documents=True)

# — Tracer setup —
tracer = LangChainTracer()

# — Conversation turns —
dialogue = ["Hi, I like C++.", "What do I like?", "Tell me about web projects from the document."]
for user in dialogue:
    with trace_as_chain_group("conv_qa", inputs={"question": user}) as cb_mgr:
        result = qa_chain.invoke({"question": user}, callbacks=[cb_mgr])
    print("AI:", result["answer"])
    logger.info(f"User: '{user}' → AI: '{result['answer']}'")

# — Done —
print("Conversation done.")

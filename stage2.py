# from dotenv import load_dotenv
# import os
# import logging
# from stage1 import validate_input  # Import your stage 1 checker

# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores.faiss import FAISS
# from langchain.schema import Document
# from langchain.agents import Tool, create_react_agent, AgentExecutor
# from langchain_core.prompts import PromptTemplate
# from langsmith import traceable
# from langchain_core.tracers.langchain import wait_for_all_tracers


# # Load environment
# load_dotenv()
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# # Configure logging
# logging.basicConfig(filename="agent.log", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
# logger = logging.getLogger()

# # Define PDF RAG tool
# @traceable(name="pdf_rag_tool")
# def pdf_rag_tool(query: str) -> str:
#     loader = PyPDFLoader("data/example2.pdf")
#     pages = loader.load()
#     full_text = "\n".join(p.page_content for p in pages)
#     splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", "."], chunk_size=200, chunk_overlap=50)
#     chunks = splitter.split_text(full_text)
#     docs = [Document(page_content=chunk) for chunk in chunks]
#     embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
#     vectorstore = FAISS.from_documents(docs, embed_model)
#     matches = vectorstore.similarity_search(query, k=3)
#     return "\n\n---\n\n".join(doc.page_content for doc in matches)

# # Wrap tool as Agent Tool
# pdf_tool = Tool(
#     name="pdf_search",
#     func=pdf_rag_tool,
#     description="Retrieve relevant context from the PDF by similarity search."
# )

# # Initialize LLM
# llm = ChatOpenAI(
#    model="gpt-4o-mini-2024-07-18",
#    temperature=0.3,
#    max_tokens=100
# )

# template = """
#     You are an agent with access to the tool {tools}.
#     Follow this format exactly:

#     Question: {input}
#     Thought: think about what to do next
#     Action: one of [{tool_names}]
#     Action Input: input for the action
#     Observation: result of the action

#     (Repeat Thought/Action/Action Input/Observation as needed)

#     If you find even a partial answer:
#     Thought: I will return the most relevant information I found
#     Final Answer: the information

#     Begin!
#     Question: {input}
#     Thought:{agent_scratchpad}
# """

# prompt = PromptTemplate.from_template(template)

# agent = create_react_agent(llm=llm, tools=[pdf_tool], prompt=prompt)

# agent_executor = AgentExecutor.from_agent_and_tools(
#     agent=agent,
#     tools=[pdf_tool],
#     verbose=True,
#     handle_parsing_errors=True,
#     max_iterations=3  # <-- add this
# )

# def handle_query(question: str) -> str:
#     logger.info(f"User input: {question}")
    
#     # Stage 1 moderation check
#     result = validate_input(question)

#     if result["status"] == "unsafe":
#         logger.warning(f"Moderation flagged input under: {result.get('categories', result.get('judgment'))}")
#         return f"❌ Input flagged as unsafe. Categories: {result.get('categories', result.get('judgment'))}"

#     if result["status"] == "ambiguous":
#         logger.warning(f"Ambiguous input: {result['judgment']}")
#         return f"⚠️ Ambiguous input: {result['judgment']}"

#     logger.info("✅ Input passed moderation and ambiguity check.")
    
#     # Stage 2 RAG Agent
#     response = agent_executor.invoke({"input": question})
#     return response

# if __name__ == "__main__":
#     while True:
#         question = input("\nEnter your query (type 'exit' to quit): ")
#         if question.lower() == "exit":
#             break
#         answer = handle_query(question)
#         print("\nAnswer:\n", answer)

#     wait_for_all_tracers()

# stage2.py
from dotenv import load_dotenv
import os, logging

from langsmith import traceable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.schema import Document
from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.tracers.langchain import wait_for_all_tracers

# --- Environment & Logging ---
load_dotenv()
os.environ["OPENAI_API_KEY"]    = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

logging.basicConfig(
    filename="agent.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- PDF Retrieval Tool ---
@traceable(name="pdf_rag_tool")
def pdf_rag_tool(query: str) -> list[str]:
    loader = PyPDFLoader("data/example2.pdf")
    pages = loader.load()
    text = "\n".join(p.page_content for p in pages)
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", "."],
        chunk_size=200,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    embed_model  = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore  = FAISS.from_documents(docs, embed_model)
    results      = vectorstore.similarity_search(query, k=3)
    # return raw chunks for downstream checks
    return [doc.page_content for doc in results]

pdf_tool = Tool(
    name="pdf_search",
    func=lambda q: "\n\n---\n\n".join(pdf_rag_tool(q)),
    description="Retrieve relevant context from the PDF by similarity search."
)

# --- Agent Setup ---
llm = ChatOpenAI(
    model="gpt-4o-mini-2024-07-18",
    temperature=0.3,
    max_tokens=100
)

template = """
You are an agent with access to the tool {tools}.
Follow this format exactly:

Question: {input}
Thought: think about what to do next
Action: one of [{tool_names}]
Action Input: input for the action
Observation: result of the action

(Repeat Thought/Action/Action Input/Observation as needed)

If you find even a partial answer:
Thought: I will return the most relevant information I found
Final Answer: the information

Begin!
Question: {input}
Thought:{agent_scratchpad}
"""
prompt = PromptTemplate.from_template(template)

agent = create_react_agent(llm=llm, tools=[pdf_tool], prompt=prompt)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=[pdf_tool],
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3
)

# --- Public API ---
def run_rag_agent(question: str) -> tuple[str, list[str]]:
    """
    Runs the RAG agent over the PDF.
    Returns (final_answer, retrieved_chunks).
    """
    logger.info(f"Stage2 – running RAG agent for question: {question!r}")
    # retrieve context first
    context_chunks = pdf_rag_tool(question)
    logger.info(f"Stage2 – retrieved {len(context_chunks)} chunks")
    # run agent to get final answer
    result = agent_executor.invoke({"input": question})
    answer = result["output"] if isinstance(result, dict) else result
    logger.info(f"Stage2 – final answer: {answer!r}")
    return answer, context_chunks

# If you run this module directly, enter into a REPL loop
if __name__ == "__main__":
    while True:
        q = input("Stage2: Enter query (or 'exit'): ")
        if q.lower() == "exit":
            break
        ans, ctx = run_rag_agent(q)
        print("\nAnswer:\n", ans)
    wait_for_all_tracers()

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os, logging

from langsmith import traceable
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.schema import Document
from langchain_openai import OpenAI  # updated import
from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate


# Load environment
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Configure logging
logging.basicConfig(filename="agent.log", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger()

# Define PDF RAG tool
@traceable(name="pdf_rag_tool")
def pdf_rag_tool(query: str) -> str:
    loader = PyPDFLoader("data/example2.pdf")
    pages = loader.load()
    full_text = "\n".join(p.page_content for p in pages)
    splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", "."], chunk_size=200, chunk_overlap=50)
    chunks = splitter.split_text(full_text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(docs, embed_model)
    matches = vectorstore.similarity_search(query, k=3)
    return "\n\n---\n\n".join(doc.page_content for doc in matches)

# Wrap tool as Agent Tool
pdf_tool = Tool(
    name="pdf_search",
    func=pdf_rag_tool,
    description="Retrieve relevant context from the PDF by similarity search."
)

# Initialize LLM
llm = ChatOpenAI(
   model="gpt-4o-mini-2024-07-18",
   temperature=0.3,     # Temparature means how creative your model will be. 0 for very safe and 1 for creativity/risks 
   max_tokens=100
)

from langchain_core.prompts import PromptTemplate
from langchain.agents.react.agent import create_react_agent
from langchain.agents import AgentExecutor, Tool
# other imports...

template = """
    You are an agent with access to the tool {tools}.
    Follow this format exactly:

    Question: {input}
    Thought: think about what to do next
    Action: one of [{tool_names}]
    Action Input: input for the action
    Observation: result of the action

    (Repeat Thought/Action/Action Input/Observation as needed)

    If you decide you know the answer:
    Thought: I know the final answer
    Final Answer: the answer

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
    handle_parsing_errors=True
)


# # Create agent using ReAct pattern
# agent = create_react_agent(llm=llm, tools=[pdf_tool], prompt=None)

# agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=[pdf_tool], verbose=True)

def handle_query(question: str) -> str:
    logger.info(f"User asks: {question}")
    result = agent_executor.invoke({"input": question})
    logger.info(f"Agent answer: {result}")
    return result

if __name__ == "__main__":
    question = "Tell me about primary actors"
    answer = handle_query(question)
    print(answer)

        
from langchain_core.tracers.langchain import wait_for_all_tracers
wait_for_all_tracers()


# # Load the PDF
# loader = PyPDFLoader("data/example.pdf")  
# data = loader.load()

# # Split into chunks
# splitter = RecursiveCharacterTextSplitter(
#     separators=["\n\n", "\n", " ", "."],
#     chunk_size=100,
#     chunk_overlap=40
# )
# chunks = splitter.split_text(data[0].page_content)
# print("---------------CHUNKS ARE:", chunks)

# # Load API key
# load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")

# # Embedding model
# embedding_model = OpenAIEmbeddings(
#     model="text-embedding-3-small",
#     openai_api_key=api_key
# )

# # Convert text chunks to Document objects
# docs = [Document(page_content=chunk) for chunk in chunks]

# # Create FAISS vector store
# vectorstore = FAISS.from_documents(docs, embedding_model)

# # Query function
# def query_store(query_text):
#     results = vectorstore.similarity_search(query_text, k=3)
#     return [doc.page_content for doc in results]

# # Query example
# query = "neural network"
# results = query_store(query)

# print("Top 3 Matching Chunks:\n")
# for i, res in enumerate(results, start=1):
#     print(i, res)

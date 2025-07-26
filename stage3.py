# stage3.py
from dotenv import load_dotenv
import os, logging

from stage1 import validate_input      # your Stage 1 module
from stage2 import run_rag_agent       # the function above

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain import LLMChain

# --- Environment & Logging ---
load_dotenv()
os.environ["OPENAI_API_KEY"]    = os.getenv("OPENAI_API_KEY")
logging.basicConfig(
    filename="stage3.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- LLM Judgement Setup ---
llm_judge = ChatOpenAI(
    model="gpt-4o-mini-2024-07-18",
    temperature=0.3,
    max_tokens=100
)

hallucination_prompt = PromptTemplate.from_template("""
You are a strict evaluator. Your job is to check if an answer is unsupported by the context.

Question:
{question}

Retrieved Context (from PDF):
{context}

Answer:
{answer}

---

Reply with exactly one of:
- "Supported"    (if every claim in the answer is backed by the context)
- "Hallucinated" (if any claim is not found in the context)
Also provide a one-sentence justification.
""")

hallucination_chain = LLMChain(llm=llm_judge, prompt=hallucination_prompt)

def detect_hallucination_with_llm(question: str, answer: str, context: list[str]) -> str:
    payload = {
        "question": question,
        "context": "\n\n".join(context),
        "answer": answer
    }
    logger.info("Stage3 - calling LLM judge for hallucination check")
    judgement = hallucination_chain.run(**payload).strip()
    logger.info(f"Stage3 - hallucination result: {judgement!r}")
    return judgement

# --- Stage 3 Pipeline ---
def run_stage3(question: str) -> None:
    # 1️⃣ Stage 1: validate
    logger.info(f"Stage3 - received question: {question!r}")
    mod = validate_input(question)
    if mod["status"] == "unsafe":
        logger.warning(f"Stage3 - unsafe input: {mod}")
        print("❌ Unsafe input:", mod.get("categories", mod.get("judgment")))
        return
    if mod["status"] == "ambiguous":
        logger.warning(f"Stage3 - ambiguous input: {mod}")
        print("⚠️ Ambiguous input:", mod["judgment"])
        return
    print("✅ Input passed safety & clarity checks.")

    # 2️⃣ Stage 2: RAG
    answer, context_chunks = run_rag_agent(question)
    print("\n🤖 Agent Answer:\n", answer)

    # 3️⃣ Hallucination Detection
    judge = detect_hallucination_with_llm(question, answer, context_chunks)
    print("\n🔍 Hallucination Check:\n", judge)

# --- CLI Runner ---
if __name__ == "__main__":
    while True:
        q = input("\nStage3: Enter query (type 'exit'): ")
        if q.lower() == "exit":
            break
        run_stage3(q)

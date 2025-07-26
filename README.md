# PDF Chatbot with Evaluation Pipeline

This project is a structured implementation of a **PDF-based Chatbot** powered by a multi-stage **Evaluation Pipeline**. Each stage ensures the chatbot behaves safely, retrieves accurately, and generates fact-supported answers, minimizing hallucinations.

## 🧠 Motivation

Large Language Models (LLMs) and Agents often suffer from **hallucinations**, **unsafe prompts**, and **non-deterministic outputs**. To combat this, I designed a chatbot that not only answers user questions based on uploaded PDF content, but also evaluates itself across multiple validation layers using agentic AI principles.

---

## 🛠️ Tech Stack

- **Python** + **LangChain**
- **OpenAI GPT-4o-mini**
- **Gemini 2.5-flash**
- **Open AI Moderation API**
- **LangSmith** for tracing
- **FAISS**, **ReAct-style Agent**, **PyPDFLoader**

---

## 🧪 Evaluation Pipeline Overview

Each query passes through **four strict evaluation stages**:

---

### 🔹 Stage 1: Input Validation

- Uses **OpenAI Moderation API** to block harmful or unsafe queries.
- Then uses a **Gemini-based LLM check** to flag ambiguous or unclear inputs.
- Only validated questions are allowed to proceed.

---

### 🔹 Stage 2: Core RAG Agent

- Uses **PyPDFLoader** and **FAISS** to retrieve relevant chunks from PDF.
- Runs a **ReAct-style LangChain Agent** to generate the answer.
- Returns both the answer and the source context used.

---

### 🔹 Stage 3: Hallucination Detection

- Implements an **LLM-as-Judge** method.
- Provides the question, context, and answer to another GPT-4o-mini model.
- The judge replies with:
  - `"Supported"` → All claims in answer are found in context.
  - `"Hallucinated"` → Any part of answer lacks supporting context.
- Justification is included for transparency.

---

### 🔹 Stage 4: Final Output Validation

- Counts the number of total questions asked and correct answers passed.
- Prints a report like:  
  `✅ 6/6 answered correctly`  
  `🎯 Completion Rate: 100%`
- All logs are stored in `agent.log`.

---

## 🚀 Running the Project

1. **Install requirements**:
    ```bash
    pip install -r requirements.txt
    ```

2. **Set environment variables**:
    Create a `.env` file:
    ```
    OPENAI_API_KEY = your-key-here
    GEMINI_API_KEY = your-key-here

    LANGCHAIN_TRACING_V2 = true
    LANGCHAIN_API_KEY = your-key-here
    LANGCHAIN_ENDPOINT = https://api.smith.langchain.com
    LANGCHAIN_PROJECT = your-langsmith-project-name
    LANGCHAIN_CALLBACKS_BACKGROUND=false

    ```

3. **Run Stage 4 CLI (includes all previous stages)**:
    ```bash
    python stage4.py
    ```

---

## 📂 Project Structure

```
.
├── stage1.py    # Input Safety & Ambiguity Validation
├── stage2.py    # RAG Agent with PDF Retrieval
├── stage3.py    # LLM-as-Judge Hallucination Detector
├── stage4.py    # Final Output Validator & Tracker
├── .env
├── README.md
```

---

## 🧾 Example Output

```bash
Stage 4: Final Output Validator
Enter query: Who is a primary actor?

✅ Input passed safety & clarity checks.

🤖 Agent Answer (from the PDF):
The primary actor is the one whose goals are fulfilled by the system.

🔍 Hallucination Check:
Supported – The answer is directly found in the retrieved context.

📄 Format Valid: True

✅ 1/1 answered successfully
🎯 Completion Rate: 100%
```

---

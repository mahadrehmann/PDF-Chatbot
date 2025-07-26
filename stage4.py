# stage4.py
from stage3 import run_stage3
import logging

# --- Logging Setup ---
logging.basicConfig(
    filename="stage4.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Counters ---
total_questions = 0
successful_answers = 0

# --- Stage 4 Pipeline ---
def run_stage4(question: str) -> None:
    global total_questions, successful_answers

    result = run_stage3(question)
    if not result:
        logger.warning("Stage4 - Question skipped due to input check.")
        return

    if result["valid"]:
        successful_answers += 1
        logger.info("Stage4 - ✅ Answer passed hallucination check.")
    else:
        logger.warning("Stage4 - ❌ Answer failed hallucination check.")

# --- CLI Loop ---
if __name__ == "__main__":
    print("Stage 4: Final Output Validator")
    print("Ask questions one by one. Type 'exit' to stop.\n")

    while True:
        q = input("Enter query: ").strip()
        if q.lower() == "exit":
            break
        if q:
            total_questions += 1
            run_stage4(q)

    print("\n📊 Final Report:")
    print(f"✅ {successful_answers}/{total_questions} answered successfully")
    if total_questions > 0:
        rate = round((successful_answers / total_questions) * 100, 2)
        print(f"🎯 Completion Rate: {rate}%")
    else:
        print("No queries were evaluated.")

# # stage4.py
# from stage3 import run_stage3
# import logging

# # --- Logging Setup ---
# logging.basicConfig(
#     filename="stage4.log",
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )
# logger = logging.getLogger(__name__)

# # --- Format Validation ---
# def is_answer_format_valid(answer: str) -> bool:
#     lines = answer.strip().split("\n")
#     bullets = [line for line in lines if line.strip().startswith("-") or line.strip().startswith("*")]
#     return len(bullets) >= 2

# # --- Counters ---
# total_questions = 0
# successful_answers = 0

# # --- Stage 4 Pipeline ---
# def run_stage4(question: str) -> None:
#     global total_questions, successful_answers
#     total_questions += 1

#     result = run_stage3(question)
#     if not result:
#         logger.warning("Stage4 - Question skipped due to input check.")
#         return

#     format_valid = is_answer_format_valid(result["answer"])
#     print("📄 Format Valid:", format_valid)

#     if result["valid"] and format_valid:
#         successful_answers += 1
#         logger.info("Stage4 - ✅ Answer passed all checks.")
#     else:
#         logger.warning("Stage4 - ❌ Answer failed final checks.")

# # --- CLI Loop ---
# if __name__ == "__main__":
#     print("Stage 4: Final Output Validator")
#     print("Ask questions one by one. Type 'exit' to stop.\n")

#     while True:
#         q = input("Enter query: ").strip()
#         if q.lower() == "exit":
#             break
#         if q:
#             run_stage4(q)

#     print("\n📊 Final Report:")
#     print(f"✅ {successful_answers}/{total_questions} answered successfully")
#     if total_questions > 0:
#         rate = round((successful_answers / total_questions) * 100, 2)
#         print(f"🎯 Completion Rate: {rate}%")
#     else:
#         print("No queries were evaluated.")

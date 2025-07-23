from dotenv import load_dotenv
import os
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")




print("\nEnter any question about the PDF (Type 'exit' to quit)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("\nThank you, good bye \n")
        break
    print("\nThe User Input is:", user_input)

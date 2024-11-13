import os
from dotenv import load_dotenv
import google.generativeai as genai
import time
from ratelimit import limits, sleep_and_retry

class LLMClient:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=self.api_key)

    @sleep_and_retry
    @limits(calls=2, period=1) # Limita a 2 chamadas por segundo
    def generate_code(self, prompt):
        response = self.model.generate_content(prompt)
        code = response.text
        return code.strip()

    @sleep_and_retry
    @limits(calls=2, period=1) # Limita a 2 chamadas por segundo
    def generate_feedback(self, prompt):
        response = self.model.generate_content(prompt)
        feedback = response.text
        return feedback.strip()

    def generate_code(self, prompt):
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        code = response.text
        return code.strip()

    def generate_feedback(self, prompt):
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        feedback = response.text
        return feedback.strip()

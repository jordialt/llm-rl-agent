"""
llm_helper.py
--------------
A lightweight helper module to interact with a small LLM
using Hugging Face Transformers. Runs efficiently on CPU.

Model: google/flan-t5-small
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class LLMHelper:
    def __init__(self, model_name: str = "google/flan-t5-small"):
        print(f"🔍 Loading model '{model_name}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print("✅ Model loaded successfully!\n")

    def generate(self, prompt: str, max_length: int = 64) -> str:
        """
        Generate a text response from the LLM given a prompt.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=max_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    # Simple test
    llm = LLMHelper()
    prompt = "Suggest a strategy to balance a pole on a cart."
    response = llm.generate(prompt)
    print("🧠 LLM response:")
    print(response)

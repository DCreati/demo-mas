"""Factory function to create the appropriate LLM instance based on configuration"""

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


def create_llm(model_name: str, local: int = 1, temperature: float = 0.5, num_predict: int = 800):
    if local == 1:
        # Use Ollama locally
        return ChatOllama(
            model=model_name,
            temperature=temperature,
            num_predict=num_predict
        )
    else:
        # Use OpenAI API
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=num_predict
        )

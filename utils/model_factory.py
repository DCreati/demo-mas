"""Factory function to create the appropriate LLM instance based on configuration"""

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


def create_llm(model_name: str, local: int = 1, temperature: float = 0.5, num_predict: int = 800):
    """
    Create and return the appropriate LLM instance.
    
    Args:
        model_name: The model name (e.g., "llama3.1:8b" or "gpt-4o-mini")
        local: 1 for Ollama, 0 for OpenAI API
        temperature: Temperature parameter for the model
        num_predict: Max tokens to predict (only used for Ollama)
    
    Returns:
        ChatOllama or ChatOpenAI instance
    """
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

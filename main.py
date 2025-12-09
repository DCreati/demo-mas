import sys
import time
import os
from pathlib import Path

from graph.state import create_initial_state, get_state_summary
from graph.workflow import create_research_workflow, run_workflow, display_workflow_summary
from utils.logger import logger, set_verbosity

VERBOSITY = 1
INTERACTIVE_MODE = False

LOCAL = 0  # 1 = Ollama, 0 = GPT 4o-mini
if LOCAL == 1:
    MODEL_NAME = "llama3.1:8b"
else:
    MODEL_NAME = "gpt-4o-mini"

SAMPLE_PAPER = """
Recent advances in deep learning have demonstrated remarkable performance in image classification tasks. 
However, standard convolutional neural networks often struggle with limited training data and exhibit 
poor generalization to out-of-distribution samples. This paper proposes a novel meta-learning framework 
that combines few-shot learning with adversarial training to improve model robustness. Our approach, 
termed "Adaptive Meta-Adversarial Networks (AMAN)", leverages episodic training to learn transferable 
representations while incorporating adversarial perturbations during meta-training. We introduce a 
dynamic task sampling strategy that progressively increases task difficulty, enabling the model to 
develop more resilient features. Experimental results on miniImageNet and tieredImageNet benchmarks 
show that AMAN achieves state-of-the-art performance, improving 5-shot classification accuracy by 3.2% 
over existing methods while maintaining computational efficiency. Furthermore, our ablation studies 
reveal that the synergy between meta-learning and adversarial training is crucial for achieving robust 
generalization. The proposed framework opens new directions for building AI systems that can rapidly 
adapt to new tasks with minimal data while remaining resilient to distributional shifts.
"""


def load_api_key() -> str:
    """Load OpenAI API key from .env file"""
    env_file = Path(".env")
    
    if not env_file.exists():
        logger.error("File .env non trovato!")
        logger.error("Copia .env.example in .env e aggiungi la tua API key OpenAI")
        sys.exit(1)
    
    try:
        with open(env_file, 'r') as f:
            for line in f:
                if line.startswith("OPENAI_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    if api_key and not api_key.startswith("sk-your-key"):
                        os.environ["OPENAI_API_KEY"] = api_key
                        logger.success("API key caricata con successo")
                        return api_key
        
        logger.error("API key non trovata in .env o non valida")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Errore durante la lettura di .env: {str(e)}")
        sys.exit(1)


def check_ollama_connection(model_name: str) -> bool:
    try:
        from langchain_ollama import ChatOllama
        
        llm = ChatOllama(model=model_name, num_predict=10)
        llm.invoke("test")
        
        logger.success("Ollama is running correctly")
        return True
        
    except Exception as e:
        logger.error(f"Ollama connection failed: {str(e)}")
        return False


def check_openai_connection() -> bool:
    """Check OpenAI API connection"""
    try:
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=10)
        llm.invoke("test")
        
        logger.success("OpenAI API connection successful")
        return True
        
    except Exception as e:
        logger.error(f"OpenAI API connection failed: {str(e)}")
        return False


def display_welcome_banner():
    banner = """
---------------------------------------------------------------
    RESEARCH PAPER ANALYSIS MAS
---------------------------------------------------------------
    """
    print(banner)


def main():
    set_verbosity(VERBOSITY)
    
    display_welcome_banner()
    
    if LOCAL == 0:
        logger.info("Loading OpenAI API key...")
        load_api_key()
        if not check_openai_connection():
            sys.exit(1)
    else:
        logger.info("Checking Ollama connection...")
        if not check_ollama_connection(MODEL_NAME):
            logger.error("Ollama is not running or model not found")
            sys.exit(1)
    
    logger.info("Using embedded sample paper abstract")
    paper_abstract = SAMPLE_PAPER.strip()
    
    if len(paper_abstract) < 100:
        logger.warning("Paper abstract is very short. Results may be limited.")
    
    logger.info(f"Paper abstract: {len(paper_abstract)} characters")
    logger.info(f"Target: Generate comprehensive research review\n")
    
    initial_state = create_initial_state(paper_abstract)
    
    if VERBOSITY >= 2:
        logger.section("INITIAL STATE")
        logger.info(get_state_summary(initial_state))
    
    try:
        workflow = create_research_workflow(model_name=MODEL_NAME, local=LOCAL)
    except Exception as e:
        logger.error(f"Failed to create workflow: {str(e)}")
        sys.exit(1)
    
    if INTERACTIVE_MODE:
        logger.info("Interactive mode enabled - press Enter after each agent")
        input("\nPress Enter to start workflow...")
    
    start_time = time.time()
    
    try:
        final_state = run_workflow(workflow, initial_state)
        
    except KeyboardInterrupt:
        logger.warning("\nWorkflow interrupted by user")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Workflow execution failed: {str(e)}")
        if VERBOSITY >= 2:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    if VERBOSITY >= 1:
        display_workflow_summary(final_state)
    
    logger.section("EXECUTION STATISTICS")
    logger.info(f"Total execution time: {elapsed_time:.2f} seconds")
    logger.info(f"Total agent messages: {len(final_state.get('messages', []))}")
    logger.info(f"Workflow iterations: {final_state.get('iteration_count', 0)}")
    logger.info(f"Analysis complete: {final_state.get('analysis_complete', False)}")
    
    logger.header("DEMONSTRATION END")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupt!")
        sys.exit(0)
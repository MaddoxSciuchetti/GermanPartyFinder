import subprocess
from typing import Optional, Tuple
import os
from config import MODEL_NAME, MODEL_TIMEOUT, TEMPERATURE, MAX_TOKENS, SYSTEM_MESSAGES, DEFAULT_LANGUAGE

def verify_model_availability() -> bool:
    """Check if the Mistral model is available in Ollama.
    
    Returns:
        bool: True if model is available, False otherwise
    """
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True
        )
        return MODEL_NAME in result.stdout
    except Exception:
        return False

def create_system_prompt(role_description: str, language: str = DEFAULT_LANGUAGE) -> str:
    """Create a system prompt that defines the model's role and behavior.
    This is where we 'fine-tune' the model for our specific use case by:
    1. Selecting the appropriate base prompt (party analysis or document analysis)
    2. Combining it with the specific role description
    3. Adding explicit instructions to maintain the role
    
    Args:
        role_description (str): The role description for the AI
        language (str): Language code ('de' for German)
    """
    # Select the appropriate base prompt based on the task
    base_prompt = SYSTEM_MESSAGES[language]["party_analysis"] if "party" in role_description.lower() else SYSTEM_MESSAGES[language]["doc_analysis"]
    
    # Combine base prompt with specific role and instructions
    return f"""Du bist ein KI-Assistent, spezialisiert auf deutsche Politik.

{base_prompt}

Zusätzliche Rolleninformation:
{role_description}

Wichtige Anweisungen:
1. Bleibe durchgehend in dieser Rolle
2. Antworte ausschließlich auf Deutsch
3. Beziehe dich auf aktuelle politische Fakten
4. Strukturiere deine Antworten klar und logisch"""

def run_model(prompt: str) -> Tuple[bool, str]:
    """Run the Mistral model with the given prompt.
    
    Args:
        prompt (str): Prompt to send to the model
        
    Returns:
        Tuple[bool, str]: Success status and response/error message
    """
    try:
        command = [
            "ollama", "run", MODEL_NAME, prompt
        ]
        
        # Get current environment variables and ensure USERPROFILE is set
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            timeout=MODEL_TIMEOUT,
            encoding='utf-8',
            env=env
        )
        
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, f"Model error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return False, f"Model timeout: Response took longer than {MODEL_TIMEOUT} seconds"
    except Exception as e:
        return False, f"Error: {str(e)}"

def ask_deepseek(prompt: str, system_prompt: Optional[str] = None, language: str = DEFAULT_LANGUAGE) -> str:
    """Send a prompt to the Mistral model through Ollama and get the response.
    
    Args:
        prompt (str): The user's question or prompt
        system_prompt (str, optional): System prompt to define model's role
        language (str): Language code ('de' or 'en')
        
    Returns:
        str: Model's response
    """
    try:
        # Check if model is available
        if not verify_model_availability():
            return f"Error: {MODEL_NAME} not found. Please run 'ollama pull {MODEL_NAME}' first."

        # Combine system prompt with user prompt if provided
        full_prompt = f"{system_prompt}\n\nHuman: {prompt}" if system_prompt else prompt
        
        # Add language instruction to prompt
        lang_instruction = "Bitte antworte auf Deutsch." if language == "de" else "Please respond in English."
        full_prompt = f"{full_prompt}\n\n{lang_instruction}"
        
        # Run the model
        success, response = run_model(full_prompt)
        if success:
            return response
        return f"Error: {response}"
            
    except Exception as e:
        return f"Error running model: {str(e)}\nPlease ensure Ollama is running and {MODEL_NAME} is installed."

def chat_with_role(role_description: str, language: str = DEFAULT_LANGUAGE):
    """Create a chat function with a specific role.
    
    Args:
        role_description (str): Description of the role for the AI
        language (str): Language code ('de' or 'en')
        
    Returns:
        function: A chat function that maintains the specified role
    """
    system_prompt = create_system_prompt(role_description, language)
    
    def chat(prompt: str) -> str:
        return ask_deepseek(prompt, system_prompt, language)
    
    return chat

if __name__ == "__main__":
    # Simple command-line interface for testing
    print("Political Analysis Chat (Type 'quit' to exit)")
    print("-" * 50)
    
    # Check model availability
    if not verify_model_availability():
        print(f"Error: {MODEL_NAME} not found. Please install it:")
        print(f"Run: ollama pull {MODEL_NAME}")
        exit(1)
    
    print(f"Using model: {MODEL_NAME}")
    
    # Language selection
    lang = input("Select language (de/en) [default: de]: ").lower() or DEFAULT_LANGUAGE
    if lang not in ["de", "en"]:
        lang = DEFAULT_LANGUAGE
    
    print("\nDefine the AI's role (press Enter twice to finish):")
    role_lines = []
    while True:
        line = input()
        if line == "":
            break
        role_lines.append(line)
    
    role = "\n".join(role_lines)
    chat_func = chat_with_role(role, lang) if role else lambda p: ask_deepseek(p, language=lang)
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nAuf Wiedersehen!" if lang == "de" else "\nGoodbye!")
            break
        
        print("\nAntwort:" if lang == "de" else "\nAnswer:", chat_func(user_input)) 
import json
import logging
import requests
from typing import Dict, Any

logger = logging.getLogger(__name__)

def check_ollama_status(host: str = "http://localhost:11434") -> bool:
    """
    Performs a connection check on the Ollama API server.

    Args:
        host: Ollama server base URL.

    Returns:
        True if the server responded, False otherwise.
    """
    try:
        response = requests.get(host, timeout=3)
        return response.status_code == 200
    except Exception:
        return False

def generate_answer(
    prompt: str,
    model_name: str = "llama3",
    host: str = "http://localhost:11434",
    temperature: float = 0.0,
    max_tokens: int = 512,
    top_p: float = 0.9,
    context_window: int = 4096,
    timeout: int = 60
) -> str:
    """
    Dispatches the RAG prompt to Gemini API or Ollama's HTTP endpoint.

    Args:
        prompt: Compiled prompt string with instructions and context.
        model_name: Name of the LLM model to request from Ollama (e.g., llama3, gemma).
        host: Ollama base host endpoint.
        temperature: Sampling temperature.
        max_tokens: Max tokens to generate (mapped to Ollama's num_predict).
        top_p: Top-p nucleus sampling probability.
        context_window: Maximum token limit for the context (mapped to Ollama's num_ctx).
        timeout: Server timeout limit.

    Returns:
        The generated answer string, or a clean error message.
    """
    import config

    # 1. Check if Gemini API is enabled
    if getattr(config, "USE_GEMINI", False) and getattr(config, "GEMINI_API_KEY", ""):
        api_key = config.GEMINI_API_KEY
        gemini_model = getattr(config, "GEMINI_MODEL", "gemini-1.5-flash")
        logger.info(f"Calling Gemini API model '{gemini_model}'...")
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={api_key}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "topP": top_p
            }
        }
        headers = {"Content-Type": "application/json"}
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if response.status_code == 200:
                result = response.json()
                try:
                    answer = result["candidates"][0]["content"]["parts"][0]["text"]
                    return answer.strip()
                except (KeyError, IndexError) as e:
                    logger.error(f"Error parsing Gemini response JSON: {e}. Raw response: {result}")
                    return "Error: Failed to parse response from Gemini API."
            else:
                logger.error(f"Gemini API returned error code {response.status_code}: {response.text}")
                return f"Error: Gemini API returned status code {response.status_code}."
        except requests.exceptions.Timeout:
            logger.error("Gemini API request timed out.")
            return "Error: Gemini API request timed out."
        except Exception as e:
            logger.error(f"Unexpected error calling Gemini API: {e}")
            return f"Error calling Gemini API: {e}"

    # 2. Fallback to local Ollama
    url = f"{host}/api/generate"
    
    # Configure Ollama parameters as payload options
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "top_p": top_p,
            "num_ctx": context_window
        }
    }

    logger.info("Calling local Ollama LLM...")
    
    # Pre-flight check to verify server availability
    if not check_ollama_status(host):
        logger.error(f"Ollama server is not running or not reachable at {host}.")
        return "Error: Ollama server is not running. Please start Ollama locally and try again."

    try:
        response = requests.post(url, json=payload, timeout=timeout)
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "").strip()
            
        elif response.status_code == 404:
            logger.error(f"Model '{model_name}' is not found in the Ollama cache.")
            return f"Error: Ollama model '{model_name}' is not available. Please run 'ollama pull {model_name}'."
            
        else:
            logger.error(f"Ollama returned HTTP error status {response.status_code}: {response.text}")
            return f"Error: Ollama server returned status code {response.status_code}."

    except requests.exceptions.Timeout:
        logger.error(f"Ollama request timed out after {timeout} seconds.")
        return "Error: Ollama request timed out. Try increasing the timeout threshold."

    except requests.exceptions.ConnectionError:
        logger.error(f"Connection failed when calling Ollama at {host}.")
        return "Error: Connection to Ollama failed. Please ensure the server is running."

    except Exception as e:
        logger.error(f"Unexpected error during Ollama API request: {e}")
        return f"Error: Unexpected model inference error: {e}"


"""
OpenRouter API client for accessing external LLM models.
Provides compatibility with the same interface as local models.
"""

import os
import requests
from typing import Optional
import time


class OpenRouterClient:
    """Client for making requests to OpenRouter API."""

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.http_referer = os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost:8001")
        self.x_title = os.getenv("OPENROUTER_X_TITLE", "Yaguarete LLM Proxy")
        
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")

    def generate_text(
        self,
        prompt: str,
        model_id: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate text using an OpenRouter model.
        
        Args:
            prompt: The input prompt
            model_id: OpenRouter model ID (e.g., "openai/gpt-4o-mini")
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
            
        Raises:
            requests.RequestException: If API call fails
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self.http_referer,
            "X-Title": self.x_title,
            "Content-Type": "application/json",
        }

        payload = {
            "model": model_id,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            if not content:
                raise ValueError("No content in OpenRouter response")
                
            return content.strip()
            
        except requests.exceptions.Timeout:
            raise RuntimeError(f"OpenRouter request timeout for model {model_id}")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"Failed to connect to OpenRouter API")
        except requests.exceptions.HTTPError as e:
            error_msg = str(e.response.text) if hasattr(e, 'response') else str(e)
            raise RuntimeError(f"OpenRouter API error: {error_msg}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error calling OpenRouter: {str(e)}")

    def is_available(self) -> bool:
        """Check if OpenRouter API is accessible."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            response = requests.get(
                "https://openrouter.ai/api/v1/models",
                headers=headers,
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            print(f"[WARNING] OpenRouter availability check failed: {e}")
            return False

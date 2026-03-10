# Author: Mohith Kunta
# GitHub: https://github.com/m-kunta

import os
from typing import Optional

def get_llm_response(prompt: str, provider: str, model: str, system_prompt: Optional[str] = None) -> str:
    """Factory to get responses from different LLM providers."""
    # Anthropic
    if provider.lower() == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        messages = [{"role": "user", "content": prompt}]
        response = client.messages.create(
            model=model,
            max_tokens=8192,
            system=system_prompt if system_prompt else anthropic.NOT_GIVEN,
            messages=messages
        )
        return response.content[0].text
        
    # OpenAI
    elif provider.lower() == "openai":
        import openai
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=8192
        )
        return response.choices[0].message.content
        
    # Gemini
    elif provider.lower() == "gemini":
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
        config_kwargs = {
            "max_output_tokens": 8192,
            "response_mime_type": "application/json"
        }
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt
            
        config = types.GenerateContentConfig(**config_kwargs)
        
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config
        )
        return response.text
        
    # Groq
    elif provider.lower() == "groq":
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=8192
        )
        return response.choices[0].message.content
        
    # Ollama
    elif provider.lower() == "ollama":
        import ollama
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = ollama.chat(
            model=model,
            messages=messages
        )
        return response['message']['content']
        
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

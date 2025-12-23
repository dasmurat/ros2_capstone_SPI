import time
import json
import requests

# Replace with your actual OpenRouter API key
API_KEY = "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"

API_URL = "https://openrouter.ai/api/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def query_model(model, prompt, max_tokens=200):
    """
    Query an LLM on OpenRouter, return response, latency, and token usage.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens
    }

    start_time = time.time()
    response = requests.post(API_URL, headers=HEADERS, data=json.dumps(payload))
    latency = time.time() - start_time

    if response.status_code != 200:
        return {"error": response.text, "latency": latency}

    data = response.json()

    # Extract text
    output_text = data["choices"][0]["message"]["content"]

    # Extract token counts (if available)
    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", None)
    completion_tokens = usage.get("completion_tokens", None)
    total_tokens = usage.get("total_tokens", None)

    return {
        "model": model,
        "response": output_text,
        "latency": latency,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens
    }

def test_example():
    """
    Run a quick test with one model and a simple question.
    """
    model = "openai/gpt-3.5-turbo"  # Change to "deepseek/deepseek-chat" etc.
    prompt = "Explain in one sentence why the sky is blue."
    
    result = query_model(model, prompt)
    print("=== Test Result ===")
    print(f"Model: {result['model']}")
    print(f"Latency: {result['latency']:.2f} sec")
    print(f"Tokens - Prompt: {result['prompt_tokens']}, Completion: {result['completion_tokens']}, Total: {result['total_tokens']}")
    print("Response:", result["response"])

if __name__ == "__main__":
    test_example()

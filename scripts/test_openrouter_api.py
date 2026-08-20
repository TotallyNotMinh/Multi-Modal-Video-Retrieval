import os
import sys
import json
import time
import urllib.request
import urllib.error

# ==============================================================================
# 🔑 PASTE YOUR OPENROUTER API KEY HERE (or set OPENROUTER_API_KEY env var)
# ==============================================================================
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY_HERE")
MODEL_NAME = "nvidia/nemotron-3-ultra-550b-a55b:free"
# Other free options on OpenRouter:
# "meta-llama/llama-3.3-70b-instruct:free"
# "google/gemini-2.0-flash-exp:free"
# ==============================================================================


def test_openrouter_connection(api_key: str, model: str = MODEL_NAME):
    if not api_key or api_key == "YOUR_OPENROUTER_API_KEY_HERE":
        print("❌ Error: Please replace 'YOUR_OPENROUTER_API_KEY_HERE' in scripts/test_openrouter_api.py with your actual OpenRouter key.")
        return False

    url = "https://openrouter.ai/api/v1/chat/completions"

    payload = {
        "model": model,
        "temperature": 0.1,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert Vietnamese transcript reconciler for TV news. Output clean JSON only."
            },
            {
                "role": "user",
                "content": (
                    "Fix the following noisy speech-to-text using the on-screen OCR banner as ground truth spelling.\n\n"
                    "Time: 00:15 - 00:22\n"
                    "Noisy ASR: 'Tối qua, trứng cuối lục 1A đoàn qua địa bàn thì trấn tân Hịp...'\n"
                    "OCR Banner: 'QUỐC LỘ 1A - TT. TÂN HIỆP - TIỀN GIANG'\n\n"
                    "Return JSON with key 'corrected_text'."
                )
            }
        ]
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/TotallyNotMinh/aic2026",
        "X-Title": "AIC 2026 Video Retrieval Pipeline"
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers
    )

    print(f"🚀 Testing OpenRouter API with model: '{model}'...")
    t0 = time.time()

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
            elapsed = time.time() - t0

            choices = result.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content", "")
                reasoning = message.get("reasoning", "")
                usage = result.get("usage", {})

                print("\n✅ SUCCESS! OpenRouter API key is valid and working.")
                print(f"⏱️ Response Time: {elapsed:.2f} seconds")
                print(f"📊 Token Usage  : Prompt: {usage.get('prompt_tokens', 'N/A')}, Completion: {usage.get('completion_tokens', 'N/A')}, Total: {usage.get('total_tokens', 'N/A')}")
                
                if reasoning:
                    print(f"\n🧠 Model Reasoning:\n{reasoning.strip()[:300]}...")
                    
                print(f"\n📝 Output Response:\n{content.strip()}")
                return True
            else:
                print(f"⚠️ Received response without choices: {result}")
                return False

    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        print(f"\n❌ OpenRouter HTTP Error {e.code}: {e.reason}")
        try:
            err_json = json.loads(err_body)
            print(f"   Details: {err_json.get('error', {}).get('message', err_body)}")
        except Exception:
            print(f"   Details: {err_body}")
        return False
    except Exception as e:
        print(f"\n❌ Network error: {e}")
        return False


if __name__ == "__main__":
    test_openrouter_connection(OPENROUTER_API_KEY, MODEL_NAME)

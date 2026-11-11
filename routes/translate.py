import os
import re
import json
from flask import Blueprint, request, jsonify
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

translate_bp = Blueprint('translate', __name__)
groq_key = os.getenv('GROQ_API_KEY', '').strip()
client = Groq(api_key=groq_key) if groq_key else None

GROQ_MODELS = [
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b"
]

def query_groq_translate(prompt, max_tokens=1500):
    if not client:
        return None
    for model_name in GROQ_MODELS:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=max_tokens
            )
            raw = response.choices[0].message.content.strip()
            # Strip think tags if present
            raw = re.sub(r'<think>[\s\S]*?</think>', '', raw).strip()
            start = raw.find('[')
            end = raw.rfind(']')
            if start != -1 and end != -1:
                parsed = json.loads(raw[start:end+1])
                if isinstance(parsed, list):
                    return parsed
        except Exception as e:
            print(f"Model {model_name} error:", e)
            continue
    return None


@translate_bp.route('/translate', methods=['POST'])
def translate_batch():
    data = request.get_json(silent=True) or {}
    texts = data.get('texts')
    target_language = data.get('target_language')

    if not texts or not isinstance(texts, list):
        return jsonify({"error": "texts (array) is required"}), 400
    if not target_language:
        return jsonify({"error": "target_language is required"}), 400

    if target_language.strip().lower() == 'english':
        return jsonify({"translations": texts})

    numbered = "\n".join([f"{i}: {t}" for i, t in enumerate(texts)])

    prompt = f"""Translate each numbered line below into {target_language}.
Keep emojis, formatting, and technical terms (PhantomBreaker, LLaMA, AI, PDF) unchanged.
Lines:
{numbered}

Respond with ONLY a JSON array of translated strings in the exact same order and length:
["translated 0", "translated 1", ...]"""

    try:
        translations = query_groq_translate(prompt)
        if isinstance(translations, list) and len(translations) == len(texts):
            return jsonify({"translations": translations})
        if isinstance(translations, list):
            padded = []
            for i, t in enumerate(texts):
                if i < len(translations) and translations[i]:
                    padded.append(translations[i])
                else:
                    padded.append(t)
            return jsonify({"translations": padded})
    except Exception as e:
        print("Translate exception:", e)

    # Graceful fallback: return original text instead of 500 error
    return jsonify({"translations": texts})

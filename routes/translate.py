import os
import json
from flask import Blueprint, request, jsonify
from groq import Groq

translate_bp = Blueprint('translate', __name__)
client = Groq(api_key=os.getenv('GROQ_API_KEY'))

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
]

def query_groq_translate(prompt, max_tokens=2000):
    last_error = None
    for model_name in GROQ_MODELS:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=max_tokens
            )
            raw = response.choices[0].message.content.strip()
            if '```json' in raw:
                raw = raw.split('```json')[1].split('```')[0].strip()
            elif '```' in raw:
                raw = raw.split('```')[1].split('```')[0].strip()
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except Exception as e:
            last_error = str(e)
            continue
    raise Exception(last_error or "All translation models failed")


@translate_bp.route('/translate', methods=['POST'])
def translate_batch():
    """
    Request body:
    {
      "texts": ["Home", "Phishing Analyzer", "Detect phishing emails..."],
      "target_language": "Tamil"
    }
    Response:
    {
      "translations": ["முகப்பு", "ஃபிஷிங் ஆய்வாளர்", "..."]
    }
    Translates in one batch call to keep it fast and cheap, and preserves
    array order so the frontend can map translations back to the same keys.
    """
    data = request.get_json(silent=True) or {}
    texts = data.get('texts')
    target_language = data.get('target_language')

    if not texts or not isinstance(texts, list):
        return jsonify({"error": "texts (array) is required"}), 400
    if not target_language:
        return jsonify({"error": "target_language is required"}), 400

    # English passthrough — no need to call the model
    if target_language.strip().lower() == 'english':
        return jsonify({"translations": texts})

    numbered = "\n".join([f"{i}: {t}" for i, t in enumerate(texts)])

    prompt = f"""Translate each numbered line below into {target_language}.
Keep the same tone (this is UI text for a cybersecurity app — keep it natural,
concise, and appropriate for buttons/headings/labels where the line is short).
Do NOT translate proper nouns/brand names like "PhantomBreaker", "LLaMA",
"HuggingFace", "AI", "PDF".

Lines to translate:
{numbered}

Respond with ONLY a JSON array of translated strings, in the exact same
order and count as the input lines, no extra text, no explanation:
["translation of line 0", "translation of line 1", ...]"""

    try:
        translations = query_groq_translate(prompt)

        if not isinstance(translations, list) or len(translations) != len(texts):
            return jsonify({"error": "Translation count mismatch, try again"}), 500

        return jsonify({"translations": translations})

    except Exception as e:
        print("=== TRANSLATE ERROR ===", str(e))
        return jsonify({"error": f"Translation failed: {str(e)}"}), 500
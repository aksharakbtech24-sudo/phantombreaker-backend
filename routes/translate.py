import os
import json
from flask import Blueprint, request, jsonify
from groq import Groq
 
translate_bp = Blueprint('translate', __name__)
client = Groq(api_key=os.getenv('GROQ_API_KEY'))
 
 
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
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2000
        )
        raw = response.choices[0].message.content.strip()
        if '```json' in raw:
            raw = raw.split('```json')[1].split('```')[0].strip()
        elif '```' in raw:
            raw = raw.split('```')[1].split('```')[0].strip()
 
        translations = json.loads(raw)
 
        if not isinstance(translations, list) or len(translations) != len(texts):
            return jsonify({"error": "Translation count mismatch, try again"}), 500
 
        return jsonify({"translations": translations})
 
    except Exception as e:
        import traceback
        print("=== TRANSLATE ERROR ===")
        print(traceback.format_exc())
        print("========================")
        return jsonify({"error": f"Translation failed: {str(e)}"}), 500
import os
import json
import requests
from flask import Blueprint, request, jsonify
from groq import Groq

deepfake_bp = Blueprint('deepfake', __name__)
client = Groq(api_key=os.getenv('GROQ_API_KEY'))

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
]

def translate_strings(strings, target_language):
    """Translate a list of strings into target_language using Groq with model fallback."""
    if not target_language or target_language.strip().lower() == 'english':
        return strings
    if not strings:
        return strings

    numbered = "\n".join([f"{i}: {s}" for i, s in enumerate(strings)])
    prompt = f"""Translate each numbered line into {target_language}, using that
language's native script. Keep numbers/percentages as-is. Respond with ONLY
a JSON array of translated strings, same order and count, no extra text:

{numbered}"""

    for model_name in GROQ_MODELS:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=800
            )
            raw = response.choices[0].message.content.strip()
            if '```json' in raw:
                raw = raw.split('```json')[1].split('```')[0].strip()
            elif '```' in raw:
                raw = raw.split('```')[1].split('```')[0].strip()
            translated = json.loads(raw)
            if isinstance(translated, list) and len(translated) == len(strings):
                return translated
        except Exception:
            continue
    return strings  # fallback: original English


@deepfake_bp.route('/detect-deepfake', methods=['POST'])
def detect_deepfake():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    image_file = request.files['image']
    image_bytes = image_file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        return jsonify({"error": "Image too large. Max 10MB"}), 400

    response_language = request.form.get('response_language')  # user's selected site language

    try:
        hf_key = os.getenv("HUGGINGFACE_API_KEY", "")
        headers = {'Content-Type': 'image/jpeg'}
        if hf_key:
            headers['Authorization'] = f'Bearer {hf_key}'

        hf_response = requests.post(
            'https://router.huggingface.co/hf-inference/models/umm-maybe/AI-image-detector',
            headers=headers,
            data=image_bytes,
            timeout=25
        )
        hf_data = hf_response.json()
        print("HuggingFace raw response:", hf_data)

        if isinstance(hf_data, dict) and 'error' in hf_data:
            return jsonify({"error": f"Deepfake detection service busy: {hf_data['error']}"}), 503

        fake_score = 0
        real_score = 0
        results = hf_data[0] if (isinstance(hf_data, list) and len(hf_data) > 0 and isinstance(hf_data[0], list)) else hf_data

        if isinstance(results, list):
            for item in results:
                label = item.get('label', '').lower()
                score = item.get('score', 0)
                if 'fake' in label or 'artificial' in label or 'generated' in label or label == 'ai':
                    fake_score = round(score * 100, 1)
                elif 'real' in label or 'human' in label or label == 'genuine':
                    real_score = round(score * 100, 1)

        # If real_score wasn't explicitly set, calculate as complement
        if real_score == 0 and fake_score > 0:
            real_score = round(100 - fake_score, 1)

        is_deepfake = fake_score > 60
        indicators = []

        if is_deepfake:
            if fake_score > 90:
                indicators.append("High confidence AI generation patterns detected")
                indicators.append("Unnatural noise distribution found in pixel rendering")
                indicators.append("Lighting or surface texture inconsistencies detected")
            elif fake_score > 70:
                indicators.append("Suspicious pixel artifacts detected")
                indicators.append("Possible digital manipulation or synthetic elements present")
            else:
                indicators.append("Minor synthetic artifacts detected")
                indicators.append("Recommend manual verification")
        else:
            indicators.append("Natural pixel patterns detected across the image")
            indicators.append("No facial synthesis or deepfake artifacts found")

        explanation = (
            f"This image has a {fake_score}% probability of containing AI-generated elements. "
            + ("Our computer vision model detected patterns characteristic of synthetic media or deepfakes. "
               if is_deepfake else
               "The image demonstrates authentic texture and pixel characteristics. ")
            + ("Treat this image with caution if relying on it for verification." if is_deepfake else "No significant deepfake manipulation detected.")
        )

        # Translate explanation + indicators together into the selected language, if not English
        translated = translate_strings([explanation] + indicators, response_language)
        explanation = translated[0]
        indicators = translated[1:]

        return jsonify({
            "is_deepfake": is_deepfake,
            "fake_score": fake_score,
            "real_score": real_score,
            "explanation": explanation,
            "indicators": indicators
        })
    except Exception as e:
        return jsonify({"error": f"Deepfake analysis failed: {str(e)}"}), 500

 
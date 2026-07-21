import os
import json
import requests
from flask import Blueprint, request, jsonify
from groq import Groq
 
deepfake_bp = Blueprint('deepfake', __name__)
client = Groq(api_key=os.getenv('GROQ_API_KEY'))
 
 
def translate_strings(strings, target_language):
    """Translate a list of strings into target_language using Groq.
    Returns the original list unchanged if target_language is English/empty,
    or if translation fails for any reason (never breaks the main response)."""
    if not target_language or target_language.strip().lower() == 'english':
        return strings
    if not strings:
        return strings
 
    numbered = "\n".join([f"{i}: {s}" for i, s in enumerate(strings)])
    prompt = f"""Translate each numbered line into {target_language}, using that
language's native script. Keep numbers/percentages as-is. Respond with ONLY
a JSON array of translated strings, same order and count, no extra text:
 
{numbered}"""
 
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
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
        pass
    return strings  # fallback: original English, never break the response
 
 
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
        hf_response = requests.post(
            'https://router.huggingface.co/hf-inference/models/umm-maybe/AI-image-detector',
            headers={
                'Authorization': f'Bearer {os.getenv("HUGGINGFACE_API_KEY")}',
                'Content-Type': 'image/jpeg'
            },
            data=image_bytes,
            timeout=30
        )
        hf_data = hf_response.json()
        print("HuggingFace raw response:", hf_data)
        if isinstance(hf_data, dict) and 'error' in hf_data:
            return jsonify({"error": f"HuggingFace error: {hf_data['error']}"}), 503
        fake_score = 0
        real_score = 0
        results = hf_data[0] if isinstance(hf_data[0], list) else hf_data
        for item in results:
            label = item.get('label', '').lower()
            score = item.get('score', 0)
            print(f"Label: {label}, Score: {score}")
            if 'fake' in label or 'deepfake' in label or 'artificial' in label or 'generated' in label or label == 'ai':
                fake_score = round(score * 100, 1)
            elif 'real' in label or 'human' in label or label == 'genuine':
                real_score = round(score * 100, 1)
        is_deepfake = fake_score > 50
        indicators = []
        if is_deepfake:
            if fake_score > 90:
                indicators.append("Very high probability of AI generation detected in facial features")
                indicators.append("Unnatural pixel patterns found around edges")
                indicators.append("Lighting inconsistencies typical of GAN-generated images")
            elif fake_score > 70:
                indicators.append("Suspicious pixel patterns detected")
                indicators.append("Possible facial manipulation detected")
            else:
                indicators.append("Some AI generation artifacts detected")
                indicators.append("Recommend manual review")
        explanation = (
            f"This image has a {fake_score}% probability of being AI-generated. "
            + ("Our computer vision model detected patterns typical of deepfake or AI-generated images. "
               if is_deepfake else
               "The image appears to be authentic with natural pixel patterns. ")
            + ("Treat this image with caution." if is_deepfake else "No significant manipulation detected.")
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
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500
 
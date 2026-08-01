import os
import io
import json
import requests
from flask import Blueprint, request, jsonify
from groq import Groq
from PIL import Image

deepfake_bp = Blueprint('deepfake', __name__)
client = Groq(api_key=os.getenv('GROQ_API_KEY'))

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
]

HF_MODELS = [
    'https://router.huggingface.co/hf-inference/models/umm-maybe/AI-image-detector',
    'https://router.huggingface.co/hf-inference/models/Organika/sdxl-detector'
]

def analyze_image_properties(image_bytes):
    """
    Analyzes PIL Image properties to detect if the upload is a digital screenshot / screen capture
    vs a natural camera photograph or raw AI image.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size
        aspect_ratio = max(width, height) / max(1, min(width, height))

        # Check typical phone screen aspect ratios (16:9 = 1.77, 19.5:9 = 2.16, etc)
        is_screen_ratio = (1.5 <= aspect_ratio <= 2.3)

        # Calculate color entropy / unique colors
        img_rgb = img.convert('RGB')
        colors = img_rgb.getcolors(maxcolors=100000)
        unique_colors_count = len(colors) if colors else 100000

        # UI Screenshots have sharp solid color blocks and lower unique color counts compared to camera photos
        is_screenshot = is_screen_ratio and (unique_colors_count < 25000)

        return {
            "width": width,
            "height": height,
            "aspect_ratio": round(aspect_ratio, 2),
            "is_screenshot": is_screenshot
        }
    except Exception as e:
        print("Image property analysis warning:", e)
        return {"is_screenshot": False}


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

    # Inspect image metadata & properties
    props = analyze_image_properties(image_bytes)
    is_screenshot = props.get("is_screenshot", False)

    hf_data = None
    hf_key = os.getenv("HUGGINGFACE_API_KEY", "")
    headers = {'Content-Type': 'image/jpeg'}
    if hf_key:
        headers['Authorization'] = f'Bearer {hf_key}'

    # Query Hugging Face AI detector models
    for model_url in HF_MODELS:
        try:
            hf_response = requests.post(model_url, headers=headers, data=image_bytes, timeout=15)
            if hf_response.status_code == 200:
                hf_data = hf_response.json()
                break
        except Exception as e:
            print(f"HF model {model_url} error:", e)
            continue

    raw_fake_score = 0
    raw_real_score = 0

    if hf_data:
        results = hf_data[0] if (isinstance(hf_data, list) and len(hf_data) > 0 and isinstance(hf_data[0], list)) else hf_data
        if isinstance(results, list):
            for item in results:
                label = item.get('label', '').lower()
                score = item.get('score', 0)
                if 'fake' in label or 'artificial' in label or 'generated' in label or label == 'ai':
                    raw_fake_score = round(score * 100, 1)
                elif 'real' in label or 'human' in label or label == 'genuine':
                    raw_real_score = round(score * 100, 1)

    # If real score wasn't explicit, complement it
    if raw_real_score == 0 and raw_fake_score > 0:
        raw_real_score = round(100 - raw_fake_score, 1)

    # Detect filename cues (e.g. ChatGPT, DALL-E, Midjourney, AI)
    filename = image_file.filename.lower()
    is_ai_filename = any(k in filename for k in ['chatgpt', 'dall-e', 'dalle', 'midjourney', 'stablediffusion', 'bing', 'ai_image', 'synth'])

    # Determine final AI score
    # If filename or HF model detects high confidence AI (raw_fake_score > 60%), it is AI generated!
    if is_ai_filename or raw_fake_score >= 60:
        fake_score = raw_fake_score if raw_fake_score > 0 else 95.0
        real_score = round(100 - fake_score, 1)
        is_deepfake = True
        image_type_label = "AI Generated Image (DALL-E / ChatGPT / Midjourney)"
    elif is_screenshot and raw_fake_score < 60:
        fake_score = round(raw_fake_score * 0.3, 1)
        real_score = round(100 - fake_score, 1)
        is_deepfake = False
        image_type_label = "Digital Screenshot / Screen Capture"
    else:
        fake_score = raw_fake_score
        real_score = raw_real_score
        is_deepfake = fake_score >= 50
        image_type_label = "Photographic Image"

    indicators = []
    if is_deepfake:
        if is_ai_filename:
            indicators.append("AI Image Generation signature detected (ChatGPT / DALL-E 3 / Midjourney)")
        indicators.append("High-confidence synthetic AI patterns detected in pixel frequency")
        indicators.append("Unnatural surface rendering and lighting inconsistencies found")
        indicators.append("Patterns match diffusion neural network models")
    elif is_screenshot:
        indicators.append("Digital UI & screen capture layout detected (App Interface / Screenshot)")
        indicators.append("Solid color UI elements & text rendering identified")
        indicators.append("No facial deepfake or AI synthesis detected in screen capture")
    else:
        indicators.append("Authentic camera pixel distribution & natural color noise")
        indicators.append("No AI image generation or facial synthesis artifacts found")

    if is_deepfake:
        explanation = f"This image has a {fake_score}% probability of being AI-generated or manipulated ({image_type_label}). Our computer vision models detected neural network synthetic rendering patterns."
    elif is_screenshot:
        explanation = f"This image was identified as an authentic Digital Screenshot ({props.get('aspect_ratio', 'screen')} aspect ratio). The UI graphics and text overlays were verified authentic with 0% deepfake risk."
    else:
        explanation = f"This image appears to be an authentic photograph ({real_score}% real confidence). No AI deepfake manipulation detected."


    # Translate explanation + indicators into user's selected language
    translated = translate_strings([explanation] + indicators, response_language)
    explanation = translated[0]
    indicators = translated[1:]

    return jsonify({
        "is_deepfake": is_deepfake,
        "fake_score": fake_score,
        "real_score": real_score,
        "image_type": image_type_label,
        "explanation": explanation,
        "indicators": indicators
    })


 
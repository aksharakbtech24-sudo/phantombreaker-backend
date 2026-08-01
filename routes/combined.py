import os
import json
import requests
from flask import Blueprint, request, jsonify
from groq import Groq

combined_bp = Blueprint('combined', __name__)
client = Groq(api_key=os.getenv('GROQ_API_KEY'))

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
]

def query_groq_fallback(prompt, max_tokens=1000):
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
            return json.loads(raw)
        except Exception:
            continue
    return None

@combined_bp.route('/combined-analysis', methods=['POST'])
def combined_analysis():
    """
    Accepts multi-part form data:
    - email_text (form field)
    - email_address (form field)
    - response_language (form field)
    - image (file upload)
    """
    email_text = request.form.get('email_text', '').strip()
    email_address = request.form.get('email_address', '').strip()
    response_language = request.form.get('response_language', 'English').strip()

    has_email = len(email_text) >= 10
    has_address = len(email_address) >= 5 and '@' in email_address
    has_image = 'image' in request.files and request.files['image'].filename != ''

    if not has_email and not has_address and not has_image:
        return jsonify({"error": "Please provide input for at least one module (Email text, Image, or Breach Email)."}), 400

    phishing_result = {"threat_score": 0, "is_phishing": False, "explanation": "No email text provided for analysis."}
    deepfake_result = {"fake_score": 0, "is_deepfake": False, "explanation": "No image provided for analysis."}
    breach_result = {"breach_count": 0, "breaches": [], "message": "No breach email provided."}

    # 1. Phishing Analysis
    if has_email:
        try:
            prompt = f"""You are an expert cybersecurity analyst. Analyze this email for phishing:
{email_text}

Respond ONLY in valid JSON:
{{
  "is_phishing": true/false,
  "threat_score": number 0-100,
  "explanation": "Brief 1-2 sentence verdict",
  "dangerous_sentences": ["sentence"],
  "manipulation_tactics": ["tactic"]
}}"""
            res = query_groq_fallback(prompt, max_tokens=600)
            if res:
                phishing_result = res
        except Exception as e:
            phishing_result["explanation"] = f"Phishing scan error: {str(e)}"

    # 2. Deepfake Analysis
    if has_image:
        image_file = request.files['image']
        image_bytes = image_file.read()
        try:
            hf_key = os.getenv("HUGGINGFACE_API_KEY", "")
            headers = {'Content-Type': 'image/jpeg'}
            if hf_key:
                headers['Authorization'] = f'Bearer {hf_key}'

            hf_resp = requests.post(
                'https://router.huggingface.co/hf-inference/models/umm-maybe/AI-image-detector',
                headers=headers,
                data=image_bytes,
                timeout=20
            )
            if hf_resp.status_code == 200:
                hf_data = hf_resp.json()
                results = hf_data[0] if (isinstance(hf_data, list) and len(hf_data) > 0 and isinstance(hf_data[0], list)) else hf_data
                fake_score = 0
                if isinstance(results, list):
                    for item in results:
                        lbl = item.get('label', '').lower()
                        sc = item.get('score', 0)
                        if 'fake' in lbl or 'artificial' in lbl or 'generated' in lbl or lbl == 'ai':
                            fake_score = round(sc * 100, 1)
                is_df = fake_score > 60
                deepfake_result = {
                    "fake_score": fake_score,
                    "is_deepfake": is_df,
                    "explanation": f"Image artificiality probability is {fake_score}%."
                }
        except Exception as e:
            deepfake_result["explanation"] = f"Deepfake scan error: {str(e)}"

    # 3. Breach Analysis
    if has_address:
        leakcheck_key = os.getenv('LEAKCHECK_API_KEY', '')
        if leakcheck_key:
            try:
                b_resp = requests.get(
                    'https://leakcheck.io/api/public',
                    params={'key': leakcheck_key, 'check': email_address, 'type': 'email'},
                    timeout=8
                )
                if b_resp.status_code == 200:
                    b_data = b_resp.json()
                    if b_data.get('success'):
                        sources = b_data.get('sources', [])
                        breaches = [{"name": s.get('name', 'Unknown'), "date": s.get('date', 'Unknown')} for s in sources]
                        breach_result = {
                            "breach_count": len(breaches),
                            "breaches": breaches,
                            "message": f"Found in {len(breaches)} data breach(es)."
                        }
                    else:
                        breach_result = {"breach_count": 0, "breaches": [], "message": "No breaches found."}
            except Exception as e:
                breach_result["message"] = f"Breach scan error: {str(e)}"

    # Calculate Unified Threat Index (UTI)
    p_score = phishing_result.get('threat_score', 0)
    d_score = deepfake_result.get('fake_score', 0)
    b_count = breach_result.get('breach_count', 0)
    b_score = min(100, b_count * 25)

    # Weighted calculation
    active_weights = 0
    total_weighted = 0

    if has_email:
        total_weighted += p_score * 0.40
        active_weights += 0.40
    if has_image:
        total_weighted += d_score * 0.35
        active_weights += 0.35
    if has_address:
        total_weighted += b_score * 0.25
        active_weights += 0.25

    if active_weights > 0:
        unified_score = min(100, int(total_weighted / active_weights))
    else:
        unified_score = 0

    # Determine Grade
    if unified_score <= 15:
        grade = "A+"
        grade_label = "SECURE"
        color = "var(--success)"
    elif unified_score <= 35:
        grade = "B"
        grade_label = "LOW RISK"
        color = "var(--accent-cyan)"
    elif unified_score <= 65:
        grade = "C"
        grade_label = "MODERATE RISK"
        color = "var(--warning)"
    else:
        grade = "F"
        grade_label = "CRITICAL THREAT"
        color = "var(--danger)"

    # Actionable Recommendations
    recommendations = []
    if p_score >= 50:
        recommendations.append("Do not click links or download attachments from analyzed email.")
    if d_score >= 50:
        recommendations.append("Treat uploaded image with caution; synthetic artifacts detected.")
    if b_count > 0:
        recommendations.append(f"Exposed in {b_count} breach(es). Immediately update account passwords and enable 2FA.")
    if len(recommendations) == 0:
        recommendations.append("Maintain good cybersecurity hygiene and monitor accounts regularly.")

    return jsonify({
        "unified_threat_score": unified_score,
        "security_grade": grade,
        "grade_label": grade_label,
        "grade_color": color,
        "phishing_module": phishing_result,
        "deepfake_module": deepfake_result,
        "breach_module": breach_result,
        "recommendations": recommendations,
        "summary": f"Security Posture Grade: {grade} ({grade_label}) with Unified Threat Score of {unified_score}/100."
    })

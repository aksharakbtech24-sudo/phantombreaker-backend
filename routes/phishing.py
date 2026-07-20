
import os
import json
import requests
from flask import Blueprint, request, jsonify
from groq import Groq
 
phishing_bp = Blueprint('phishing', __name__)
client = Groq(api_key=os.getenv('GROQ_API_KEY'))
 
@phishing_bp.route('/analyze-phishing', methods=['POST'])
def analyze_phishing():
    data = request.get_json()
    email_text = data.get('email_text', '')
 
    if not email_text or len(email_text) < 10:
        return jsonify({"error": "Email text too short"}), 400
 
    prompt = f"""You are an expert cybersecurity analyst specializing in phishing detection.
Analyze this email and respond ONLY in valid JSON format.
 
Email to analyze:
{email_text}
 
IMPORTANT LANGUAGE RULE:
First detect the language the email is written in (English, Hindi, Telugu, Tamil,
Kannada, Malayalam, Bengali, Marathi, Gujarati, Punjabi, Odia, Urdu, Assamese,
Sanskrit, or any other Indian or world language such as Spanish, French, Arabic,
Chinese, Japanese, German, Russian, Portuguese, Korean, Italian, Turkish, etc).
Then write "explanation", "dangerous_sentences", and "manipulation_tactics"
IN THAT SAME DETECTED LANGUAGE — not in English — using that language's native
script. Only "language_detected" and the JSON keys themselves stay in English.
If the email is in English, respond in English as normal.
 
Respond with ONLY this JSON structure, no extra text:
{{
  "is_phishing": true or false,
  "threat_score": number between 0 and 100,
  "dangerous_sentences": ["exact sentence from email that is dangerous, written in the detected language"],
  "manipulation_tactics": ["tactic name, written in the detected language, e.g. Urgency/तात्कालिकता/అత్యవసరత"],
  "explanation": "2-3 sentence explanation written in the detected language, of why this is or isnt phishing",
  "language_detected": "the exact language name in English, e.g. Telugu"
}}"""
 
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1200
        )
        raw = response.choices[0].message.content.strip()
        # Clean up response
        if '```json' in raw:
            raw = raw.split('```json')[1].split('```')[0].strip()
        elif '```' in raw:
            raw = raw.split('```')[1].split('```')[0].strip()
        result = json.loads(raw)
    except Exception as e:
        return jsonify({"error": f"AI analysis failed: {str(e)}"}), 500
 
    # Sapling AI detection
    ai_score = 0
    try:
        sapling_response = requests.post(
            'https://api.sapling.ai/api/v1/aidetect',
            json={
                'key': os.getenv('SAPLING_API_KEY'),
                'text': email_text
            },
            timeout=10
        )
        if sapling_response.status_code == 200:
            sapling_data = sapling_response.json()
            ai_score = round(sapling_data.get('score', 0) * 100)
    except:
        ai_score = 0
 
    result['ai_written_score'] = ai_score
    result['combined_threat_score'] = min(100, int(
        result.get('threat_score', 0) * 0.7 + ai_score * 0.3
    ))
 
    return jsonify(result)
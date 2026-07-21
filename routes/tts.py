import io
import base64
from flask import Blueprint, request, jsonify
from gtts import gTTS
 
tts_bp = Blueprint('tts', __name__)
 
# Maps site language names to gTTS language codes.
# gTTS (Google's free TTS) supports these Indian languages natively.
# Languages not listed here have no free TTS voice available and will
# fall back to English on the frontend.
GTTS_LANG_CODES = {
    'English': 'en',
    'Hindi': 'hi',
    'Bengali': 'bn',
    'Telugu': 'te',
    'Marathi': 'mr',
    'Tamil': 'ta',
    'Urdu': 'ur',
    'Gujarati': 'gu',
    'Kannada': 'kn',
    'Malayalam': 'ml',
    'Nepali': 'ne',
}
 
 
@tts_bp.route('/tts', methods=['POST'])
def text_to_speech():
    """
    Request body: { "text": "...", "language": "Telugu" }
    Response: { "audio_base64": "...", "supported": true/false, "used_language": "Telugu" }
 
    Generates real speech audio server-side via gTTS, so playback doesn't
    depend on the user's device having a voice installed for that language.
    If the language isn't supported by gTTS, falls back to English audio
    and reports supported: false so the frontend can show the right badge/state.
    """
    data = request.get_json(silent=True) or {}
    text = data.get('text', '')
    language = data.get('language', 'English')
 
    if not text:
        return jsonify({"error": "text is required"}), 400
 
    lang_code = GTTS_LANG_CODES.get(language)
    supported = lang_code is not None
    used_language = language if supported else 'English'
 
    try:
        tts = gTTS(text=text, lang=(lang_code or 'en'))
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        audio_base64 = base64.b64encode(buf.read()).decode('utf-8')
 
        return jsonify({
            "audio_base64": audio_base64,
            "supported": supported,
            "used_language": used_language
        })
    except Exception as e:
        return jsonify({"error": f"TTS generation failed: {str(e)}"}), 500
 
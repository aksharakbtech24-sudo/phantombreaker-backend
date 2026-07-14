import os
import requests
from flask import Blueprint, request, jsonify

breach_bp = Blueprint('breach', __name__)

@breach_bp.route('/check-breach', methods=['GET'])
def check_breach():
    email = request.args.get('email', '')

    if not email or '@' not in email:
        return jsonify({"error": "Invalid email address"}), 400

    leakcheck_key = os.getenv('LEAKCHECK_API_KEY', '')

    if not leakcheck_key:
        return jsonify({"error": "Breach scanner API not configured"}), 500

    try:
        response = requests.get(
            'https://leakcheck.io/api/public',
            params={
                'key': leakcheck_key,
                'check': email,
                'type': 'email'
            },
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                sources = data.get('sources', [])
                breaches = [{
                    "name": s.get('name', 'Unknown'),
                    "domain": s.get('name', '').lower().replace(' ', '') + '.com',
                    "breach_date": s.get('date', 'Unknown'),
                    "added_date": s.get('date', 'Unknown'),
                    "pwn_count": 0,
                    "data_classes": s.get('fields', []),
                    "is_verified": True
                } for s in sources]

                return jsonify({
                    "email": email,
                    "breach_count": len(breaches),
                    "breaches": breaches,
                    "source": "live"
                })
            else:
                return jsonify({
                    "email": email,
                    "breach_count": 0,
                    "breaches": [],
                    "source": "live",
                    "message": data.get('message', 'No breaches found')
                })
        else:
            return jsonify({"error": f"API error: {response.status_code}"}), 500

    except Exception as e:
        return jsonify({"error": f"Breach scanner error: {str(e)}"}), 500
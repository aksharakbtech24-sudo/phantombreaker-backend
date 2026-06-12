import os
import requests
from flask import Blueprint, request, jsonify

breach_bp = Blueprint('breach', __name__)

MOCK_BREACHES = [
    {
        "name": "Adobe",
        "domain": "adobe.com",
        "breach_date": "2013-10-04",
        "added_date": "2013-12-04",
        "pwn_count": 152445165,
        "data_classes": ["Email addresses", "Password hints", "Passwords", "Usernames"],
        "is_verified": True
    },
    {
        "name": "LinkedIn",
        "domain": "linkedin.com",
        "breach_date": "2016-05-18",
        "added_date": "2016-05-22",
        "pwn_count": 164611595,
        "data_classes": ["Email addresses", "Passwords"],
        "is_verified": True
    },
    {
        "name": "Canva",
        "domain": "canva.com",
        "breach_date": "2019-05-24",
        "added_date": "2019-08-09",
        "pwn_count": 137272116,
        "data_classes": ["Email addresses", "Geographic locations", "Names", "Passwords", "Usernames"],
        "is_verified": True
    },
    {
        "name": "Facebook",
        "domain": "facebook.com",
        "breach_date": "2019-01-01",
        "added_date": "2021-04-03",
        "pwn_count": 509458528,
        "data_classes": ["Email addresses", "Phone numbers", "Names", "Geographic locations"],
        "is_verified": True
    },
    {
        "name": "Twitter",
        "domain": "twitter.com",
        "breach_date": "2022-07-01",
        "added_date": "2022-11-27",
        "pwn_count": 211524284,
        "data_classes": ["Email addresses", "Phone numbers"],
        "is_verified": True
    }
]

@breach_bp.route('/check-breach', methods=['GET'])
def check_breach():
    email = request.args.get('email', '')

    if not email or '@' not in email:
        return jsonify({"error": "Invalid email address"}), 400

    # Try real HIBP API first
    hibp_key = os.getenv('HIBP_API_KEY', '')
    if hibp_key and hibp_key != 'skip_for_now':
        try:
            headers = {
                'hibp-api-key': hibp_key,
                'User-Agent': 'PhantomBreaker-Security-App'
            }
            response = requests.get(
                f'https://haveibeenpwned.com/api/v3/breachedaccount/{email}',
                headers=headers,
                params={'truncateResponse': 'false'},
                timeout=10
            )
            if response.status_code == 200:
                breaches_raw = response.json()
                breaches = [{
                    "name": b.get('Name', ''),
                    "domain": b.get('Domain', ''),
                    "breach_date": b.get('BreachDate', ''),
                    "added_date": b.get('AddedDate', '')[:10] if b.get('AddedDate') else '',
                    "pwn_count": b.get('PwnCount', 0),
                    "data_classes": b.get('DataClasses', []),
                    "is_verified": b.get('IsVerified', False)
                } for b in breaches_raw]
                return jsonify({
                    "email": email,
                    "breach_count": len(breaches),
                    "breaches": breaches,
                    "source": "live"
                })
            elif response.status_code == 404:
                return jsonify({
                    "email": email,
                    "breach_count": 0,
                    "breaches": [],
                    "source": "live"
                })
        except:
            pass

    # Use realistic demo data
    import hashlib
    email_hash = int(hashlib.md5(email.encode()).hexdigest(), 16)

    if '@gmail.com' in email or '@yahoo.com' in email or '@hotmail.com' in email:
        num_breaches = (email_hash % 3) + 2
    else:
        num_breaches = email_hash % 3

    selected = MOCK_BREACHES[:num_breaches]

    return jsonify({
        "email": email,
        "breach_count": len(selected),
        "breaches": selected,
        "source": "demo",
        "note": "Demo mode - connect HIBP API for live data"
    })
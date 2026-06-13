import os
from flask import Flask
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# ✅ FIXED — was broken before
CORS(app, resources={r"/api/*": {"origins": [
    "https://phantombreaker.vercel.app",
    "http://localhost:3000"
]}})

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["100 per hour", "20 per minute"]
)

@app.route('/api/health')
def health():
    return {
        "status": "PhantomBreaker online",
        "version": "1.0.0",
        "modules": ["phishing", "deepfake", "breach"]
    }

from routes.phishing import phishing_bp
from routes.deepfake import deepfake_bp
from routes.breach import breach_bp

app.register_blueprint(phishing_bp, url_prefix='/api')
app.register_blueprint(deepfake_bp, url_prefix='/api')
app.register_blueprint(breach_bp, url_prefix='/api')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
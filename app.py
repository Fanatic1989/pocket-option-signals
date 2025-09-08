# app.py
import os
from flask import Flask, jsonify
from routes import bp as dashboard_bp
from utils import init_db, ensure_config, schedule_jobs, TIMEZONE

def create_app():
    app = Flask(__name__)
    app.secret_key = os.getenv("FLASK_SECRET", os.urandom(32))

    init_db()
    ensure_config()
    schedule_jobs()

    app.register_blueprint(dashboard_bp)

    @app.route('/_up', methods=['GET','HEAD'])
    def _up(): return ('', 200)

    @app.route('/health', methods=['GET','HEAD'])
    def health():
        from flask import request
        if request.method == 'HEAD':
            return ('', 200)
        return jsonify({'ok': True, 'tz': TIMEZONE})
    return app

app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT','8000')), debug=False)

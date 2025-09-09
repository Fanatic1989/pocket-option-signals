import os
from flask import Flask
from routes import bp as dashboard_bp

# Import Live Engine to enable in-web live signaling without a paid worker
from live_engine import ENGINE

def create_app():
    app = Flask(__name__)
    app.secret_key = os.getenv("SECRET_KEY","devkey")
    app.register_blueprint(dashboard_bp)

    # Optional autostart: set LIVE_AUTOSTART=1 on Render to begin live loop when web boots
    if os.getenv("LIVE_AUTOSTART","0") == "1":
        ENGINE.start()

    return app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)

web:    gunicorn -w 1 -k gthread -b 0.0.0.0:$PORT app:app
worker: python -u auto_worker_all_strategy.py

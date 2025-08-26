# app/worker.py
import os
import redis
import threading
from flask import Flask
from rq import Queue
from rq.worker import SimpleWorker
from app.config import REDIS_URL, QUEUE_NAME

# Extra safety on macOS
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

# Create simple Flask app for healthcheck
health_app = Flask(__name__)

@health_app.route('/health')
def health():
    return {"status": "worker_healthy"}

@health_app.route('/')
def root():
    return {"service": "billboard-worker", "status": "running"}

def run_health_server():
    """Run health endpoint on separate thread"""
    port = int(os.environ.get("PORT", 8000))
    health_app.run(host="0.0.0.0", port=port, debug=False)

def run_worker():
    """Run the actual RQ worker"""
    listen = [QUEUE_NAME]
    redis_conn = redis.from_url(REDIS_URL)
    queues = [Queue(name, connection=redis_conn) for name in listen]
    worker = SimpleWorker(queues, connection=redis_conn)
    worker.work()

if __name__ == "__main__":
    # Start health server in background thread
    health_thread = threading.Thread(target=run_health_server, daemon=True)
    health_thread.start()
    
    print("Worker health server started")
    
    # Run worker in main thread
    print("Starting RQ worker...")
    run_worker()

# app/worker.py
import os
import redis
from rq import Queue
from rq.worker import SimpleWorker  # <- in-process worker, no fork
from app.config import REDIS_URL, QUEUE_NAME

# Extra safety on macOS; avoids ObjC post-fork issues even if other libs fork
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

listen = [QUEUE_NAME]
redis_conn = redis.from_url(REDIS_URL)

if __name__ == "__main__":
    queues = [Queue(name, connection=redis_conn) for name in listen]
    worker = SimpleWorker(queues, connection=redis_conn)  # no forking
    worker.work()

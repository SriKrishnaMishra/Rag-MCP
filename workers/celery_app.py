"""Celery configuration for ingestion, reindex, and evaluation jobs."""
from celery import Celery

from app.config import settings

celery_app = Celery("rag_workers", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.task_default_queue = "rag-jobs"
celery_app.conf.task_track_started = True
celery_app.conf.task_acks_late = True
celery_app.conf.task_reject_on_worker_lost = True
celery_app.conf.worker_prefetch_multiplier = 1
celery_app.conf.broker_transport_options = {"visibility_timeout": 21600}
celery_app.conf.result_backend_transport_options = {"visibility_timeout": 21600}
celery_app.conf.visibility_timeout = 21600
celery_app.conf.task_serializer = "json"
celery_app.conf.result_serializer = "json"
celery_app.conf.accept_content = ["json"]
celery_app.conf.result_expires = 3600
celery_app.conf.imports = ("workers.tasks",)

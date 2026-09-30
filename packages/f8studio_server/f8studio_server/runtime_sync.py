from __future__ import annotations

from .models import DeployJob, JobStatus


def service_was_deployed(job: DeployJob | None, service_id: str) -> bool:
    return (job is not None and job.status in {JobStatus.succeeded, JobStatus.partially_failed}
            and any(item.service_id == service_id and item.success for item in job.service_results))

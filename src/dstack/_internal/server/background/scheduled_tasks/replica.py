"""Scheduled task to sync worker replicas with router replicas (e.g. SGLang PD disaggregation)."""

from typing import Any, Dict, List, Literal, Optional, TypedDict

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from dstack._internal.core.models.configurations import ServiceConfiguration
from dstack._internal.core.models.runs import JobStatus, RunSpec, RunStatus
from dstack._internal.proxy.gateway.services.model_routers.sglang import (
    update_workers_in_router_replica,
)
from dstack._internal.server.db import get_session_ctx
from dstack._internal.server.models import InstanceModel, JobModel, RunModel
from dstack._internal.server.services.jobs import get_job_provisioning_data
from dstack._internal.server.services.replica import get_service_replica_client
from dstack._internal.server.services.runs.replicas import (
    is_replica_registered,
    job_belongs_to_group,
)
from dstack._internal.utils.logging import get_logger

logger = get_logger(__name__)


class WorkerPayloadResult(TypedDict):
    status: Literal["ready", "not_ready"]
    payload: Optional[Dict[str, Any]]


BATCH_SIZE = 10


def _get_router_job(run_model: RunModel, router_group) -> JobModel | None:
    """Return the single router job if found and registered. Services have 1 job per replica."""
    group_name = router_group.name
    assert group_name is not None, "Replica group name is set by validation"
    router_jobs = [
        j
        for j in run_model.jobs
        if job_belongs_to_group(j, group_name) and j.status == JobStatus.RUNNING
    ]
    if not router_jobs or not is_replica_registered(router_jobs):
        return None
    return router_jobs[0]


def _needs_router(run_model: RunModel) -> bool:
    """True if run is a service with at least one replica group that has a router."""
    run_spec = RunSpec.__response__.parse_raw(run_model.run_spec)
    if run_spec.configuration.type != "service":
        return False
    if not isinstance(run_spec.configuration, ServiceConfiguration):
        return False
    replica_groups = run_spec.configuration.replica_groups
    return any(g.router is not None for g in replica_groups)


async def process_all_replicas_registered() -> None:
    """
    For each RUNNING service with router groups, sync registered workers in router
    with each replica. Compares router's GET /workers with our replicas,
    adds missing and removes extras.
    """
    async with get_session_ctx() as session:
        res = await session.execute(
            select(RunModel)
            .where(RunModel.status == RunStatus.RUNNING)
            .options(
                selectinload(RunModel.jobs)
                .selectinload(JobModel.instance)
                .selectinload(InstanceModel.project)
            )
            .order_by(RunModel.last_processed_at.asc())
            .limit(BATCH_SIZE)
        )
        run_models = res.scalars().all()

    runs_to_process = [r for r in run_models if _needs_router(r)]
    for run_model in runs_to_process:
        try:
            await _process_run(run_model)
        except Exception:
            logger.exception(
                "%s/%s: failed to register workers with router",
                run_model.project.name if run_model.project else "?",
                run_model.run_name,
            )


async def _process_run(run_model: RunModel) -> None:
    """Process for updating workers with router replica."""
    run_spec = RunSpec.__response__.parse_raw(run_model.run_spec)
    config = run_spec.configuration
    replica_groups = config.replica_groups
    service_port = config.port.container_port
    # Only one replica in router group, enforced by ServiceConfiguration validation.
    router_group = next((g for g in replica_groups if g.router is not None), None)
    assert router_group is not None  # _needs_router guarantees at least one

    # target_workers: List of dicts the router should have, e.g.:
    #   {"url": "http://10.0.1.246:8000", "worker_type": "prefill", "bootstrap_port": 8998}
    #   {"url": "http://10.0.1.247:8000", "worker_type": "decode"}
    #   {"url": "http://10.0.1.248:8000", "worker_type": "regular"}
    target_workers = await _build_target_workers(run_model, run_spec, replica_groups, service_port)

    router_job = _get_router_job(run_model, router_group)
    if router_job is None:
        return
    try:
        async with get_service_replica_client(router_job) as client:
            await update_workers_in_router_replica(client, target_workers)
    except Exception as e:
        logger.warning(
            "%s: failed to sync workers with router: %r",
            router_job.job_name,
            e,
        )


async def _build_target_workers(
    run_model: RunModel,
    run_spec: RunSpec,
    replica_groups: List,
    service_port: int,
) -> List[Dict[str, Any]]:
    """Build target worker payloads from registered non-router replicas."""
    payloads: List[Dict[str, Any]] = []
    config = run_spec.configuration
    if not isinstance(config, ServiceConfiguration):
        return payloads

    for group in replica_groups:
        if group.router is not None:
            continue
        assert group.name is not None, "Replica group name is set by validation"
        group_name = group.name
        for job in run_model.jobs:
            if not job_belongs_to_group(job, group_name):
                continue
            if job.status != JobStatus.RUNNING:
                continue
            if not is_replica_registered([job]):
                continue
            jpd = get_job_provisioning_data(job)
            if jpd is None:
                continue
            ip = jpd.internal_ip or jpd.hostname
            if not ip:
                continue
            worker_url = f"http://{ip}:{service_port}"
            result = await _get_worker_payload(job, worker_url)
            if result["status"] == "ready" and result["payload"]:
                payloads.append(result["payload"])
            elif result["status"] == "not_ready":
                logger.debug("Worker %s not ready", worker_url)
    return payloads


async def _get_worker_payload(job_model: JobModel, worker_url: str) -> WorkerPayloadResult:
    """Fetch worker info via server_info; return payload for router registration."""
    try:
        async with get_service_replica_client(job_model) as client:
            resp = await client.get(
                "http://dstack/server_info",
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") != "ready":
                    return {"status": "not_ready", "payload": None}
                mode = data.get("disaggregation_mode", "")
                if mode == "prefill":
                    bootstrap_port = data.get("disaggregation_bootstrap_port")
                    return {
                        "status": "ready",
                        "payload": {
                            "url": worker_url,
                            "worker_type": "prefill",
                            "bootstrap_port": bootstrap_port,
                        },
                    }
                elif mode == "decode":
                    return {
                        "status": "ready",
                        "payload": {"url": worker_url, "worker_type": "decode"},
                    }
                else:
                    return {
                        "status": "ready",
                        "payload": {"url": worker_url, "worker_type": "regular"},
                    }
    except Exception as e:
        logger.debug("Could not fetch server_info for worker %s: %r", worker_url, e)
    return {"status": "not_ready", "payload": None}

import shutil
import subprocess
import sys
import time
from typing import List, Optional, Union

import httpx
import psutil

from dstack._internal.core.models.routers import AnyServiceRouterConfig, RouterType
from dstack._internal.proxy.lib.errors import UnexpectedProxyError
from dstack._internal.utils.logging import get_logger

from .base import Router, RouterContext

logger = get_logger(__name__)


class SglangRouter(Router):
    """SGLang router implementation with 1:1 service-to-router."""

    TYPE = RouterType.SGLANG

    def __init__(self, config: AnyServiceRouterConfig, context: RouterContext):
        """Initialize SGLang router.

        Args:
            config: SGLang router configuration (policy, cache_threshold, etc.)
            context: Runtime context for the router (host, port, logging, etc.)
        """
        super().__init__(context=context, config=config)
        self.config = config

    def pid_from_tcp_ipv4_port(self, port: int) -> Optional[int]:
        """
        Return PID of the process listening on the given TCP IPv4 port.
        If no process is found, return None.
        """
        for conn in psutil.net_connections(kind="tcp4"):
            if conn.laddr and conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
                return conn.pid
        return None

    def start(self) -> None:
        try:
            logger.info("Starting sglang-router-new on port %s...", self.context.port)

            # Prometheus port is offset by 10000 from router port to keep it in a separate range
            prometheus_port = self.context.port + 10000

            cmd = [
                sys.executable,
                "-m",
                "sglang_router.launch_router",
                "--host",
                self.context.host,
                "--port",
                str(self.context.port),
                "--prometheus-port",
                str(prometheus_port),
                "--prometheus-host",
                self.context.host,
                "--log-level",
                self.context.log_level,
                "--log-dir",
                str(self.context.log_dir),
                "--policy",
                self.config.policy,
            ]
            if self.config.pd_disaggregation:
                cmd.append("--pd-disaggregation")

            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            time.sleep(2)

            if not self.is_running():
                raise UnexpectedProxyError(
                    f"Failed to start sglang router on port {self.context.port}"
                )

            logger.info(
                "Sglang router started successfully on port %s (prometheus on %s)",
                self.context.port,
                prometheus_port,
            )

        except Exception:
            logger.exception("Failed to start sglang-router")
            raise

    def stop(self) -> None:
        try:
            pid = self.pid_from_tcp_ipv4_port(self.context.port)

            if pid:
                logger.debug(
                    "Stopping sglang-router process (PID: %s) on port %s",
                    pid,
                    self.context.port,
                )
                try:
                    proc = psutil.Process(pid)
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        logger.warning(
                            "Process %s did not terminate gracefully, forcing kill", pid
                        )
                        proc.kill()
                except psutil.NoSuchProcess:
                    logger.debug("sglang-router process %s already exited before stop()", pid)
            else:
                logger.debug("No sglang-router process found on port %s", self.context.port)

            # Clean up router logs
            if self.context.log_dir.exists():
                logger.debug("Cleaning up router logs for port %s...", self.context.port)
                shutil.rmtree(self.context.log_dir, ignore_errors=True)

        except Exception:
            logger.exception("Failed to stop sglang-router")
            raise

    def is_running(self) -> bool:
        """Check if the SGLang router is running and responding to HTTP requests on the assigned port."""
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"http://{self.context.host}:{self.context.port}/workers")
                return response.status_code == 200
        except httpx.RequestError as e:
            logger.debug(
                "Sglang router not responding on port %s: %s",
                self.context.port,
                e,
            )
            return False

    def remove_replicas(self, replica_urls: List[str]) -> None:
        for replica_url in replica_urls:
            self._remove_worker_from_router(replica_url)

    def update_replicas(self, replica_urls: List[str]) -> None:
        """Update replicas for service, replacing the current set."""
        # Query router to get current worker URLs
        current_workers = self._get_router_workers()
        current_worker_urls: set[str] = set()
        for worker in current_workers:
            url = worker.get("url")
            if url and isinstance(url, str):
                # Normalize URL by removing trailing slashes to avoid path artifacts
                normalized_url = url.rstrip("/")
                current_worker_urls.add(normalized_url)
        # Normalize target URLs to ensure consistent comparison
        target_worker_urls = {url.rstrip("/") for url in replica_urls}

        # Workers to add
        workers_to_add = target_worker_urls - current_worker_urls
        # Workers to remove
        workers_to_remove = current_worker_urls - target_worker_urls

        if workers_to_add:
            logger.info(
                "Sglang router update: adding %d workers for router on port %s",
                len(workers_to_add),
                self.context.port,
            )
        if workers_to_remove:
            logger.info(
                "Sglang router update: removing %d workers for router on port %s",
                len(workers_to_remove),
                self.context.port,
            )

        # Add workers
        for worker_url in sorted(workers_to_add):
            success = self._register_worker(worker_url)
            if not success:
                logger.warning("Failed to add worker %s, continuing with others", worker_url)

        # Remove workers
        for worker_url in sorted(workers_to_remove):
            success = self._remove_worker_from_router(worker_url)
            if not success:
                logger.warning("Failed to remove worker %s, continuing with others", worker_url)

    def _get_router_workers(self) -> List[dict]:
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"http://{self.context.host}:{self.context.port}/workers")
                if response.status_code == 200:
                    response_data = response.json()
                    workers = response_data.get("workers", [])
                    return workers
                return []
        except Exception:
            logger.exception("Error getting sglang router workers")
            return []

    def _add_worker_to_router(
        self,
        url: str,
        worker_type: str = "regular",
        bootstrap_port: Optional[int] = None,
    ) -> bool:
        try:
            payload: dict = {"url": url, "worker_type": worker_type}
            if bootstrap_port is not None:
                payload["bootstrap_port"] = bootstrap_port
            with httpx.Client(timeout=5.0) as client:
                response = client.post(
                    f"http://{self.context.host}:{self.context.port}/workers",
                    json=payload,
                )
                if response.status_code == 202:
                    response_data = response.json()
                    if response_data.get("status") == "accepted":
                        logger.info(
                            "Worker %s (type=%s) accepted by sglang router on port %s",
                            url,
                            worker_type,
                            self.context.port,
                        )
                        return True
                    else:
                        logger.error(
                            "Sglang router on port %s failed to accept worker: %s",
                            self.context.port,
                            response_data,
                        )
                        return False
                else:
                    logger.error(
                        "Failed to add worker %s: status %d, %s",
                        url,
                        response.status_code,
                        response.text,
                    )
                    return False
        except Exception:
            logger.exception("Error adding worker %s", url)
            return False

    def _register_worker(self, url: str) -> bool:
        if not self.config.pd_disaggregation:
            return self._add_worker_to_router(url, "regular", None)

        server_info_url = f"{url}/server_info"
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.get(server_info_url)
                if resp.status_code != 200:
                    return False
                data = resp.json()
                if data.get("status") != "ready":
                    return False
                disaggregation_mode = data.get("disaggregation_mode", "")
                if disaggregation_mode == "prefill":
                    worker_type = "prefill"
                    bootstrap_port = data.get("disaggregation_bootstrap_port")
                elif disaggregation_mode == "decode":
                    worker_type = "decode"
                    bootstrap_port = None
                else:
                    worker_type = "regular"
                    bootstrap_port = None
                logger.info(
                    "Registering worker %s (type=%s)",
                    url,
                    worker_type,
                )
                return self._add_worker_to_router(
                    url,
                    worker_type,
                    bootstrap_port,
                )
        except Exception:
            logger.exception("Error registering worker %s", url)
            return False

    def _remove_worker_from_router(self, worker_url: str) -> bool:
        try:
            current_workers = self._get_router_workers()
            worker_id = None
            for worker in current_workers:
                url = worker.get("url")
                if url and isinstance(url, str) and url == worker_url:
                    worker_id = worker.get("id")
                    if worker_id and isinstance(worker_id, str):
                        break
            if not worker_id:
                logger.error("No worker id found for url %s", worker_url)
                return False
            with httpx.Client(timeout=5.0) as client:
                response = client.delete(
                    f"http://{self.context.host}:{self.context.port}/workers/{worker_id}"
                )
                if response.status_code == 202:
                    response_data = response.json()
                    if response_data.get("status") == "accepted":
                        logger.info(
                            "Removed worker %s from sglang router on port %s",
                            worker_url,
                            self.context.port,
                        )
                        return True
                    else:
                        logger.error(
                            "Sglang router on port %s failed to remove worker: %s",
                            self.context.port,
                            response_data,
                        )
                        return False
                else:
                    logger.error(
                        "Failed to remove worker %s: status %d, %s",
                        worker_url,
                        response.status_code,
                        response.text,
                    )
                    return False
        except Exception:
            logger.exception("Error removing worker %s", worker_url)
            return False


# ---------------------------------------------------------------------------
# Async API for server scheduled task (connects to router via SSH tunnel)
# ---------------------------------------------------------------------------

ROUTER_BASE_URL = "http://dstack"


async def update_workers_in_router_replica(
    client: Union[httpx.AsyncClient, object],
    target_workers: List[dict],
) -> None:
    """
    Sync workers with an SGLang router. Compares router's registered workers
    (GET /workers) with target_workers from our replicas. Adds missing, removes extras.

    target_workers: List of dicts with url, worker_type (optional), bootstrap_port (optional).
    """
    current_workers = await _async_get_router_workers(client)
    # Normalize URLs by removing trailing slashes for consistent comparison.
    current_worker_urls: set[str] = set()
    # Map URL -> worker_id for removals. DELETE /workers/{id} requires the router-assigned id;
    url_to_worker_id: dict[str, str] = {}
    for w in current_workers:
        u = w.get("url")
        if u and isinstance(u, str):
            norm = u.rstrip("/")
            current_worker_urls.add(norm)
            worker_id = w.get("id")
            if worker_id and isinstance(worker_id, str):
                url_to_worker_id[norm] = worker_id

    # Map URL -> full payload (url, worker_type, bootstrap_port) for additions.
    target_payload_by_url: dict[str, dict] = {}
    for w in target_workers:
        u = w.get("url")
        if u and isinstance(u, str):
            target_payload_by_url[u.rstrip("/")] = w

    target_urls = set(target_payload_by_url.keys())
    to_add = target_urls - current_worker_urls
    to_remove = current_worker_urls - target_urls

    if to_add:
        logger.info("SGLang router sync: adding %d workers", len(to_add))
    if to_remove:
        logger.info("SGLang router sync: removing %d workers", len(to_remove))

    for url in to_add:
        payload = target_payload_by_url.get(url, {})
        if not await _async_add_worker(client, payload):
            logger.warning("Failed to add worker %s, continuing", url)

    for url in to_remove:
        worker_id = url_to_worker_id.get(url)
        if not worker_id:
            logger.warning("No worker id found for url %s, skipping remove", url)
            continue
        if not await _async_remove_worker(client, worker_id, url):
            logger.warning("Failed to remove worker %s, continuing", url)


async def _async_get_router_workers(client: Union[httpx.AsyncClient, object]) -> List[dict]:
    """Fetch current workers from router via async HTTP client (e.g. SSH tunnel)."""
    try:
        resp = await client.get(f"{ROUTER_BASE_URL}/workers", timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("workers", [])
    except Exception:
        logger.exception("Error getting SGLang router workers")
    return []


async def _async_add_worker(
    client: Union[httpx.AsyncClient, object],
    payload: dict,
) -> bool:
    """Add worker to router via async HTTP client."""
    url = payload.get("url")
    if not url:
        return False
    worker_type = payload.get("worker_type", "regular")
    bootstrap_port = payload.get("bootstrap_port")
    post_payload: dict = {"url": url, "worker_type": worker_type}
    if bootstrap_port is not None:
        post_payload["bootstrap_port"] = bootstrap_port
    try:
        resp = await client.post(
            f"{ROUTER_BASE_URL}/workers",
            json=post_payload,
            timeout=30.0,
        )
        if resp.status_code == 202:
            data = resp.json()
            if data.get("status") == "accepted":
                logger.info("Worker %s (type=%s) accepted by SGLang router", url, worker_type)
                return True
        logger.warning("Failed to add worker %s: status %d %s", url, resp.status_code, resp.text)
    except Exception:
        logger.exception("Error adding worker %s", url)
    return False


async def _async_remove_worker(
    client: Union[httpx.AsyncClient, object],
    worker_id: str,
    worker_url: str,
) -> bool:
    """Remove worker from router via async HTTP client."""
    try:
        resp = await client.delete(
            f"{ROUTER_BASE_URL}/workers/{worker_id}",
            timeout=5.0,
        )
        if resp.status_code == 202:
            data = resp.json()
            if data.get("status") == "accepted":
                logger.info("Removed worker %s from SGLang router", worker_url)
                return True
        logger.warning(
            "Failed to remove worker %s: status %d %s", worker_url, resp.status_code, resp.text
        )
    except Exception:
        logger.exception("Error removing worker %s", worker_url)
    return False

import json
import shutil
import subprocess
import time
import urllib.parse
from typing import Dict, List, Optional

from dstack._internal.core.models.routers import SGLangRouterConfig
from dstack._internal.proxy.gateway.const import DSTACK_DIR_ON_GATEWAY
from dstack._internal.utils.logging import get_logger

from .base import Replica, ReplicaGroup, Router, RouterContext

logger = get_logger(__name__)


class SglangRouter(Router):
    """SGLang router implementation using IGW (Inference Gateway) mode for multi-model serving."""

    TYPE = "sglang"  # Router type identifier

    def __init__(self, router_config: SGLangRouterConfig, context: Optional[RouterContext] = None):
        """Initialize SGLang router.

        Args:
            router_config: SGLang router configuration (policy, cache_threshold, etc.)
            context: Runtime context for the router (host, port, logging, etc.)
        """
        super().__init__(context)
        self.config = router_config
        self._domain_to_model_id: Dict[str, str] = {}  # domain -> model_id
        self._domain_to_ports: Dict[
            str, List[int]
        ] = {}  # domain -> allocated sglang worker ports.
        self._next_worker_port: int = 10001  # Starting port for worker endpoints

    def start(self) -> None:
        """Start the SGLang router process."""
        try:
            logger.info("Starting sglang-router...")

            # Determine active venv (blue or green)
            version_file = DSTACK_DIR_ON_GATEWAY / "version"
            if version_file.exists():
                version = version_file.read_text().strip()
            else:
                version = "blue"

            # Use Python from the active venv
            venv_python = DSTACK_DIR_ON_GATEWAY / version / "bin" / "python3"

            cmd = [
                str(venv_python),
                "-m",
                "sglang_router.launch_router",
                "--host",
                self.context.host,
                "--port",
                str(self.context.port),
                "--enable-igw",
                "--log-level",
                self.context.log_level,
                "--log-dir",
                str(self.context.log_dir),
            ]

            if hasattr(self.config, "policy") and self.config.policy:
                cmd.extend(["--policy", self.config.policy])

            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Wait for router to start
            time.sleep(2)

            # Verify router is running
            if not self.is_running():
                raise Exception("Failed to start sglang router")

            logger.info("Sglang router started successfully")

        except Exception as e:
            logger.error(f"Failed to start sglang-router: {e}")
            raise

    def stop(self) -> None:
        """Stop the SGLang router process."""
        try:
            result = subprocess.run(
                ["pgrep", "-f", "sglang::router"], capture_output=True, timeout=5
            )
            if result.returncode == 0:
                logger.info("Stopping sglang-router process...")
                subprocess.run(["pkill", "-f", "sglang::router"], timeout=5)
            else:
                logger.debug("No sglang-router process found to stop")

            # Clean up router logs
            if self.context.log_dir.exists():
                logger.debug("Cleaning up router logs...")
                shutil.rmtree(self.context.log_dir, ignore_errors=True)
            else:
                logger.debug("No router logs directory found to clean up")

        except Exception as e:
            logger.error(f"Failed to stop sglang-router: {e}")
            raise

    def is_running(self) -> bool:
        """Check if the SGLang router is running and responding to HTTP requests."""
        try:
            result = subprocess.run(
                ["curl", "-s", f"http://{self.context.host}:{self.context.port}/workers"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Error checking sglang router status: {e}")
            return False

    def has_replica_group(self, group_id: str) -> bool:
        """Check if a replica group with the given group_id exists."""
        return group_id in self._domain_to_model_id.values()

    def add_replica_group(self, domain: str, group_id: str, num_replicas: int) -> None:
        """Add a new replica group with the specified number of replicas.

        This allocates ports but does NOT register replicas with the router.
        Use get_replica_group(group_id) to retrieve the allocated ReplicaGroup.
        """
        if self.has_replica_group(group_id):
            raise ValueError(f"Replica group {group_id} already exists")

        # Store domain -> model_id mapping
        self._domain_to_model_id[domain] = group_id

        # Allocate ports for replicas
        allocated_ports = []
        for _ in range(num_replicas):
            allocated_ports.append(self._next_worker_port)
            self._next_worker_port += 1

        self._domain_to_ports[domain] = allocated_ports

        logger.debug(
            f"Allocated replica group {group_id} (domain {domain}) with {num_replicas} replicas "
            f"on ports {allocated_ports}"
        )

    def get_replica_group(self, group_id: str) -> Optional[ReplicaGroup]:
        """Get the ReplicaGroup for a given group_id with allocated replica URLs.

        This is useful after calling add_replica_group to get the allocated URLs.
        """
        # Find domain for this group_id
        domain = None
        for d, model_id in self._domain_to_model_id.items():
            if model_id == group_id:
                domain = d
                break

        if domain is None:
            return None

        # Get allocated ports
        allocated_ports = self._domain_to_ports.get(domain, [])

        # Create Replica objects with URLs
        replicas = [Replica(url=f"http://{self.context.host}:{port}") for port in allocated_ports]

        return ReplicaGroup(id=group_id, replicas=replicas)

    def update_replica_group(self, group: ReplicaGroup) -> None:
        """Update an existing replica group.

        This allocates/deallocates ports but does NOT register replicas with the router.
        """
        # Find domain for this group_id
        domain = None
        for d, model_id in self._domain_to_model_id.items():
            if model_id == group.id:
                domain = d
                break

        if domain is None:
            raise ValueError(f"Replica group {group.id} not found")

        # Update allocated ports based on current replicas
        allocated_ports = [int(r.url.rsplit(":", 1)[-1]) for r in group.replicas]
        self._domain_to_ports[domain] = allocated_ports

        # Update next_worker_port if needed
        if allocated_ports:
            max_port = max(allocated_ports)
            if max_port >= self._next_worker_port:
                self._next_worker_port = max_port + 1

        logger.debug(
            f"Updated replica group {group.id} (domain {domain}) with {len(group.replicas)} replicas "
            f"on ports {allocated_ports}"
        )

    def remove_replica_group(self, domain: str, group_id: str) -> None:
        """Remove a replica group and all its replicas from the router."""
        # Remove all workers for this model_id from the router
        current_workers = self._get_router_workers(group_id)
        for worker in current_workers:
            self._remove_worker_from_router(worker["url"])

        # Clean up internal state
        if domain in self._domain_to_model_id:
            del self._domain_to_model_id[domain]
        if domain in self._domain_to_ports:
            del self._domain_to_ports[domain]

        logger.debug(f"Removed replica group {group_id} (domain {domain})")

    def add_replicas(self, group_id: str, replicas: List[Replica]) -> None:
        """Add replicas to an existing group."""
        for replica in replicas:
            self._add_worker_to_router(replica.url, group_id)

    def remove_replicas(self, group_id: str, replicas: List[Replica]) -> None:
        """Remove replicas from an existing group."""
        for replica in replicas:
            self._remove_worker_from_router(replica.url)

    def update_replicas(self, group_id: str, replicas: List[Replica]) -> None:
        """Update replicas for a group, replacing the current set."""
        # Get current workers for this model_id
        current_workers = self._get_router_workers(group_id)
        current_worker_urls = {worker["url"] for worker in current_workers}

        # Calculate target worker URLs
        target_worker_urls = {replica.url for replica in replicas}

        # Workers to add
        workers_to_add = target_worker_urls - current_worker_urls
        # Workers to remove
        workers_to_remove = current_worker_urls - target_worker_urls

        if workers_to_add:
            logger.info("Sglang router update: adding %d workers", len(workers_to_add))
        if workers_to_remove:
            logger.info("Sglang router update: removing %d workers", len(workers_to_remove))

        # Add workers
        for worker_url in sorted(workers_to_add):
            success = self._add_worker_to_router(worker_url, group_id)
            if not success:
                logger.warning("Failed to add worker %s, continuing with others", worker_url)

        # Remove workers
        for worker_url in sorted(workers_to_remove):
            success = self._remove_worker_from_router(worker_url)
            if not success:
                logger.warning("Failed to remove worker %s, continuing with others", worker_url)

    def list_replica_groups(self) -> List[ReplicaGroup]:
        """List all replica groups managed by this router."""
        groups = []
        for domain, model_id in self._domain_to_model_id.items():
            # Get current workers from router
            workers = self._get_router_workers(model_id)
            replicas = [Replica(url=worker["url"]) for worker in workers]
            groups.append(ReplicaGroup(id=model_id, replicas=replicas))
        return groups

    # Private helper methods

    def _get_router_workers(self, model_id: str) -> List[dict]:
        """Get all workers for a specific model_id from the router."""
        try:
            result = subprocess.run(
                ["curl", "-s", f"http://{self.context.host}:{self.context.port}/workers"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                response = json.loads(result.stdout.decode())
                workers = response.get("workers", [])
                # Filter by model_id
                workers = [w for w in workers if w.get("model_id") == model_id]
                return workers
            return []
        except Exception as e:
            logger.error(f"Error getting sglang router workers: {e}")
            return []

    def _add_worker_to_router(self, worker_url: str, model_id: str) -> bool:
        """Add a single worker to the router."""
        try:
            payload = {"url": worker_url, "worker_type": "regular", "model_id": model_id}
            result = subprocess.run(
                [
                    "curl",
                    "-X",
                    "POST",
                    f"http://{self.context.host}:{self.context.port}/workers",
                    "-H",
                    "Content-Type: application/json",
                    "-d",
                    json.dumps(payload),
                ],
                capture_output=True,
                timeout=5,
            )

            if result.returncode == 0:
                response = json.loads(result.stdout.decode())
                if response.get("status") == "accepted":
                    logger.info("Added worker %s to sglang router", worker_url)
                    return True
                else:
                    logger.error("Failed to add worker %s: %s", worker_url, response)
                    return False
            else:
                logger.error("Failed to add worker %s: %s", worker_url, result.stderr.decode())
                return False
        except Exception as e:
            logger.error(f"Error adding worker {worker_url}: {e}")
            return False

    def _remove_worker_from_router(self, worker_url: str) -> bool:
        """Remove a single worker from the router."""
        try:
            # URL encode the worker URL for the DELETE request
            encoded_url = urllib.parse.quote(worker_url, safe="")

            result = subprocess.run(
                [
                    "curl",
                    "-X",
                    "DELETE",
                    f"http://{self.context.host}:{self.context.port}/workers/{encoded_url}",
                ],
                capture_output=True,
                timeout=5,
            )

            if result.returncode == 0:
                response = json.loads(result.stdout.decode())
                if response.get("status") == "accepted":
                    logger.info("Removed worker %s from sglang router", worker_url)
                    return True
                else:
                    logger.error("Failed to remove worker %s: %s", worker_url, response)
                    return False
            else:
                logger.error("Failed to remove worker %s: %s", worker_url, result.stderr.decode())
                return False
        except Exception as e:
            logger.error(f"Error removing worker {worker_url}: {e}")
            return False

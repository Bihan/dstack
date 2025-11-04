from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel

from dstack._internal.core.models.routers import AnyRouterConfig


class RouterContext(BaseModel):
    """Context for router initialization and configuration."""

    class Config:
        frozen = True

    host: str = "127.0.0.1"
    port: int = 3000
    log_dir: Path = Path("./router_logs")
    log_level: Literal["debug", "info", "warning", "error"] = "info"


class Replica(BaseModel):
    """Represents a single replica (worker) endpoint managed by the router.

    The model field identifies which model this replica serves.
    In SGLang, model = model_id (e.g., "meta-llama/Meta-Llama-3.1-8B-Instruct").
    """

    url: str
    model: str  # (e.g., "meta-llama/Meta-Llama-3.1-8B-Instruct")


class Router(ABC):
    """Abstract base class for router implementations (e.g., SGLang, vLLM).

    A router manages the lifecycle of worker replicas and handles request routing.
    Different router implementations may have different mechanisms for managing
    replicas (HTTP API, configuration files, etc.).
    """

    def __init__(
        self,
        router_config: Optional[AnyRouterConfig] = None,
        context: Optional[RouterContext] = None,
    ):
        """Initialize router with context.

        Args:
            router_config: Optional router configuration (implementation-specific)
            context: Runtime context for the router (host, port, logging, etc.)
        """
        self.context = context or RouterContext()

    @abstractmethod
    def start(self) -> None:
        """Start the router process.

        Raises:
            Exception: If the router fails to start.
        """
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop the router process.

        Raises:
            Exception: If the router fails to stop.
        """
        ...

    @abstractmethod
    def is_running(self) -> bool:
        """Check if the router is currently running and responding.

        Returns:
            True if the router is running and healthy, False otherwise.
        """
        ...

    @abstractmethod
    def register_replicas(self, domain: str, model_id: str, num_replicas: int) -> List[Replica]:
        """Register a model and assign replicas to it (allocate ports/URLs for workers).

        This method handles both new model registration and updates to existing models.
        If the model already exists, it updates the replica count; otherwise, it creates a new model.

        Args:
            domain: The domain name for this service (for reference/namespacing).
            model_id: The model identifier (e.g., "meta-llama/Meta-Llama-3.1-8B-Instruct").
            num_replicas: The number of replicas to allocate for this model.

        Returns:
            List of Replica objects with allocated URLs and model_id set.

        Raises:
            Exception: If allocation fails.
        """
        ...

    @abstractmethod
    def unregister_replicas(self, domain: str) -> None:
        """Unregister replicas for a domain (remove model and unassign all its replicas).

        Args:
            domain: The domain name for this service.

        Raises:
            Exception: If removal fails or domain is not found.
        """
        ...

    @abstractmethod
    def add_replicas(self, replicas: List[Replica]) -> None:
        """Register replicas with the router (actual HTTP API calls to add workers).

        Args:
            replicas: The list of replicas to register. Each replica must have model_id set.

        Raises:
            Exception: If adding replicas fails.
        """
        ...

    @abstractmethod
    def remove_replicas(self, replicas: List[Replica]) -> None:
        """Unregister replicas from the router (actual HTTP API calls to remove workers).

        Args:
            replicas: The list of replicas to unregister. Each replica must have model_id set.

        Raises:
            Exception: If removing replicas fails.
        """
        ...

    @abstractmethod
    def update_replicas(self, replicas: List[Replica]) -> None:
        """Update replicas for a model, replacing the current set.

        Args:
            replicas: The new list of replicas for this model. Each replica must have model_id set.

        Raises:
            Exception: If updating replicas fails.
        """
        ...

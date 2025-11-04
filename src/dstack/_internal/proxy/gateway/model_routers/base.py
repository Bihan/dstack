from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, List, Literal, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
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
    """Represents a single replica (worker) endpoint managed by the router."""

    url: str


class ReplicaGroup(BaseModel):
    """Represents a logical group of replicas that share one identity (id).

    In SGLang, id = model_id. A ReplicaGroup corresponds to a service in dstack.
    Multiple services can share the same router instance (e.g., SGLang's IGW mode).
    """

    id: str
    replicas: List[Replica] = Field(default_factory=list)


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

    # Group lifecycle
    @abstractmethod
    def add_replica_group(self, domain: str, group_id: str, num_replicas: int) -> None:
        """Add a new replica group with the specified number of replicas.

        Args:
            domain: The domain name for this service (for reference/namespacing).
            group_id: The unique identifier for this replica group (e.g., model_id).
            num_replicas: The number of replicas to allocate.

        Raises:
            Exception: If the group already exists or allocation fails.
        """
        ...

    @abstractmethod
    def update_replica_group(self, group: ReplicaGroup) -> None:
        """Update an existing replica group.

        Args:
            group: The ReplicaGroup to update.

        Raises:
            Exception: If the group does not exist or update fails.
        """
        ...

    @abstractmethod
    def remove_replica_group(self, domain: str, group_id: str) -> None:
        """Remove a replica group and all its replicas from the router.

        Args:
            domain: The domain name for this service.
            group_id: The unique identifier for the replica group (e.g., model_id).

        Raises:
            Exception: If removal fails.
        """
        ...

    @abstractmethod
    def add_replicas(self, group_id: str, replicas: List[Replica]) -> None:
        """Add replicas to an existing group.

        Args:
            group_id: The unique identifier for the replica group (e.g., model_id).
            replicas: The list of replicas to add.

        Raises:
            Exception: If adding replicas fails.
        """
        ...

    @abstractmethod
    def remove_replicas(self, group_id: str, replicas: List[Replica]) -> None:
        """Remove replicas from an existing group.

        Args:
            group_id: The unique identifier for the replica group (e.g., model_id).
            replicas: The list of replicas to remove.

        Raises:
            Exception: If removing replicas fails.
        """
        ...

    @abstractmethod
    def update_replicas(self, group_id: str, replicas: List[Replica]) -> None:
        """Update replicas for a group, replacing the current set.

        Args:
            group_id: The unique identifier for the replica group (e.g., model_id).
            replicas: The new list of replicas for this group.

        Raises:
            Exception: If updating replicas fails.
        """
        ...

    @abstractmethod
    def has_replica_group(self, group_id: str) -> bool:
        """Check if a replica group with the given group_id exists.

        Args:
            group_id: The identifier of the replica group (e.g., model_id).

        Returns:
            True if the group exists, False otherwise.
        """
        ...

    @abstractmethod
    def list_replica_groups(self) -> List[ReplicaGroup]:
        """List all replica groups managed by this router.

        Returns:
            A list of all ReplicaGroups.
        """
        ...

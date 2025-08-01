"""
Modified Hotaisle create_instance Implementation

Based on analysis of dstack's architecture, this implementation addresses the connection
issue after server restart by ensuring the shim is properly started and running before
create_instance returns JobProvisioningData.

Key Changes:
1. Install and start shim synchronously within create_instance
2. Wait for shim to be healthy before returning
3. Ensure proper hostname/IP is set for server communication
4. Remove dependency on daemon thread in update_provisioning_data

Background Context:
- Server background tasks use SSH tunnels to communicate with shim on port 3001
- Health checks via _instance_healthcheck function expect shim to be running
- JobProvisioningData with hostname enables immediate server communication
- Current hotaisle backend loses connection after server restart due to
  shim process being tied to SSH session lifetime
"""

import shlex
import subprocess
import tempfile
import time
from typing import Optional

from dstack._internal.core.backends.base.compute import (
    generate_unique_instance_name,
    get_shim_commands,
)
from dstack._internal.core.errors import ProvisioningError
from dstack._internal.core.models.configurations import InstanceConfiguration
from dstack._internal.core.models.instances import InstanceOffer
from dstack._internal.core.models.runs import JobProvisioningData

MAX_INSTANCE_NAME_LEN = 63
SHIM_STARTUP_TIMEOUT = 300  # 5 minutes
SHIM_HEALTH_CHECK_INTERVAL = 10  # seconds


def create_instance(
    self,
    instance_offer: InstanceOffer,
    instance_config: InstanceConfiguration,
) -> JobProvisioningData:
    """
    Modified create_instance that ensures shim is running before returning.

    This addresses the connection issue after server restart by:
    1. Creating VM as before
    2. Waiting for VM to be running
    3. Installing and starting shim synchronously
    4. Verifying shim health before returning
    5. Returning JobProvisioningData with proper hostname
    """
    instance_name = generate_unique_instance_name(
        instance_config, max_length=MAX_INSTANCE_NAME_LEN
    )
    project_ssh_key = instance_config.ssh_keys[0]
    self.api_client.upload_ssh_key(project_ssh_key.public)
    vm_payload = self.get_payload_from_offer(instance_offer.instance)
    vm_data = self.api_client.create_virtual_machine(vm_payload, instance_name)

    # Wait for VM to be running and get IP
    vm_ip = self._wait_for_vm_running(vm_data["name"])
    if not vm_ip:
        raise ProvisioningError(f"VM {instance_name} failed to start or get IP")

    # Install and start shim synchronously
    self._install_and_start_shim(
        hostname=vm_ip,
        project_ssh_public_key=instance_config.ssh_keys[0].public,
        project_ssh_private_key=instance_config.ssh_keys[0].private,
        arch=instance_offer.instance.resources.cpu_arch,
    )

    # Return JobProvisioningData with hostname set for immediate server communication
    return JobProvisioningData(
        backend=instance_offer.backend,
        instance_type=instance_offer.instance,
        instance_id=vm_data["name"],
        hostname=vm_ip,  # Set hostname immediately for server communication
        internal_ip=None,
        region=instance_offer.region,
        price=instance_offer.price,
        username="hotaisle",
        ssh_port=22,
        dockerized=True,
        ssh_proxy=None,
        backend_data=vm_ip,  # Keep IP in backend_data as backup
    )


def _wait_for_vm_running(self, vm_name: str, timeout: int = 300) -> Optional[str]:
    """
    Wait for VM to be in running state and return its IP address.

    Args:
        vm_name: Name of the VM to wait for
        timeout: Maximum time to wait in seconds

    Returns:
        VM IP address if successful, None if timeout or failed
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            vm_state = self.api_client.get_vm_state(vm_name)
            if vm_state == "running":
                # Get VM details to extract IP
                vm_details = self.api_client.get_vm_details(vm_name)
                if vm_details and "ip_address" in vm_details:
                    return vm_details["ip_address"]
            elif vm_state in ["failed", "error", "terminated"]:
                return None
        except Exception as e:
            # Log error but continue waiting
            print(f"Error checking VM state: {e}")

        time.sleep(10)  # Wait 10 seconds before next check

    return None


def _install_and_start_shim(
    self,
    hostname: str,
    project_ssh_public_key: str,
    project_ssh_private_key: str,
    arch: str,
) -> None:
    """
    Install and start shim synchronously, ensuring it's healthy before returning.

    This replaces the daemon thread approach with synchronous execution to ensure
    the shim is running when create_instance returns.

    Args:
        hostname: VM hostname/IP
        project_ssh_public_key: SSH public key for authorization
        project_ssh_private_key: SSH private key for connection
        arch: CPU architecture for shim binary selection
    """
    # First, setup the instance (update packages, etc.)
    self._setup_instance_sync(hostname, project_ssh_private_key)

    # Get shim installation and startup commands
    commands = get_shim_commands(
        authorized_keys=[project_ssh_public_key],
        arch=arch,
    )

    # Execute shim installation commands
    install_command = "sudo sh -c " + shlex.quote(" && ".join(commands))
    self._run_ssh_command_sync(
        hostname=hostname,
        ssh_private_key=project_ssh_private_key,
        command=install_command,
    )

    # Wait for shim to be healthy
    self._wait_for_shim_health(hostname, project_ssh_private_key)


def _setup_instance_sync(self, hostname: str, ssh_private_key: str) -> None:
    """
    Setup instance synchronously (blocking version of _setup_instance).
    """
    setup_commands = ("sudo apt-get update",)
    self._run_ssh_command_sync(
        hostname=hostname, ssh_private_key=ssh_private_key, command=" && ".join(setup_commands)
    )


def _run_ssh_command_sync(self, hostname: str, ssh_private_key: str, command: str) -> None:
    """
    Run SSH command synchronously with proper error handling.

    This is a blocking version of _run_ssh_command that raises exceptions
    on failure instead of silently ignoring them.
    """
    with tempfile.NamedTemporaryFile("w+", 0o600) as f:
        f.write(ssh_private_key)
        f.flush()

        result = subprocess.run(
            [
                "ssh",
                "-F",
                "none",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "ConnectTimeout=30",
                "-o",
                "ServerAliveInterval=10",
                "-o",
                "ServerAliveCountMax=3",
                "-i",
                f.name,
                f"hotaisle@{hostname}",
                command,
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise ProvisioningError(
                f"SSH command failed on {hostname}: {result.stderr} (stdout: {result.stdout})"
            )


def _wait_for_shim_health(self, hostname: str, ssh_private_key: str) -> None:
    """
    Wait for shim to be healthy by checking its HTTP endpoint.

    This ensures the shim is actually running and responsive before create_instance
    returns, so the server's background tasks can immediately establish communication.
    """
    start_time = time.time()
    while time.time() - start_time < SHIM_STARTUP_TIMEOUT:
        try:
            # Check if shim is running and responding on its HTTP port
            health_check_command = "curl -f -s http://localhost:3001/healthcheck"
            self._run_ssh_command_sync(
                hostname=hostname,
                ssh_private_key=ssh_private_key,
                command=health_check_command,
            )
            # If we get here, shim is healthy
            return
        except ProvisioningError:
            # Shim not ready yet, continue waiting
            pass

        time.sleep(SHIM_HEALTH_CHECK_INTERVAL)

    raise ProvisioningError(f"Shim failed to become healthy within {SHIM_STARTUP_TIMEOUT} seconds")


def update_provisioning_data(
    self,
    provisioning_data: JobProvisioningData,
    project_ssh_public_key: str,
    project_ssh_private_key: str,
):
    """
    Modified update_provisioning_data that doesn't start shim in daemon thread.

    Since shim is now started synchronously in create_instance, this method
    only needs to verify the VM is still running and update hostname if needed.
    """
    vm_state = self.api_client.get_vm_state(provisioning_data.instance_id)
    if vm_state == "running":
        # Hostname should already be set from create_instance
        if provisioning_data.hostname is None and provisioning_data.backend_data:
            provisioning_data.hostname = provisioning_data.backend_data

        # No need to start shim in daemon thread - it's already running
        # The server's background tasks can now communicate immediately
    elif vm_state in ["failed", "error", "terminated"]:
        raise ProvisioningError(f"VM {provisioning_data.instance_id} is in {vm_state} state")


"""
Integration Notes:

1. Replace the existing create_instance method in
   src/dstack/_internal/core/backends/hotaisle/compute.py with this implementation

2. Add the new helper methods (_wait_for_vm_running, _install_and_start_shim, etc.)
   to the HotaisleCompute class

3. Update imports to include ProvisioningError:
   from dstack._internal.core.errors import ProvisioningError

4. The modified update_provisioning_data removes the daemon thread approach
   since shim startup is now handled synchronously

5. This ensures that when create_instance returns JobProvisioningData:
   - VM is running
   - Shim is installed and healthy
   - Hostname is set for immediate server communication
   - Server background tasks can establish SSH tunnels immediately

6. Connection issues after server restart should be resolved because:
   - Shim runs as a proper daemon (via get_shim_commands systemd setup)
   - No dependency on SSH session lifetime
   - Server can re-establish tunnels immediately using JobProvisioningData hostname
"""

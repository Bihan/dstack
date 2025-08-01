import shlex
import subprocess
import tempfile
import time
from typing import List, Optional

import gpuhunt
from gpuhunt.providers.hotaisle import HotAisleProvider

from dstack._internal.core.backends.base.compute import (
    Compute,
    ComputeWithCreateInstanceSupport,
    generate_unique_instance_name,
    get_shim_commands,
)
from dstack._internal.core.backends.base.offers import get_catalog_offers
from dstack._internal.core.backends.hotaisle.api_client import HotaisleAPIClient
from dstack._internal.core.backends.hotaisle.models import HotaisleConfig
from dstack._internal.core.models.backends.base import BackendType
from dstack._internal.core.models.instances import (
    InstanceAvailability,
    InstanceConfiguration,
    InstanceOfferWithAvailability,
)
from dstack._internal.core.models.placement import PlacementGroup
from dstack._internal.core.models.runs import JobProvisioningData, Requirements
from dstack._internal.utils.logging import get_logger

logger = get_logger(__name__)

MAX_INSTANCE_NAME_LEN = 60


class HotaisleCompute(
    ComputeWithCreateInstanceSupport,
    Compute,
):
    def __init__(self, config: HotaisleConfig):
        super().__init__()
        self.config = config
        self.api_client = HotaisleAPIClient(config.creds.api_key, config.team_handle)
        self.catalog = gpuhunt.Catalog(balance_resources=False, auto_reload=False)
        self.catalog.add_provider(
            HotAisleProvider(api_key=config.creds.api_key, team_handle=config.team_handle)
        )

    def get_offers(
        self, requirements: Optional[Requirements] = None
    ) -> List[InstanceOfferWithAvailability]:
        offers = get_catalog_offers(
            backend=BackendType.HOTAISLE,
            locations=self.config.regions or None,
            requirements=requirements,
            catalog=self.catalog,
        )
        offers = [
            InstanceOfferWithAvailability(
                **offer.dict(), availability=InstanceAvailability.AVAILABLE
            )
            for offer in offers
        ]
        return offers

    def get_payload_from_offer(self, instance_type) -> dict:
        # Only two instance types are available in Hotaisle with CPUs: 8-core and 13-core. Other fields are
        # not configurable.
        cpu_cores = instance_type.resources.cpus
        if cpu_cores == 8:
            cpu_model = "Xeon Platinum 8462Y+"
            frequency = 2800000000
        else:  # cpu_cores == 13
            cpu_model = "Xeon Platinum 8470"
            frequency = 2000000000

        return {
            "cpu_cores": cpu_cores,
            "cpus": {
                "count": 1,
                "manufacturer": "Intel",
                "model": cpu_model,
                "cores": cpu_cores,
                "frequency": frequency,
            },
            "disk_capacity": 13194139533312,
            "ram_capacity": 240518168576,
            "gpus": [
                {
                    "count": len(instance_type.resources.gpus),
                    "manufacturer": "AMD",
                    "model": "MI300X",
                }
            ],
        }

    # MODIFIED create_instance method - commented out for revert
    # def create_instance(
    #     self,
    #     instance_offer: InstanceOfferWithAvailability,
    #     instance_config: InstanceConfiguration,
    #     placement_group: Optional[PlacementGroup],
    # ) -> JobProvisioningData:
    #     """
    #     Modified create_instance that ensures shim is running before returning.
    #
    #     This addresses the connection issue after server restart by:
    #     1. Creating VM as before
    #     2. Waiting for VM to be running
    #     3. Installing and starting shim synchronously
    #     4. Verifying shim health before returning
    #     5. Returning JobProvisioningData with proper hostname
    #     """
    #     instance_name = generate_unique_instance_name(
    #         instance_config, max_length=MAX_INSTANCE_NAME_LEN
    #     )
    #     project_ssh_key = instance_config.ssh_keys[0]
    #     self.api_client.upload_ssh_key(project_ssh_key.public)
    #     vm_payload = self.get_payload_from_offer(instance_offer.instance)
    #     vm_data = self.api_client.create_virtual_machine(vm_payload, instance_name)
    #
    #     # Wait for VM to be running and get IP
    #     vm_ip = self._wait_for_vm_running(vm_data["name"])
    #     if not vm_ip:
    #         # Fallback to using the IP from vm_data if _wait_for_vm_running doesn't work
    #         vm_ip = vm_data.get("ip_address")
    #         if not vm_ip:
    #             raise Exception(f"VM {instance_name} failed to start or get IP")
    #
    #     logger.info(f"VM {instance_name} is running with IP: {vm_ip}")
    #
    #     # Install and start shim synchronously
    #     # Note: We need to get the project SSH private key from elsewhere since
    #     # instance_config.ssh_keys may not contain the private key
    #     # We'll need to handle this in update_provisioning_data or get it from project
    #     logger.info(f"Skipping shim installation in create_instance for now - will be handled in update_provisioning_data")
    #
    #     # Return JobProvisioningData with hostname set for immediate server communication
    #     return JobProvisioningData(
    #         backend=instance_offer.backend,
    #         instance_type=instance_offer.instance,
    #         instance_id=vm_data["name"],
    #         hostname=vm_ip,  # Set hostname immediately for server communication
    #         internal_ip=None,
    #         region=instance_offer.region,
    #         price=instance_offer.price,
    #         username="hotaisle",
    #         ssh_port=22,
    #         dockerized=True,
    #         ssh_proxy=None,
    #         backend_data=vm_ip,  # Keep IP in backend_data as backup
    #     )

    def create_instance(
        self,
        instance_offer: InstanceOfferWithAvailability,
        instance_config: InstanceConfiguration,
        placement_group: Optional[PlacementGroup],
    ) -> JobProvisioningData:
        instance_name = generate_unique_instance_name(
            instance_config, max_length=MAX_INSTANCE_NAME_LEN
        )
        project_ssh_key = instance_config.ssh_keys[0]
        self.api_client.upload_ssh_key(project_ssh_key.public)
        vm_payload = self.get_payload_from_offer(instance_offer.instance)
        vm_data = self.api_client.create_virtual_machine(vm_payload, instance_name)

        # Start shim installation in create_instance using nohup for true daemonization
        hostname = vm_data["ip_address"]
        commands = get_shim_commands(
            authorized_keys=[project_ssh_key.public],
            arch=instance_offer.instance.resources.cpu_arch,
        )

        # Debug: Log the individual shim commands
        logger.info(f"Individual shim commands for {vm_data['name']}:")
        for i, cmd in enumerate(commands):
            logger.info(f"  Command {i + 1}: {cmd}")

        launch_command = "sudo sh -c " + shlex.quote(" && ".join(commands))

        # Debug: Log the final launch command
        logger.info(f"Final launch command for {vm_data['name']}: {launch_command}")

        # Hardcode private key path for testing
        import os

        private_key_path = os.path.expanduser("~/.dstack/ssh/id_rsa")

        # Use nohup to daemonize shim installation independently of server process
        _start_runner_with_nohup(
            hostname=hostname,
            project_ssh_private_key_path=private_key_path,
            launch_command=launch_command,
            instance_id=vm_data["name"],
        )
        logger.info(
            f"Started shim installation for {vm_data['name']} using nohup with key: {private_key_path}"
        )

        return JobProvisioningData(
            backend=instance_offer.backend,
            instance_type=instance_offer.instance,
            instance_id=vm_data["name"],
            hostname=vm_data["ip_address"],  # Set hostname immediately for server communication
            internal_ip=None,
            region=instance_offer.region,
            price=instance_offer.price,
            username="hotaisle",
            ssh_port=22,
            dockerized=True,
            ssh_proxy=None,
            backend_data=vm_data["ip_address"],
        )

    # MODIFIED update_provisioning_data method - commented out for revert
    # def update_provisioning_data(
    #     self,
    #     provisioning_data: JobProvisioningData,
    #     project_ssh_public_key: str,
    #     project_ssh_private_key: str,
    # ):
    #     """
    #     Modified update_provisioning_data that installs shim synchronously.
    #
    #     Since create_instance doesn't have access to the private key, we handle
    #     shim installation here where both public and private keys are available.
    #     """
    #     vm_state = self.api_client.get_vm_state(provisioning_data.instance_id)
    #     if vm_state == "running":
    #         # Hostname should already be set from create_instance
    #         if provisioning_data.hostname is None and provisioning_data.backend_data:
    #             provisioning_data.hostname = provisioning_data.backend_data
    #
    #         # Install and start shim synchronously instead of using daemon thread
    #         logger.info(f"Installing shim on {provisioning_data.hostname} synchronously")
    #         try:
    #             self._install_and_start_shim(
    #                 hostname=provisioning_data.hostname,
    #                 project_ssh_public_key=project_ssh_public_key,
    #                 project_ssh_private_key=project_ssh_private_key,
    #                 arch=provisioning_data.instance_type.resources.cpu_arch,
    #             )
    #             logger.info(f"Shim successfully installed on {provisioning_data.hostname}")
    #         except Exception as e:
    #             logger.error(f"Failed to install shim on {provisioning_data.hostname}: {e}")
    #             raise
    #
    #     elif vm_state in ["failed", "error", "terminated"]:
    #         logger.error(f"VM {provisioning_data.instance_id} is in {vm_state} state")

    def update_provisioning_data(
        self,
        provisioning_data: JobProvisioningData,
        project_ssh_public_key: str,
        project_ssh_private_key: str,
    ):
        vm_state = self.api_client.get_vm_state(provisioning_data.instance_id)
        if vm_state == "running":
            # Hostname already set in create_instance - shim installation already started
            logger.info(
                f"VM {provisioning_data.instance_id} is running, hostname: {provisioning_data.hostname}"
            )
        elif vm_state in ["failed", "error", "terminated"]:
            logger.error(f"VM {provisioning_data.instance_id} is in {vm_state} state")

    def terminate_instance(
        self, instance_id: str, region: str, backend_data: Optional[str] = None
    ):
        vm_name = instance_id
        self.api_client.terminate_virtual_machine(vm_name)


def _wait_for_vm_running(instance_id: str, api_client, max_wait_time: int = 300):
    """Wait for VM to be in running state, with timeout."""
    start_time = time.time()
    while time.time() - start_time < max_wait_time:
        try:
            vm_state = api_client.get_vm_state(instance_id)
            logger.info(f"VM {instance_id} state: {vm_state}")

            if vm_state == "running":
                return True
            elif vm_state in ["failed", "error", "terminated"]:
                raise Exception(f"VM {instance_id} failed with state: {vm_state}")

            time.sleep(10)  # Wait 10 seconds before checking again
        except Exception as e:
            logger.warning(f"Error checking VM state for {instance_id}: {e}")
            time.sleep(10)

    raise Exception(f"VM {instance_id} did not reach running state within {max_wait_time} seconds")


def _start_runner_with_nohup(
    hostname: str,
    project_ssh_private_key_path: str,
    launch_command: str,
    instance_id: str,
):
    """
    Start shim installation using nohup to make it independent of the server process.
    This ensures the installation continues even if the dstack server restarts.
    """
    # Create a script that waits for VM to be running, then installs shim
    # Escape the launch command for safe bash usage
    escaped_launch_command = launch_command.replace("'", "'\"'\"'")

    script_content = f"""#!/bin/bash
# Wait for VM to be running
echo "Waiting for VM {instance_id} to be accessible via SSH..."
while true; do
    # Try to connect and check if VM is ready
    if ssh -F none -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i {project_ssh_private_key_path} hotaisle@{hostname} "echo 'VM ready'" 2>/dev/null; then
        echo "VM {instance_id} is ready, starting shim installation..."
        break
    fi
    echo "VM {instance_id} not ready yet, waiting 10 seconds..."
    sleep 10
done

# Setup instance
echo "Setting up instance..."
echo "Running: ssh -F none -o StrictHostKeyChecking=no -i {project_ssh_private_key_path} hotaisle@{hostname} 'sudo apt-get update'"
ssh -F none -o StrictHostKeyChecking=no -i {project_ssh_private_key_path} hotaisle@{hostname} "sudo apt-get update"

# Launch shim
echo "Launching shim..."
echo "Shim launch command: {launch_command}"
echo "Escaped launch command: {escaped_launch_command}"
echo "Running: ssh -F none -o StrictHostKeyChecking=no -i {project_ssh_private_key_path} hotaisle@{hostname} '{escaped_launch_command}'"
ssh -F none -o StrictHostKeyChecking=no -i {project_ssh_private_key_path} hotaisle@{hostname} '{escaped_launch_command}'

# Check if shim is running
echo "Checking if shim process is running..."
ssh -F none -o StrictHostKeyChecking=no -i {project_ssh_private_key_path} hotaisle@{hostname} 'ps aux | grep -E "(dstack|shim)" | grep -v grep || echo "No dstack/shim processes found"'

# Check if shim port is listening
echo "Checking if shim port 3001 is listening..."
ssh -F none -o StrictHostKeyChecking=no -i {project_ssh_private_key_path} hotaisle@{hostname} 'netstat -tlnp | grep :3001 || echo "Port 3001 not listening"'

echo "Shim installation completed for {instance_id}"
"""

    # Write script to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(script_content)
        script_path = f.name

    # Log the script content for debugging
    logger.info(f"Created shim installation script for {instance_id}:")
    logger.info(f"Script path: {script_path}")
    logger.info(f"Original launch command: {launch_command}")
    logger.info(f"Escaped launch command: {escaped_launch_command}")
    # Don't log the full script content as it's very long, just log key parts
    logger.info(f"Script will execute: ssh hotaisle@{hostname} '{escaped_launch_command}'")

    # Make script executable
    import os
    import stat

    os.chmod(script_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)

    # Create log file for script output
    log_file = f"/tmp/shim_install_{instance_id}.log"

    # Run script with nohup to make it independent of server process
    nohup_command = ["nohup", "bash", script_path]

    # Start the process detached from parent but log output
    with open(log_file, "w") as log_f:
        subprocess.Popen(
            nohup_command,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,  # Create new session on Unix
        )

    logger.info(f"Started nohup shim installation script: {script_path}")
    logger.info(f"Script output will be logged to: {log_file}")


def _start_runner(
    hostname: str,
    project_ssh_private_key_path: str,
    launch_command: str,
    instance_id: str,
    api_client,
):
    # Wait for VM to be in running state before proceeding
    logger.info(f"Waiting for VM {instance_id} to be in running state...")
    _wait_for_vm_running(instance_id, api_client)
    logger.info(f"VM {instance_id} is running, starting shim installation...")

    _setup_instance(
        hostname=hostname,
        ssh_private_key_path=project_ssh_private_key_path,
    )
    _launch_runner(
        hostname=hostname,
        ssh_private_key_path=project_ssh_private_key_path,
        launch_command=launch_command,
    )


def _setup_instance(
    hostname: str,
    ssh_private_key_path: str,
):
    setup_commands = ("sudo apt-get update",)
    _run_ssh_command(
        hostname=hostname,
        ssh_private_key_path=ssh_private_key_path,
        command=" && ".join(setup_commands),
    )


def _launch_runner(
    hostname: str,
    ssh_private_key_path: str,
    launch_command: str,
):
    _run_ssh_command(
        hostname=hostname,
        ssh_private_key_path=ssh_private_key_path,
        command=launch_command,
    )


def _run_ssh_command(hostname: str, ssh_private_key_path: str, command: str):
    subprocess.run(
        [
            "ssh",
            "-F",
            "none",
            "-o",
            "StrictHostKeyChecking=no",
            "-i",
            ssh_private_key_path,
            f"hotaisle@{hostname}",
            command,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

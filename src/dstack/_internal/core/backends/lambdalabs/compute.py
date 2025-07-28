import hashlib
import shlex
import subprocess
import tempfile
from threading import Thread
from typing import Dict, List, Optional

from dstack._internal.core.backends.base.compute import (
    Compute,
    ComputeWithCreateInstanceSupport,
    generate_unique_instance_name,
    get_shim_commands,
)
from dstack._internal.core.backends.base.offers import get_catalog_offers
from dstack._internal.core.backends.lambdalabs.api_client import LambdaAPIClient
from dstack._internal.core.backends.lambdalabs.models import LambdaConfig
from dstack._internal.core.models.backends.base import BackendType
from dstack._internal.core.models.instances import (
    InstanceAvailability,
    InstanceConfiguration,
    InstanceOffer,
    InstanceOfferWithAvailability,
)
from dstack._internal.core.models.placement import PlacementGroup
from dstack._internal.core.models.runs import JobProvisioningData, Requirements

MAX_INSTANCE_NAME_LEN = 60


class LambdaCompute(
    ComputeWithCreateInstanceSupport,
    Compute,
):
    def __init__(self, config: LambdaConfig):
        super().__init__()
        self.config = config
        self.api_client = LambdaAPIClient(config.creds.api_key)

    def get_offers(
        self, requirements: Optional[Requirements] = None
    ) -> List[InstanceOfferWithAvailability]:
        offers = get_catalog_offers(
            backend=BackendType.LAMBDA,
            locations=self.config.regions or None,
            requirements=requirements,
        )
        offers_with_availability = self._get_offers_with_availability(offers)
        return offers_with_availability

    def create_instance(
        self,
        instance_offer: InstanceOfferWithAvailability,
        instance_config: InstanceConfiguration,
        placement_group: Optional[PlacementGroup],
    ) -> JobProvisioningData:
        # Step 1: Generate instance name (keep Lambda pattern)
        instance_name = generate_unique_instance_name(
            instance_config, max_length=MAX_INSTANCE_NAME_LEN
        )

        # Step 2: Get SSH key (keep Lambda pattern for now)
        project_ssh_key = instance_config.ssh_keys[0]

        # Step 3: Upload SSH key to Hotaisle FIRST
        print(f"Uploading SSH key to Hotaisle for instance {instance_name}...")
        ssh_upload_success = self._upload_ssh_key_to_hotaisle(project_ssh_key.public)
        if not ssh_upload_success:
            raise Exception("Failed to upload SSH key to Hotaisle")

        # Step 4: Create Hotaisle VM payload (adapted from hotaisle/compute.py)
        vm_payload = {
            "cpu_cores": 13,  # Hardcoded like in hotaisle/compute.py
            "cpus": {
                "count": 1,
                "manufacturer": "Intel",
                "model": "Xeon Platinum 8470",
                "cores": 13,
                "frequency": 2600000000,
            },
            "disk_capacity": 13194139533312,  # Exact value from working hotaisle example
            "ram_capacity": 240518168576,  # Exact value from working hotaisle example
            "gpus": [{"count": 1, "manufacturer": "AMD", "model": "MI300X"}],
        }

        # Step 5: Call Hotaisle API instead of Lambda API
        # Replace Lambda's launch_instances with Hotaisle's create_vm
        import os

        import requests

        hotaisle_api_key = os.getenv("HOTAISLE_API_KEY")
        if not hotaisle_api_key:
            raise Exception("HOTAISLE_API_KEY environment variable is required")

        url = "https://admin.hotaisle.app/api/teams/dstackai/virtual_machines/"
        headers = {
            "accept": "application/json",
            "Authorization": hotaisle_api_key,
            "Content-Type": "application/json",
        }

        print(f"Creating Hotaisle VM: {instance_name}")
        print(f"Payload: {vm_payload}")

        response = requests.post(url, headers=headers, json=vm_payload, timeout=60)

        if response.status_code not in [200, 201]:
            raise Exception(
                f"Hotaisle VM creation failed: {response.status_code} - {response.text}"
            )

        vm_data = response.json()
        print(f"Hotaisle VM created: {vm_data['name']} at {vm_data['ip_address']}")

        # Step 6: Return JobProvisioningData (adapted for Hotaisle)
        return JobProvisioningData(
            backend=instance_offer.backend,
            instance_type=instance_offer.instance,
            instance_id=vm_data["name"],  # Use VM name as instance_id
            hostname=None,  # Set to None initially - will be set by update_provisioning_data
            internal_ip=None,
            region=instance_offer.region,
            price=instance_offer.price,
            username="hotaisle",  # Hotaisle username instead of "ubuntu"
            ssh_port=vm_data["ssh_access"]["port"],  # Use port from response
            dockerized=True,
            ssh_proxy=None,
            backend_data=vm_data[
                "ip_address"
            ],  # Store IP in backend_data for update_provisioning_data
        )

    def _upload_ssh_key_to_hotaisle(self, public_key: str) -> bool:
        """Upload SSH public key to Hotaisle user account"""
        import os

        import requests

        hotaisle_api_key = os.getenv("HOTAISLE_API_KEY")
        if not hotaisle_api_key:
            raise Exception("HOTAISLE_API_KEY environment variable is required")

        url = "https://admin.hotaisle.app/api/user/ssh_keys/"
        headers = {
            "accept": "application/json",
            "Authorization": hotaisle_api_key,
            "Content-Type": "application/json",
        }

        payload = {"authorized_key": public_key}

        print("Uploading SSH key to Hotaisle...")
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)

            if response.status_code in [200, 201]:
                print("SSH key uploaded successfully!")
                return True
            elif response.status_code == 409:
                print("SSH key already exists in Hotaisle account - continuing...")
                return True  # This is success - key already exists
            else:
                print(f"SSH key upload failed. Status: {response.status_code}")
                print(f"Response: {response.text}")
                return False

        except Exception as e:
            print(f"SSH key upload exception: {e}")
            return False

    def update_provisioning_data(
        self,
        provisioning_data: JobProvisioningData,
        project_ssh_public_key: str,
        project_ssh_private_key: str,
    ):
        # Check Hotaisle VM state directly
        vm_state = self._check_hotaisle_vm_state(provisioning_data.instance_id)
        if vm_state == "running":
            # Set hostname from backend_data now that VM is ready
            if provisioning_data.hostname is None and provisioning_data.backend_data:
                provisioning_data.hostname = provisioning_data.backend_data
                print(
                    f"Hotaisle VM {provisioning_data.instance_id} is running at {provisioning_data.hostname}"
                )

            # Get shim commands and start installation
            commands = get_shim_commands(
                authorized_keys=[project_ssh_public_key],
                arch=provisioning_data.instance_type.resources.cpu_arch,
            )
            # shim is assumed to be run under root
            launch_command = "sudo sh -c " + shlex.quote(" && ".join(commands))
            print(f"Launching shim installation with command: {launch_command}")
            # Start shim installation in background thread
            thread = Thread(
                target=_start_runner,
                kwargs={
                    "hostname": provisioning_data.hostname,
                    "project_ssh_private_key": project_ssh_private_key,
                    "launch_command": launch_command,
                },
                daemon=True,
            )
            thread.start()
        else:
            print(f"Hotaisle VM {provisioning_data.instance_id} not ready yet, state: {vm_state}")

    def _check_hotaisle_vm_state(self, vm_name: str) -> str:
        """Check Hotaisle VM state using the state API"""
        import os

        import requests

        hotaisle_api_key = os.getenv("HOTAISLE_API_KEY")
        if not hotaisle_api_key:
            print("Warning: HOTAISLE_API_KEY not found, assuming VM is ready")
            return "running"

        url = f"https://admin.hotaisle.app/api/teams/dstackai/virtual_machines/{vm_name}/state/"
        headers = {
            "accept": "application/json",
            "Authorization": hotaisle_api_key,
        }

        try:
            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code == 200:
                state_data = response.json()
                vm_state = state_data.get("state", "unknown")
                print(f"Hotaisle VM {vm_name} state: {vm_state}")
                return vm_state
            else:
                print(
                    f"Failed to get VM state. Status: {response.status_code}, Response: {response.text}"
                )
                return "unknown"

        except Exception as e:
            print(f"Exception checking VM state: {e}")
            return "unknown"

    def _run_hotaisle_setup_commands(self, hostname: str, ssh_private_key: str) -> bool:
        """Run basic setup commands on Hotaisle VM (adapted from hotaisle/compute.py)"""
        setup_commands = (
            "mkdir -p /home/hotaisle/.dstack",
            "sudo apt-get update",
        )

        print(f"Running setup commands on {hostname}...")
        combined_command = " && ".join(setup_commands)

        # Create a temporary file for the SSH key and run the command
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
                    "-i",
                    f.name,
                    f"hotaisle@{hostname}",
                    combined_command,
                ],
                capture_output=True,
                text=True,
            )

        if result.returncode == 0:
            print("Setup commands completed successfully!")
            return True
        else:
            print(f"Setup commands failed: {result.stderr}")
            return False

    def terminate_instance(
        self, instance_id: str, region: str, backend_data: Optional[str] = None
    ):
        pass
        # self.api_client.terminate_instances(instance_ids=[instance_id])

    def _get_offers_with_availability(
        self, offers: List[InstanceOffer]
    ) -> List[InstanceOfferWithAvailability]:
        instance_availability = {
            instance_name: [
                region["name"] for region in details["regions_with_capacity_available"]
            ]
            for instance_name, details in self.api_client.list_instance_types().items()
        }
        availability_offers = []
        for offer in offers:
            availability = InstanceAvailability.NOT_AVAILABLE
            if offer.region in instance_availability.get(offer.instance.name, []):
                availability = InstanceAvailability.AVAILABLE
            availability_offers.append(
                InstanceOfferWithAvailability(**offer.dict(), availability=availability)
            )
        return availability_offers


def _add_project_ssh_key(
    api_client: LambdaAPIClient,
    project_ssh_public_key: str,
) -> str:
    ssh_keys = api_client.list_ssh_keys()
    ssh_key_names: List[str] = [k["name"] for k in ssh_keys]
    project_key_name = _add_ssh_key(api_client, ssh_key_names, project_ssh_public_key)
    return project_key_name


def _add_ssh_key(api_client: LambdaAPIClient, ssh_key_names: List[str], public_key: str) -> str:
    key_name = _get_ssh_key_name(public_key)
    if key_name in ssh_key_names:
        return key_name
    api_client.add_ssh_key(name=key_name, public_key=public_key)
    return key_name


def _get_ssh_key_name(public_key: str) -> str:
    return hashlib.sha1(public_key.encode()).hexdigest()[-16:]


def _get_instance_info(api_client: LambdaAPIClient, instance_id: str) -> Optional[Dict]:
    # TODO: use get instance https://cloud.lambdalabs.com/api/v1/docs#operation/getInstance
    instances = api_client.list_instances()
    instance_id_to_instance_map = {i["id"]: i for i in instances}
    instance = instance_id_to_instance_map.get(instance_id)
    return instance


def _start_runner(
    hostname: str,
    project_ssh_private_key: str,
    launch_command: str,
):
    _setup_instance(
        hostname=hostname,
        ssh_private_key=project_ssh_private_key,
    )
    _launch_runner(
        hostname=hostname,
        ssh_private_key=project_ssh_private_key,
        launch_command=launch_command,
    )


def _setup_instance(
    hostname: str,
    ssh_private_key: str,
):
    # Use simplified setup commands for Hotaisle (no NVIDIA toolkit needed for AMD GPUs)
    setup_commands = (
        "mkdir -p /home/hotaisle/.dstack",  # Changed to hotaisle home directory
        "sudo mkdir -p /root/.dstack",  # Also create root directory for shim
        "sudo apt-get update",  # Just update packages
    )
    _run_ssh_command(
        hostname=hostname, ssh_private_key=ssh_private_key, command=" && ".join(setup_commands)
    )


def _launch_runner(
    hostname: str,
    ssh_private_key: str,
    launch_command: str,
):
    _run_ssh_command(
        hostname=hostname,
        ssh_private_key=ssh_private_key,
        command=launch_command,
    )


def _run_ssh_command(hostname: str, ssh_private_key: str, command: str):
    with tempfile.NamedTemporaryFile("w+", 0o600) as f:
        f.write(ssh_private_key)
        f.flush()
        subprocess.run(
            [
                "ssh",
                "-F",
                "none",
                "-o",
                "StrictHostKeyChecking=no",
                "-i",
                f.name,
                f"hotaisle@{hostname}",
                command,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

import shlex
import subprocess
import tempfile
from threading import Thread
from typing import List, Optional

from dstack._internal.core.backends.base.compute import (
    Compute,
    ComputeWithCreateInstanceSupport,
    generate_unique_instance_name,
    get_shim_commands,
)
from dstack._internal.core.backends.base.offers import get_catalog_offers
from dstack._internal.core.backends.hotaisle.models import HotaisleConfig
from dstack._internal.core.models.backends.base import BackendType
from dstack._internal.core.models.instances import (
    InstanceAvailability,
    InstanceConfiguration,
    InstanceOffer,
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
        # Set up catalog with Hotaisle provider (following VastAI pattern)
        import gpuhunt
        from gpuhunt.providers.hotaisle import HotAisleProvider

        self.catalog = gpuhunt.Catalog(balance_resources=False, auto_reload=False)
        self.catalog.add_provider(
            HotAisleProvider(api_key=config.creds.api_key, team_handle=config.team_handle)
        )

    def get_offers(
        self, requirements: Optional[Requirements] = None
    ) -> List[InstanceOfferWithAvailability]:
        print(f"DEBUG: Getting offers for requirements: {requirements}")
        print(f"DEBUG: Config regions: {self.config.regions}")

        offers = get_catalog_offers(
            backend=BackendType.HOTAISLE,
            locations=self.config.regions or None,
            requirements=requirements,
            catalog=self.catalog,
        )
        print(f"DEBUG: Found {len(offers)} offers: {offers}")
        offers_with_availability = self._get_offers_with_availability(offers)
        print(f"DEBUG: Offers with availability: {len(offers_with_availability)}")
        return offers_with_availability

    def create_instance(
        self,
        instance_offer: InstanceOfferWithAvailability,
        instance_config: InstanceConfiguration,
        placement_group: Optional[PlacementGroup],
    ) -> JobProvisioningData:
        # Step 1: Generate instance name
        instance_name = generate_unique_instance_name(
            instance_config, max_length=MAX_INSTANCE_NAME_LEN
        )

        # Step 2: Get SSH key
        project_ssh_key = instance_config.ssh_keys[0]

        # Step 3: Upload SSH key to Hotaisle FIRST (before VM creation)
        print(f"Uploading SSH key to Hotaisle for instance {instance_name}...")
        ssh_upload_success = self._upload_ssh_key_to_hotaisle(project_ssh_key.public)
        if not ssh_upload_success:
            print("Warning: SSH key upload failed, but continuing with VM creation")

        # Step 4: Create Hotaisle VM payload based on instance offer
        instance_type = instance_offer.instance

        # For now, use hardcoded values as in your working example
        # TODO: Map instance_offer details to appropriate Hotaisle VM specs
        vm_payload = {
            "cpu_cores": instance_type.resources.cpus,
            "cpus": {
                "count": 1,
                "manufacturer": "Intel",
                "model": "Xeon Platinum 8470",
                "cores": instance_type.resources.cpus,
                "frequency": 2600000000,
            },
            "disk_capacity": int(instance_type.resources.disk.size_mib * 1024 * 1024)
            if instance_type.resources.disk
            else 13194139533312,
            "ram_capacity": int(instance_type.resources.memory_mib * 1024 * 1024),
            "gpus": [
                {
                    "count": len(instance_type.resources.gpus),
                    "manufacturer": "AMD",  # Hotaisle uses AMD GPUs
                    "model": instance_type.resources.gpus[0].name
                    if instance_type.resources.gpus
                    else "MI300X",
                }
            ]
            if instance_type.resources.gpus
            else [],
        }

        # Step 5: Call Hotaisle API to create VM

        import requests

        # Use credentials from config instead of environment variables
        api_key = self.config.creds.api_key
        team_handle = self.config.team_handle

        url = f"https://admin.hotaisle.app/api/teams/{team_handle}/virtual_machines/"
        headers = {
            "accept": "application/json",
            "Authorization": api_key,
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
        print(f"SSH access info: {vm_data.get('ssh_access', 'Not found')}")

        # Step 6: Return JobProvisioningData
        return JobProvisioningData(
            backend=instance_offer.backend,
            instance_type=instance_offer.instance,
            instance_id=vm_data["name"],  # Use VM name as instance_id
            hostname=None,  # Set to None initially - will be set by update_provisioning_data
            internal_ip=None,
            region=instance_offer.region,
            price=instance_offer.price,
            username="hotaisle",  # Hotaisle username
            ssh_port=vm_data["ssh_access"]["port"],  # Use port from response
            dockerized=True,
            ssh_proxy=None,
            backend_data=vm_data[
                "ip_address"
            ],  # Store IP in backend_data for update_provisioning_data
        )

    def _upload_ssh_key_to_hotaisle(self, public_key: str) -> bool:
        """Upload SSH public key to Hotaisle user account"""
        import requests

        # Use credentials from config
        api_key = self.config.creds.api_key

        url = "https://admin.hotaisle.app/api/user/ssh_keys/"
        headers = {
            "accept": "application/json",
            "Authorization": api_key,
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

            # Ensure SSH key is uploaded and accessible for this VM
            print(f"Ensuring SSH key is available for VM {provisioning_data.instance_id}...")
            self._ensure_ssh_key_available(
                project_ssh_public_key, provisioning_data.hostname, project_ssh_private_key
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
        import requests

        # Use credentials from config
        api_key = self.config.creds.api_key
        team_handle = self.config.team_handle

        url = (
            f"https://admin.hotaisle.app/api/teams/{team_handle}/virtual_machines/{vm_name}/state/"
        )
        headers = {
            "accept": "application/json",
            "Authorization": api_key,
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

    def _ensure_ssh_key_available(self, public_key: str, hostname: str, private_key: str) -> bool:
        """Ensure SSH key is available on the VM, with retries and re-upload if needed"""
        import time

        max_retries = 3
        retry_delay = 10  # seconds

        for attempt in range(max_retries):
            print(f"Testing SSH access to {hostname} (attempt {attempt + 1}/{max_retries})...")

            # Test SSH access
            with tempfile.NamedTemporaryFile("w+", 0o600) as f:
                f.write(private_key)
                f.flush()
                result = subprocess.run(
                    [
                        "ssh",
                        "-o",
                        "ConnectTimeout=10",
                        "-o",
                        "StrictHostKeyChecking=no",
                        "-i",
                        f.name,
                        f"hotaisle@{hostname}",
                        "echo 'SSH access successful'",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )

            if result.returncode == 0:
                print(f"SSH access to {hostname} successful!")
                return True
            else:
                print(f"SSH access failed: {result.stderr}")

            if attempt < max_retries - 1:
                # Re-upload SSH key and wait before retry
                print(f"Re-uploading SSH key and waiting {retry_delay} seconds before retry...")
                self._upload_ssh_key_to_hotaisle(public_key)
                time.sleep(retry_delay)

        print(f"Failed to establish SSH access to {hostname} after {max_retries} attempts")
        return False

    def terminate_instance(
        self, instance_id: str, region: str, backend_data: Optional[str] = None
    ):
        """Terminate Hotaisle VM instance"""
        import requests

        # Use credentials from config
        api_key = self.config.creds.api_key
        team_handle = self.config.team_handle

        # instance_id contains the VM name from Hotaisle
        vm_name = instance_id
        url = f"https://admin.hotaisle.app/api/teams/{team_handle}/virtual_machines/{vm_name}/"
        headers = {
            "accept": "application/json",
            "Authorization": api_key,
        }

        print(f"Terminating Hotaisle VM: {vm_name}")

        try:
            response = requests.delete(url, headers=headers, timeout=30)

            if response.status_code in [200, 204]:
                print(f"Hotaisle VM {vm_name} terminated successfully!")
            elif response.status_code == 404:
                print(f"Hotaisle VM {vm_name} not found (may already be terminated)")
            else:
                print(f"Failed to terminate VM {vm_name}. Status: {response.status_code}")
                print(f"Response: {response.text}")
                raise Exception(f"VM termination failed: {response.status_code} - {response.text}")

        except requests.RequestException as e:
            print(f"Network error while terminating VM {vm_name}: {e}")
            raise Exception(f"Network error during VM termination: {e}")
        except Exception as e:
            print(f"Unexpected error while terminating VM {vm_name}: {e}")
            raise

    def _get_offers_with_availability(
        self, offers: List[InstanceOffer]
    ) -> List[InstanceOfferWithAvailability]:
        # For online providers like Hotaisle, we assume all offers are available
        # since gpuhunt fetches them dynamically
        return [
            InstanceOfferWithAvailability(
                **offer.dict(), availability=InstanceAvailability.AVAILABLE
            )
            for offer in offers
        ]


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

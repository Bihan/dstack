import argparse
import shlex
import subprocess
import tempfile
from threading import Thread
from typing import Any, Dict

import requests

# Import the REAL shim commands from dstack
try:
    from dstack._internal.core.backends.base.compute import get_shim_commands

    REAL_SHIM_AVAILABLE = True
    print("Using REAL dstack shim commands")
except ImportError:
    REAL_SHIM_AVAILABLE = False
    print("Warning: Real dstack shim not available, using simplified version")


def upload_ssh_key_to_hotaisle(api_key: str, public_key: str) -> bool:
    """Upload SSH public key to Hotaisle user account"""
    url = "https://admin.hotaisle.app/api/user/ssh_keys/"

    headers = {
        "accept": "application/json",
        "Authorization": api_key,
        "Content-Type": "application/json",
    }

    payload = {"authorized_key": public_key}

    print("Uploading SSH key to Hotaisle...")
    response = requests.post(url, headers=headers, json=payload, timeout=30)

    if response.status_code in [200, 201]:
        print("SSH key uploaded successfully!")
        return True
    elif response.status_code == 409:
        print("SSH key already exists in Hotaisle account - skipping upload")
        return True
    else:
        print(f"SSH key upload failed. Status: {response.status_code}")
        print(f"Response: {response.text}")
        return False


def _run_ssh_command(hostname: str, ssh_private_key: str, command: str):
    """Run SSH command on Hotaisle VM"""
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
                command,
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"SSH command failed: {result.stderr}")
        else:
            print("SSH command completed successfully")

        return result


def run_setup_commands(hostname: str, ssh_private_key: str):
    """Run basic setup commands on Hotaisle VM"""
    setup_commands = (
        "mkdir -p /home/hotaisle/.dstack",
        "sudo apt-get update",
    )

    print(f"Running setup commands on {hostname}...")
    combined_command = " && ".join(setup_commands)
    result = _run_ssh_command(hostname, ssh_private_key, combined_command)

    if result.returncode == 0:
        print("Setup commands completed successfully!")
    else:
        print("Setup commands failed!")

    return result.returncode == 0


def get_simplified_shim_commands(authorized_keys: list[str], arch: str = "amd64") -> list[str]:
    """Fallback simplified shim commands if real ones not available"""
    dstack_runner_url = f"https://dstack-runner-downloads.s3.eu-west-1.amazonaws.com/latest/binaries/dstack-runner-linux-{arch}"

    commands = [
        f"sudo curl -L -o /usr/local/bin/dstack-runner {dstack_runner_url}",
        "sudo chmod +x /usr/local/bin/dstack-runner",
        "sudo mkdir -p /root/.dstack",
        f"echo '{chr(10).join(authorized_keys)}' | sudo tee /root/.ssh/authorized_keys",
        "sudo chmod 600 /root/.ssh/authorized_keys",
        "sudo nohup /usr/local/bin/dstack-runner --log-level 6 start --http-port 10999 --ssh-port 10022 --temp-dir /tmp/runner --home-dir /root --working-dir /root > /dev/null 2>&1 &",
    ]

    return commands


def run_dstack_runner(
    hostname: str, ssh_private_key: str, public_key: str, use_real_shim: bool = True
):
    """Start dstack runner on Hotaisle VM using REAL or simplified shim commands"""
    print(f"Starting dstack runner on {hostname}...")

    if use_real_shim and REAL_SHIM_AVAILABLE:
        print("Using REAL dstack shim commands...")
        # Use the actual dstack shim commands with proper parameters
        commands = get_shim_commands(
            authorized_keys=[public_key],
            arch="amd64",  # AMD64 architecture for Hotaisle VMs
            is_privileged=False,
            pjrt_device=None,
            base_path="/home/hotaisle",  # Use hotaisle home directory
            bin_path=None,
            backend_shim_env=None,
        )
    else:
        print("Using simplified shim commands...")
        commands = get_simplified_shim_commands(authorized_keys=[public_key])

    # Combine commands like Lambda Labs does
    launch_command = "sudo sh -c " + shlex.quote(" && ".join(commands))

    print(f"Generated {len(commands)} shim commands")
    print(f"First few commands: {commands[:3] if len(commands) > 3 else commands}")

    # Run in background thread
    thread = Thread(
        target=_start_runner,
        kwargs={
            "hostname": hostname,
            "ssh_private_key": ssh_private_key,
            "launch_command": launch_command,
        },
        daemon=True,
    )
    thread.start()
    print("dstack runner startup initiated in background...")


def _start_runner(hostname: str, ssh_private_key: str, launch_command: str):
    """Start dstack runner (mimics Lambda Labs pattern)"""
    print(f"Executing runner setup on {hostname}...")
    result = _run_ssh_command(hostname, ssh_private_key, launch_command)

    if result.returncode == 0:
        print(f"dstack runner started successfully on {hostname}")
    else:
        print(f"dstack runner startup failed on {hostname}")
        print(f"Error: {result.stderr}")


def create_hotaisle_instance(
    api_key: str,
    team_name: str,
    cpu_cores: int,
    ram_gb: int,
    disk_gb: int,
    gpu_model: str = "MI300X",
    gpu_count: int = 1,
    ssh_public_key: str = None,
    ssh_private_key: str = None,
    start_runner: bool = False,
    use_real_shim: bool = True,  # New parameter
) -> Dict[str, Any]:
    """Create a Hotaisle VM instance with SSH key setup and optional dstack runner"""

    # Step 1: Upload SSH key if provided
    if ssh_public_key:
        if not upload_ssh_key_to_hotaisle(api_key, ssh_public_key):
            raise Exception("Failed to upload SSH key to Hotaisle")

    # Step 2: Create VM
    url = f"https://admin.hotaisle.app/api/teams/{team_name}/virtual_machines/"

    headers = {
        "accept": "application/json",
        "Authorization": api_key,
        "Content-Type": "application/json",
    }

    payload = {
        "cpu_cores": 13,
        "cpus": {
            "count": 1,
            "manufacturer": "Intel",
            "model": "Xeon Platinum 8470",
            "cores": 13,
            "frequency": 2600000000,
        },
        "disk_capacity": 13194139533312,
        "ram_capacity": 240518168576,
        "gpus": [{"count": gpu_count, "manufacturer": "AMD", "model": gpu_model}],
    }

    print("Creating Hotaisle VM...")
    response = requests.post(url, headers=headers, json=payload, timeout=60)

    if response.status_code in [200, 201]:
        vm_data = response.json()
        print("VM created successfully!")
        print(f"VM Name: {vm_data['name']}")
        print(f"IP Address: {vm_data['ip_address']}")

        # Step 3: Run setup commands
        if ssh_private_key:
            print("\nRunning setup commands...")
            success = run_setup_commands(vm_data["ip_address"], ssh_private_key)
            if not success:
                print("Warning: Setup commands failed, but continuing...")

            # Step 4: Start dstack runner if requested
            if start_runner and ssh_public_key:
                print("\nStarting dstack runner...")
                run_dstack_runner(
                    hostname=vm_data["ip_address"],
                    ssh_private_key=ssh_private_key,
                    public_key=ssh_public_key,
                    use_real_shim=use_real_shim,
                )

        return vm_data
    else:
        print(f"Failed to create VM. Status: {response.status_code}")
        print(f"Response: {response.text}")
        raise Exception(f"VM creation failed: {response.status_code} - {response.text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a Hotaisle VM instance with dstack runner"
    )
    parser.add_argument("--api_key", required=True, help="Hotaisle API key")
    parser.add_argument("--team_name", default="dstackai", help="Team name (default: dstackai)")
    parser.add_argument("--ssh_public_key", help="SSH public key to upload")
    parser.add_argument("--ssh_private_key_file", help="SSH private key file for setup commands")
    parser.add_argument("--start_runner", action="store_true", help="Start dstack runner on VM")
    parser.add_argument(
        "--use_real_shim",
        action="store_true",
        default=True,
        help="Use real dstack shim commands (default: True)",
    )
    parser.add_argument(
        "--use_simplified_shim",
        action="store_true",
        help="Use simplified shim commands instead of real ones",
    )

    args = parser.parse_args()

    # Default dstack SSH keys
    default_ssh_public_key = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC1FHT9bAq53ev9Pte2jYFt6VIP450JALUeNHI7gptlpqpzqkqNjQB0di/QaQMj+2LURV3Lo1Qx5gHyqs+J3t16U7/P0if077pK5jBwPzXySwKenbHg7HDycLuRX+JY3ALsqZL8r5u2DtsEHd+YrIVQ8n3/zRdXxUln9+X4bS3/D/BeKwoBmxYachVf8r/8rdwk/7Kj39bnJ+mn8hY2/VqMIGaYkQd3cO2Zgbg3DOjeX+PwotPCmnEiKG3BxXDUlvlO30ZQ/UTCqsoTdlTMMUASh4lI1eeW5azYlZggE0MVI7O8Bju+VY1pNhyeX7CzKoC4MPB0i3qUtVmGrUFvnx2f dstack"

    # Read private key if file provided
    ssh_private_key = None
    if args.ssh_private_key_file:
        with open(args.ssh_private_key_file, "r") as f:
            ssh_private_key = f.read()

    ssh_public_key = args.ssh_public_key or default_ssh_public_key
    use_real_shim = not args.use_simplified_shim

    print(
        f"Testing Hotaisle VM creation with {'REAL' if use_real_shim else 'SIMPLIFIED'} dstack shim..."
    )

    try:
        vm_result = create_hotaisle_instance(
            api_key=args.api_key,
            team_name=args.team_name,
            cpu_cores=13,
            ram_gb=224,
            disk_gb=12800,
            ssh_public_key=ssh_public_key,
            ssh_private_key=ssh_private_key,
            start_runner=args.start_runner,
            use_real_shim=use_real_shim,
        )

        print("\nSuccess! VM is ready:")
        print(f"ssh hotaisle@{vm_result['ip_address']}")

        if args.start_runner:
            print("dstack runner should be starting...")

    except Exception as e:
        print(f"Error: {e}")


# import argparse
# import subprocess
# import tempfile
# from typing import Any, Dict

# import requests


# def upload_ssh_key_to_hotaisle(api_key: str, public_key: str) -> bool:
#     """Upload SSH public key to Hotaisle user account"""
#     url = "https://admin.hotaisle.app/api/user/ssh_keys/"

#     headers = {
#         "accept": "application/json",
#         "Authorization": api_key,
#         "Content-Type": "application/json",
#     }

#     payload = {
#         "authorized_key": public_key
#     }

#     print("Uploading SSH key to Hotaisle...")
#     response = requests.post(url, headers=headers, json=payload, timeout=30)

#     if response.status_code in [200, 201]:
#         print("SSH key uploaded successfully!")
#         return True
#     elif response.status_code == 409:
#         print("SSH key already exists in Hotaisle account - skipping upload")
#         return True  # This is actually success - key is already there
#     else:
#         print(f"SSH key upload failed. Status: {response.status_code}")
#         print(f"Response: {response.text}")
#         return False


# def _run_ssh_command(hostname: str, ssh_private_key: str, command: str):
#     """Run SSH command on Hotaisle VM"""
#     with tempfile.NamedTemporaryFile("w+", 0o600) as f:
#         f.write(ssh_private_key)
#         f.flush()
#         result = subprocess.run(
#             [
#                 "ssh",
#                 "-F",
#                 "none",
#                 "-o",
#                 "StrictHostKeyChecking=no",
#                 "-i",
#                 f.name,
#                 f"hotaisle@{hostname}",  # Use hotaisle username
#                 command,
#             ],
#             capture_output=True,
#             text=True,
#         )

#         if result.returncode != 0:
#             print(f"SSH command failed: {result.stderr}")
#         else:
#             print(f"SSH command completed successfully")

#         return result


# def run_setup_commands(hostname: str, ssh_private_key: str):
#     """Run basic setup commands on Hotaisle VM"""
#     setup_commands = (
#         "mkdir -p /home/hotaisle/.dstack",  # Changed to hotaisle home directory
#         "sudo apt-get update",              # Just update packages
#     )

#     print(f"Running setup commands on {hostname}...")
#     combined_command = " && ".join(setup_commands)
#     result = _run_ssh_command(hostname, ssh_private_key, combined_command)

#     if result.returncode == 0:
#         print("Setup commands completed successfully!")
#     else:
#         print("Setup commands failed!")

#     return result.returncode == 0


# def create_hotaisle_instance(
#     api_key: str,
#     team_name: str,
#     cpu_cores: int,
#     ram_gb: int,
#     disk_gb: int,
#     gpu_model: str = "MI300X",
#     gpu_count: int = 1,
#     ssh_public_key: str = None,
#     ssh_private_key: str = None,
# ) -> Dict[str, Any]:
#     """
#     Create a Hotaisle VM instance with SSH key setup and basic configuration

#     Args:
#         api_key: Hotaisle authorization key
#         team_name: Team name (e.g., 'dstackai')
#         cpu_cores: Number of CPU cores
#         ram_gb: RAM in GB
#         disk_gb: Disk size in GB
#         gpu_model: GPU model (default: MI300X)
#         gpu_count: Number of GPUs (default: 1)
#         ssh_public_key: SSH public key to upload (optional)
#         ssh_private_key: SSH private key for setup commands (optional)

#     Returns:
#         VM response dict with name, ip_address, ssh_access, etc.
#     """

#     # Step 1: Upload SSH key if provided
#     if ssh_public_key:
#         if not upload_ssh_key_to_hotaisle(api_key, ssh_public_key):
#             raise Exception("Failed to upload SSH key to Hotaisle")

#     # Step 2: Create VM
#     url = f"https://admin.hotaisle.app/api/teams/{team_name}/virtual_machines/"

#     headers = {
#         "accept": "application/json",
#         "Authorization": api_key,
#         "Content-Type": "application/json",
#     }

#     payload = {
#         "cpu_cores": 13,
#         "cpus": {
#             "count": 1,
#             "manufacturer": "Intel",
#             "model": "Xeon Platinum 8470",
#             "cores": 13,
#             "frequency": 2600000000,
#         },
#         "disk_capacity": 13194139533312,  # Exact value from working curl
#         "ram_capacity": 240518168576,  # Exact value from working curl
#         "gpus": [{"count": gpu_count, "manufacturer": "AMD", "model": gpu_model}],
#     }

#     print("Creating Hotaisle VM...")
#     print(f"URL: {url}")
#     print(f"Payload: {payload}")

#     response = requests.post(url, headers=headers, json=payload, timeout=60)

#     if response.status_code == 200 or response.status_code == 201:
#         vm_data = response.json()
#         print("VM created successfully!")
#         print(f"VM Name: {vm_data['name']}")
#         print(f"IP Address: {vm_data['ip_address']}")
#         print(f"SSH: hotaisle@{vm_data['ip_address']}")

#         # Step 3: Run setup commands if private key provided
#         if ssh_private_key:
#             print("\nRunning setup commands...")
#             success = run_setup_commands(vm_data['ip_address'], ssh_private_key)
#             if not success:
#                 print("Warning: Setup commands failed, but VM was created successfully")

#         return vm_data
#     else:
#         print(f"Failed to create VM. Status: {response.status_code}")
#         print(f"Response: {response.text}")
#         raise Exception(f"VM creation failed: {response.status_code} - {response.text}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Create a Hotaisle VM instance with setup")
#     parser.add_argument("--api_key", required=True, help="Hotaisle API key")
#     parser.add_argument("--team_name", default="dstackai", help="Team name (default: dstackai)")
#     parser.add_argument("--cpu_cores", type=int, default=13, help="Number of CPU cores (default: 13)")
#     parser.add_argument("--ram_gb", type=int, default=224, help="RAM in GB (default: 224)")
#     parser.add_argument("--disk_gb", type=int, default=12800, help="Disk size in GB (default: 12800)")
#     parser.add_argument("--gpu_model", default="MI300X", help="GPU model (default: MI300X)")
#     parser.add_argument("--gpu_count", type=int, default=1, help="Number of GPUs (default: 1)")
#     parser.add_argument("--ssh_public_key", help="SSH public key to upload")
#     parser.add_argument("--ssh_private_key_file", help="SSH private key file for setup commands")

#     args = parser.parse_args()

#     # Default dstack SSH keys
#     default_ssh_public_key = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC1FHT9bAq53ev9Pte2jYFt6VIP450JALUeNHI7gptlpqpzqkqNjQB0di/QaQMj+2LURV3Lo1Qx5gHyqs+J3t16U7/P0if077pK5jBwPzXySwKenbHg7HDycLuRX+JY3ALsqZL8r5u2DtsEHd+YrIVQ8n3/zRdXxUln9+X4bS3/D/BeKwoBmxYachVf8r/8rdwk/7Kj39bnJ+mn8hY2/VqMIGaYkQd3cO2Zgbg3DOjeX+PwotPCmnEiKG3BxXDUlvlO30ZQ/UTCqsoTdlTMMUASh4lI1eeW5azYlZggE0MVI7O8Bju+VY1pNhyeX7CzKoC4MPB0i3qUtVmGrUFvnx2f dstack"

#     # Read private key if file provided
#     ssh_private_key = None
#     if args.ssh_private_key_file:
#         with open(args.ssh_private_key_file, 'r') as f:
#             ssh_private_key = f.read()

#     ssh_public_key = args.ssh_public_key or default_ssh_public_key

#     print("Testing Hotaisle VM creation with SSH setup...")

#     try:
#         vm_result = create_hotaisle_instance(
#             api_key=args.api_key,
#             team_name=args.team_name,
#             cpu_cores=args.cpu_cores,
#             ram_gb=args.ram_gb,
#             disk_gb=args.disk_gb,
#             gpu_model=args.gpu_model,
#             gpu_count=args.gpu_count,
#             ssh_public_key=ssh_public_key,
#             ssh_private_key=ssh_private_key,
#         )

#         print("\nSuccess! VM is ready:")
#         print(f"ssh hotaisle@{vm_result['ip_address']}")

#     except Exception as e:
#         print(f"Error: {e}")

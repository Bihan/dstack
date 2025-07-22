import argparse
from typing import Any, Dict

import requests


def create_hotaisle_instance(
    api_key: str,
    team_name: str,
    cpu_cores: int,
    ram_gb: int,
    disk_gb: int,
    gpu_model: str = "MI300X",
    gpu_count: int = 1,
) -> Dict[str, Any]:
    """
    Create a Hotaisle VM instance

    Args:
        api_key: Hotaisle authorization key
        team_name: Team name (e.g., 'dstackai')
        cpu_cores: Number of CPU cores
        ram_gb: RAM in GB
        disk_gb: Disk size in GB
        gpu_model: GPU model (default: MI300X)
        gpu_count: Number of GPUs (default: 1)

    Returns:
        VM response dict with name, ip_address, ssh_access, etc.
    """
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
        "disk_capacity": 13194139533312,  # Exact value from working curl
        "ram_capacity": 240518168576,  # Exact value from working curl
        "gpus": [{"count": gpu_count, "manufacturer": "AMD", "model": gpu_model}],
    }

    print("Creating Hotaisle VM...")
    print(f"URL: {url}")
    print(f"Payload: {payload}")

    response = requests.post(url, headers=headers, json=payload, timeout=60)

    if response.status_code == 200 or response.status_code == 201:
        vm_data = response.json()
        print("VM created successfully!")
        print(f"VM Name: {vm_data['name']}")
        print(f"IP Address: {vm_data['ip_address']}")
        print(f"SSH: hotaisle@{vm_data['ip_address']}")
        return vm_data
    else:
        print(f"Failed to create VM. Status: {response.status_code}")
        print(f"Response: {response.text}")
        raise Exception(f"VM creation failed: {response.status_code} - {response.text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a Hotaisle VM instance")
    parser.add_argument("--api_key", required=True, help="Hotaisle API key")
    parser.add_argument("--team_name", default="dstackai", help="Team name (default: dstackai)")
    parser.add_argument(
        "--cpu_cores", type=int, default=13, help="Number of CPU cores (default: 13)"
    )
    parser.add_argument("--ram_gb", type=int, default=224, help="RAM in GB (default: 224)")
    parser.add_argument(
        "--disk_gb", type=int, default=12800, help="Disk size in GB (default: 12800)"
    )
    parser.add_argument("--gpu_model", default="MI300X", help="GPU model (default: MI300X)")
    parser.add_argument("--gpu_count", type=int, default=1, help="Number of GPUs (default: 1)")
    args = parser.parse_args()

    print("Testing Hotaisle VM creation...")

    try:
        vm_result = create_hotaisle_instance(
            api_key=args.api_key,
            team_name=args.team_name,
            cpu_cores=args.cpu_cores,
            ram_gb=args.ram_gb,
            disk_gb=args.disk_gb,
            gpu_model=args.gpu_model,
            gpu_count=args.gpu_count,
        )

        print("\nSuccess! You can SSH with:")
        print(f"ssh hotaisle@{vm_result['ip_address']}")

    except Exception as e:
        print(f"Error: {e}")

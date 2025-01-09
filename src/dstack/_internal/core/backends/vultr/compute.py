import json
from typing import List, Optional

import requests

from dstack._internal.core.backends.base import Compute
from dstack._internal.core.backends.base.compute import (
    get_instance_name,
    get_shim_commands,
)
from dstack._internal.core.backends.base.offers import get_catalog_offers
from dstack._internal.core.backends.vultr.api_client import VultrApiClient
from dstack._internal.core.backends.vultr.config import VultrConfig
from dstack._internal.core.errors import BackendError, ProvisioningError
from dstack._internal.core.models.backends.base import BackendType
from dstack._internal.core.models.instances import (
    InstanceAvailability,
    InstanceConfiguration,
    InstanceOffer,
    InstanceOfferWithAvailability,
    SSHKey,
)
from dstack._internal.core.models.runs import Job, JobProvisioningData, Requirements, Run
from dstack._internal.core.models.volumes import Volume
from dstack._internal.utils.logging import get_logger

logger = get_logger(__name__)


class VultrCompute(Compute):
    def __init__(self, config: VultrConfig):
        self.config = config
        self.api_client = VultrApiClient(config.creds.api_key)

    def get_offers(
        self, requirements: Optional[Requirements] = None
    ) -> List[InstanceOfferWithAvailability]:
        offers = get_catalog_offers(
            backend=BackendType.VULTR,
            requirements=requirements,
            locations=self.config.regions or None,
            extra_filter=_supported_instances,
        )
        offers = [
            InstanceOfferWithAvailability(
                **offer.dict(), availability=InstanceAvailability.AVAILABLE
            )
            for offer in offers
        ]
        return offers

    def run_job(
        self,
        run: Run,
        job: Job,
        instance_offer: InstanceOfferWithAvailability,
        project_ssh_public_key: str,
        project_ssh_private_key: str,
        volumes: List[Volume],
    ) -> JobProvisioningData:
        instance_config = InstanceConfiguration(
            project_name=run.project_name,
            instance_name=get_instance_name(run, job),
            ssh_keys=[SSHKey(public=project_ssh_public_key.strip())],
            user=run.user,
        )
        return self.create_instance(instance_offer, instance_config)

    def create_instance(
        self, instance_offer: InstanceOfferWithAvailability, instance_config: InstanceConfiguration
    ) -> JobProvisioningData:
        public_keys = instance_config.get_public_keys()
        commands = get_shim_commands(authorized_keys=public_keys)
        shim_commands = "#!/bin/sh\n" + " ".join([" && ".join(commands)])
        instance_id = self.api_client.launch_instance(
            region=instance_offer.region,
            label=instance_config.instance_name,
            plan=instance_offer.instance.name,
            startup_script=shim_commands,
            public_keys=public_keys,
        )

        launched_instance = JobProvisioningData(
            backend=instance_offer.backend,
            instance_type=instance_offer.instance,
            instance_id=instance_id,
            hostname=None,
            internal_ip=None,
            region=instance_offer.region,
            price=instance_offer.price,
            ssh_port=22,
            username="root",
            ssh_proxy=None,
            dockerized=True,
            backend_data=json.dumps(
                {
                    "plan_type": "bare-metal"
                    if "vbm" in instance_offer.instance.name
                    else "vm_instance"
                }
            ),
        )
        return launched_instance

    def terminate_instance(
        self, instance_id: str, region: str, backend_data: Optional[str] = None
    ) -> None:
        plan_type = json.loads(backend_data)["plan_type"]
        try:
            self.api_client.terminate_instance(instance_id=instance_id, plan_type=plan_type)
        except requests.HTTPError as e:
            raise BackendError(e.response.text)

    def update_provisioning_data(
        self,
        provisioning_data: JobProvisioningData,
        project_ssh_public_key: str,
        project_ssh_private_key: str,
    ):
        plan_type = json.loads(provisioning_data.backend_data)["plan_type"]
        instance_data = self.api_client.get_instance(provisioning_data.instance_id, plan_type)
        # Access specific fields
        instance_status = instance_data["status"]
        instance_main_ip = instance_data["main_ip"]
        if instance_status == "active":
            provisioning_data.hostname = instance_main_ip
        if instance_status == "failed":
            raise ProvisioningError("VM entered FAILED state")


def _supported_instances(offer: InstanceOffer) -> bool:
    if offer.instance.resources.spot:
        return False
    for instance in [
        "vbm-112c-2048gb-8-a100-gpu",
        "vbm-112c-2048gb-8-h100-gpu",
        "vbm-128c-2048gb-amd",
        "vbm-24c-256gb-amd",
        "vbm-24c-384gb-amd",
        "vbm-256c-2048gb-8-mi300x-gpu",
        "vbm-32c-755gb-amd",
        "vbm-48c-1024gb-4-a100-gpu",
        "vbm-4c-32gb",
        "vbm-64c-1536gb-amd",
        "vbm-64c-2048gb-8-l40-gpu",
        "vbm-6c-32gb",
        "vbm-8c-132gb",
        "vbm-8c-132gb-v2",
        "vc2-16c-64gb",
        "vc2-16c-64gb-sc1",
        "vc2-1c-0.5gb-free",
        "vc2-1c-1gb",
        "vc2-1c-1gb-sc1",
        "vc2-1c-2gb",
        "vc2-1c-2gb-sc1",
        "vc2-24c-96gb",
        "vc2-24c-96gb-sc1",
        "vc2-2c-2gb",
        "vc2-2c-2gb-sc1",
        "vc2-2c-4gb",
        "vc2-2c-4gb-sc1",
        "vc2-4c-8gb",
        "vc2-4c-8gb-sc1",
        "vc2-6c-16gb",
        "vc2-6c-16gb-sc1",
        "vc2-8c-32gb",
        "vc2-8c-32gb-sc1",
        "vcg-a100-12c-120g-80vram",
        "vcg-a100-1c-12g-8vram",
        "vcg-a100-1c-6g-4vram",
        "vcg-a100-24c-240g-160vram",
        "vcg-a100-2c-15g-10vram",
        "vcg-a100-3c-30g-20vram",
        "vcg-a100-48c-480g-320vram",
        "vcg-a100-6c-60g-40vram",
        "vcg-a100-96c-960g-640vram",
        "vcg-a16-12c-128g-32vram",
        "vcg-a16-24c-256g-64vram",
        "vcg-a16-2c-16g-4vram",
        "vcg-a16-2c-8g-2vram",
        "vcg-a16-3c-32g-8vram",
        "vcg-a16-48c-496g-128vram",
        "vcg-a16-6c-64g-16vram",
        "vcg-a16-96c-960g-256vram",
        "vcg-a40-12c-60g-24vram",
        "vcg-a40-1c-5g-2vram",
        "vcg-a40-24c-120g-48vram",
        "vcg-a40-2c-10g-4vram",
        "vcg-a40-4c-20g-8vram",
        "vcg-a40-6c-30g-12vram",
        "vcg-a40-8c-40g-16vram",
        "vcg-a40-96c-480g-192vram",
        "vcg-l40s-128c-1500g-384vram",
        "vcg-l40s-16c-180g-48vram",
        "vcg-l40s-32c-375g-96vram",
        "vcg-l40s-64c-750g-192vram",
        "vhf-12c-48gb",
        "vhf-12c-48gb-sc1",
        "vhf-16c-58gb",
        "vhf-16c-58gb-sc1",
        "vhf-1c-1gb",
        "vhf-1c-1gb-sc1",
        "vhf-1c-2gb",
        "vhf-1c-2gb-sc1",
        "vhf-2c-2gb",
        "vhf-2c-2gb-sc1",
        "vhf-2c-4gb",
        "vhf-2c-4gb-sc1",
        "vhf-3c-8gb",
        "vhf-3c-8gb-sc1",
        "vhf-4c-16gb",
        "vhf-4c-16gb-sc1",
        "vhf-6c-24gb",
        "vhf-6c-24gb-sc1",
        "vhf-8c-32gb",
        "vhf-8c-32gb-sc1",
        "vhp-12c-24gb-amd",
        "vhp-12c-24gb-amd-sc1",
        "vhp-12c-24gb-intel",
        "vhp-12c-24gb-intel-sc1",
        "vhp-1c-1gb-amd",
        "vhp-1c-1gb-amd-sc1",
        "vhp-1c-1gb-intel",
        "vhp-1c-1gb-intel-sc1",
        "vhp-1c-2gb-amd",
        "vhp-1c-2gb-amd-sc1",
        "vhp-1c-2gb-intel",
        "vhp-1c-2gb-intel-sc1",
        "vhp-2c-2gb-amd",
        "vhp-2c-2gb-amd-sc1",
        "vhp-2c-2gb-intel",
        "vhp-2c-2gb-intel-sc1",
        "vhp-2c-4gb-amd",
        "vhp-2c-4gb-amd-sc1",
        "vhp-2c-4gb-intel",
        "vhp-2c-4gb-intel-sc1",
        "vhp-4c-12gb-amd",
        "vhp-4c-12gb-amd-sc1",
        "vhp-4c-12gb-intel",
        "vhp-4c-12gb-intel-sc1",
        "vhp-4c-8gb-amd",
        "vhp-4c-8gb-amd-sc1",
        "vhp-4c-8gb-intel",
        "vhp-4c-8gb-intel-sc1",
        "vhp-8c-16gb-amd",
        "vhp-8c-16gb-amd-sc1",
        "vhp-8c-16gb-intel",
        "vhp-8c-16gb-intel-sc1",
        "voc-c-16c-32gb-300s-amd",
        "voc-c-16c-32gb-300s-amd-sc1",
        "voc-c-16c-32gb-500s-amd",
        "voc-c-16c-32gb-500s-amd-sc1",
        "voc-c-1c-2gb-25s-amd",
        "voc-c-1c-2gb-25s-amd-sc1",
        "voc-c-2c-4gb-50s-amd",
        "voc-c-2c-4gb-50s-amd-sc1",
        "voc-c-2c-4gb-75s-amd",
        "voc-c-2c-4gb-75s-amd-sc1",
        "voc-c-32c-64gb-1000s-amd",
        "voc-c-32c-64gb-1000s-amd-sc1",
        "voc-c-32c-64gb-500s-amd",
        "voc-c-32c-64gb-500s-amd-sc1",
        "voc-c-4c-8gb-150s-amd",
        "voc-c-4c-8gb-150s-amd-sc1",
        "voc-c-4c-8gb-75s-amd",
        "voc-c-4c-8gb-75s-amd-sc1",
        "voc-c-8c-16gb-150s-amd",
        "voc-c-8c-16gb-150s-amd-sc1",
        "voc-c-8c-16gb-300s-amd",
        "voc-c-8c-16gb-300s-amd-sc1",
        "voc-g-16c-64gb-320s-amd",
        "voc-g-16c-64gb-320s-amd-sc1",
        "voc-g-1c-4gb-30s-amd",
        "voc-g-1c-4gb-30s-amd-sc1",
        "voc-g-24c-96gb-480s-amd",
        "voc-g-24c-96gb-480s-amd-sc1",
        "voc-g-2c-8gb-50s-amd",
        "voc-g-2c-8gb-50s-amd-sc1",
        "voc-g-32c-128gb-640s-amd",
        "voc-g-32c-128gb-640s-amd-sc1",
        "voc-g-40c-160gb-768s-amd",
        "voc-g-40c-160gb-768s-amd-sc1",
        "voc-g-4c-16gb-80s-amd",
        "voc-g-4c-16gb-80s-amd-sc1",
        "voc-g-64c-192gb-960s-amd",
        "voc-g-64c-192gb-960s-amd-sc1",
        "voc-g-8c-32gb-160s-amd",
        "voc-g-8c-32gb-160s-amd-sc1",
        "voc-g-96c-256gb-1280s-amd",
        "voc-g-96c-256gb-1280s-amd-sc1",
        "voc-m-16c-128gb-1600s-amd",
        "voc-m-16c-128gb-1600s-amd-sc1",
        "voc-m-16c-128gb-3200s-amd",
        "voc-m-16c-128gb-3200s-amd-sc1",
        "voc-m-16c-128gb-800s-amd",
        "voc-m-16c-128gb-800s-amd-sc1",
        "voc-m-1c-8gb-50s-amd",
        "voc-m-1c-8gb-50s-amd-sc1",
        "voc-m-24c-192gb-1200s-amd",
        "voc-m-24c-192gb-1200s-amd-sc1",
        "voc-m-24c-192gb-2400s-amd",
        "voc-m-24c-192gb-2400s-amd-sc1",
        "voc-m-24c-192gb-4800s-amd",
        "voc-m-24c-192gb-4800s-amd-sc1",
        "voc-m-2c-16gb-100s-amd",
        "voc-m-2c-16gb-100s-amd-sc1",
        "voc-m-2c-16gb-200s-amd",
        "voc-m-2c-16gb-200s-amd-sc1",
        "voc-m-2c-16gb-400s-amd",
        "voc-m-2c-16gb-400s-amd-sc1",
        "voc-m-32c-256gb-1600s-amd",
        "voc-m-32c-256gb-1600s-amd-sc1",
        "voc-m-32c-256gb-3200s-amd",
        "voc-m-32c-256gb-3200s-amd-sc1",
        "voc-m-4c-32gb-200s-amd",
        "voc-m-4c-32gb-200s-amd-sc1",
        "voc-m-4c-32gb-400s-amd",
        "voc-m-4c-32gb-400s-amd-sc1",
        "voc-m-4c-32gb-800s-amd",
        "voc-m-4c-32gb-800s-amd-sc1",
        "voc-m-8c-64gb-1600s-amd",
        "voc-m-8c-64gb-1600s-amd-sc1",
        "voc-m-8c-64gb-400s-amd",
        "voc-m-8c-64gb-400s-amd-sc1",
        "voc-m-8c-64gb-800s-amd",
        "voc-m-8c-64gb-800s-amd-sc1",
        "voc-s-16c-128gb-2560s-amd",
        "voc-s-16c-128gb-2560s-amd-sc1",
        "voc-s-16c-128gb-3840s-amd",
        "voc-s-16c-128gb-3840s-amd-sc1",
        "voc-s-1c-8gb-150s-amd",
        "voc-s-1c-8gb-150s-amd-sc1",
        "voc-s-24c-192gb-3840s-amd",
        "voc-s-24c-192gb-3840s-amd-sc1",
        "voc-s-24c-192gb-5760s-amd",
        "voc-s-24c-192gb-5760s-amd-sc1",
        "voc-s-2c-16gb-320s-amd",
        "voc-s-2c-16gb-320s-amd-sc1",
        "voc-s-2c-16gb-480s-amd",
        "voc-s-2c-16gb-480s-amd-sc1",
        "voc-s-32c-256gb-5760s-amd",
        "voc-s-32c-256gb-5760s-amd-sc1",
        "voc-s-4c-32gb-640s-amd",
        "voc-s-4c-32gb-640s-amd-sc1",
        "voc-s-4c-32gb-960s-amd",
        "voc-s-4c-32gb-960s-amd-sc1",
        "voc-s-8c-64gb-1280s-amd",
        "voc-s-8c-64gb-1280s-amd-sc1",
        "voc-s-8c-64gb-1920s-amd",
        "voc-s-8c-64gb-1920s-amd-sc1",
    ]:
        if offer.instance.name == instance:
            return True
    return False

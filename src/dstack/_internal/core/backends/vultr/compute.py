from typing import List, Optional

from dstack._internal.core.backends.base import Compute
from dstack._internal.core.backends.base.compute import (
    get_instance_name,
)
from dstack._internal.core.backends.vultr.api_client import VultrApiClient
from dstack._internal.core.backends.vultr.config import VultrConfig
from dstack._internal.core.models.instances import (
    InstanceConfiguration,
    InstanceOfferWithAvailability,
    SSHKey,
)
from dstack._internal.core.models.runs import Job, JobProvisioningData, Requirements, Run
from dstack._internal.core.models.volumes import Volume


class VultrCompute(Compute):
    def __init__(self, config: VultrConfig):
        self.config = config
        self.api_client = VultrApiClient(config.creds.api_key)

    def get_offers(
        self, requirements: Optional[Requirements] = None
    ) -> List[InstanceOfferWithAvailability]:
        pass

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
            job_docker_config=None,
            user=run.user,
        )
        return self.create_instance(instance_offer, instance_config)

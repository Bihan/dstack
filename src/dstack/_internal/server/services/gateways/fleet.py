from sqlalchemy.ext.asyncio import AsyncSession

from dstack._internal.core.backends.base.compute import get_gateway_container_commands
from dstack._internal.core.models.configurations import ServiceConfiguration
from dstack._internal.core.models.gateways import GatewayConfiguration
from dstack._internal.core.models.repos.virtual import DEFAULT_VIRTUAL_REPO_ID, VirtualRunRepoData
from dstack._internal.core.models.runs import RunSpec
from dstack._internal.server.models import GatewayModel
from dstack._internal.server.services.runs import submit_run
from dstack._internal.utils import common as common_utils


async def submit_fleet_gateway_run(
    session: AsyncSession,
    gateway_model: GatewayModel,
    configuration: GatewayConfiguration,
):
    """
    Submit a run that executes the gateway as a container job on the fleet.
    """
    project = gateway_model.project
    if gateway_model.created_by_user is None:
        raise ValueError(
            "Gateway has no creating user (created_by_user_id); cannot submit fleet run"
        )
    user = gateway_model.created_by_user
    user_ssh_public_key = common_utils.get_or_error(user.ssh_public_key)

    commands = get_gateway_container_commands(
        authorized_keys=[user_ssh_public_key],
        router=configuration.router,
    )
    fleets = common_utils.get_or_error(configuration.fleets)
    service_config = ServiceConfiguration(
        image="ubuntu:22.04",
        commands=commands,
        port=8000,
        fleets=fleets,
    )
    run_spec = RunSpec(
        run_name=f"gateway-{gateway_model.name}",
        repo_id=DEFAULT_VIRTUAL_REPO_ID,
        repo_data=VirtualRunRepoData(),
        configuration=service_config,
    )

    run = await submit_run(
        session=session,
        user=user,
        project=project,
        run_spec=run_spec,
    )
    gateway_model.run_id = run.id
    return run

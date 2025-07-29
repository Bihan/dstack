from typing import Annotated, List, Literal, Optional, Union

from pydantic import Field

from dstack._internal.core.models.common import CoreModel


# Hotaisle API key for authentication
class HotaisleAPIKeyCreds(CoreModel):
    type: Annotated[Literal["api_key"], Field(description="The type of credentials")] = "api_key"
    api_key: Annotated[str, Field(description="The Hotaisle API key")]


AnyHotaisleCreds = HotaisleAPIKeyCreds
HotaisleCreds = AnyHotaisleCreds


class HotaisleBackendConfig(CoreModel):
    """
    The backend config used in the API, server/config.yml, `HotaisleConfigurator`.
    It also serves as a base class for other backend config models.
    Should not include creds.
    """

    type: Annotated[
        Literal["hotaisle"],
        Field(description="The type of backend"),
    ] = "hotaisle"
    team_handle: Annotated[str, Field(description="The Hotaisle team handle")]
    regions: Annotated[
        Optional[List[str]],
        Field(description="The list of Hotaisle regions. Omit to use all regions"),
    ] = None
    # TODO: Add additional backend parameters if necessary


class HotaisleBackendConfigWithCreds(HotaisleBackendConfig):
    """
    Same as `HotaisleBackendConfig` but also includes creds.
    """

    creds: Annotated[AnyHotaisleCreds, Field(description="The credentials")]


AnyHotaisleBackendConfig = Union[HotaisleBackendConfig, HotaisleBackendConfigWithCreds]


class HotaisleBackendFileConfigWithCreds(HotaisleBackendConfig):
    """
    Same as `HotaisleBackendConfig` but also includes creds.
    Used for config file parsing in server/config.yml.
    """

    creds: Annotated[AnyHotaisleCreds, Field(description="The credentials")]


class HotaisleStoredConfig(HotaisleBackendConfig):
    """
    The backend config used for config parameters in the DB.
    Can extend `HotaisleBackendConfig` with additional parameters.
    """

    pass


class HotaisleConfig(HotaisleStoredConfig):
    """
    The backend config used by `HotaisleBackend` and `HotaisleCompute`.
    """

    creds: AnyHotaisleCreds

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import azure.core.exceptions
from azure.core.credentials import TokenCredential
from azure.mgmt import network as network_mgmt
from azure.mgmt import resource as resource_mgmt
from azure.mgmt import subscription as subscription_mgmt
from azure.mgmt.network.models import (
    AddressSpace,
    NetworkSecurityGroup,
    SecurityRule,
    SecurityRuleAccess,
    SecurityRuleDirection,
    SecurityRuleProtocol,
    Subnet,
    VirtualNetwork,
)
from azure.mgmt.resource.resources.models import ResourceGroup

from dstack._internal.core.backends.azure import AzureBackend, auth, compute, resources
from dstack._internal.core.backends.azure import utils as azure_utils
from dstack._internal.core.backends.azure.config import AzureConfig
from dstack._internal.core.errors import (
    BackendAuthError,
    BackendError,
    ServerClientError,
)
from dstack._internal.core.models.backends.azure import (
    AnyAzureConfigInfo,
    AzureClientCreds,
    AzureConfigInfo,
    AzureConfigInfoWithCreds,
    AzureConfigInfoWithCredsPartial,
    AzureConfigValues,
    AzureCreds,
    AzureDefaultCreds,
    AzureStoredConfig,
)
from dstack._internal.core.models.backends.base import (
    BackendType,
    ConfigElement,
    ConfigElementValue,
    ConfigMultiElement,
)
from dstack._internal.core.models.common import is_core_model_instance
from dstack._internal.server import settings
from dstack._internal.server.models import BackendModel, DecryptedString, ProjectModel
from dstack._internal.server.services.backends.configurators.base import (
    TAGS_MAX_NUM,
    Configurator,
    raise_invalid_credentials_error,
)

LOCATIONS = [
    ("(US) Central US", "centralus"),
    ("(US) East US, Virginia", "eastus"),
    ("(US) East US 2, Virginia", "eastus2"),
    ("(US) South Central US, Texas", "southcentralus"),
    ("(US) West US 2, Washington", "westus2"),
    ("(US) West US 3, Phoenix", "westus3"),
    ("(Canada) Canada Central, Toronto", "canadacentral"),
    ("(Europe) France Central, Paris", "francecentral"),
    ("(Europe) Germany West Central, Frankfurt", "germanywestcentral"),
    ("(Europe) North Europe, Ireland", "northeurope"),
    ("(Europe) Sweden Central, Gävle", "swedencentral"),
    ("(Europe) UK South, London", "uksouth"),
    ("(Europe) West Europe", "westeurope"),
    ("(Asia Pacific) Southeast Asia, Singapore", "southeastasia"),
    ("(Asia Pacific) East Asia", "eastasia"),
    ("(South America) Brazil South", "brazilsouth"),
]
LOCATION_VALUES = [loc[1] for loc in LOCATIONS]
DEFAULT_LOCATIONS = LOCATION_VALUES
MAIN_LOCATION = "eastus"


class AzureConfigurator(Configurator):
    TYPE: BackendType = BackendType.AZURE

    def get_default_configs(self) -> List[AzureConfigInfoWithCreds]:
        if not auth.default_creds_available():
            return []
        try:
            credential, _ = auth.authenticate(AzureDefaultCreds())
        except BackendAuthError:
            return []
        tenant_id_element = self._get_tenant_id_element(credential=credential)
        tenant_ids = [v.value for v in tenant_id_element.values]
        subscription_id_element = self._get_subscription_id_element(credential=credential)
        subscription_ids = [v.value for v in subscription_id_element.values]
        configs = []
        for tenant_id in tenant_ids:
            for subscription_id in subscription_ids:
                config = AzureConfigInfoWithCreds(
                    tenant_id=tenant_id,
                    subscription_id=subscription_id,
                    locations=DEFAULT_LOCATIONS,
                    creds=AzureDefaultCreds(),
                )
                configs.append(config)
        return configs

    def get_config_values(self, config: AzureConfigInfoWithCredsPartial) -> AzureConfigValues:
        config_values = AzureConfigValues()
        config_values.default_creds = (
            settings.DEFAULT_CREDS_ENABLED and auth.default_creds_available()
        )
        if config.creds is None:
            return config_values
        if (
            is_core_model_instance(config.creds, AzureDefaultCreds)
            and not settings.DEFAULT_CREDS_ENABLED
        ):
            raise_invalid_credentials_error(fields=[["creds"]])
        if is_core_model_instance(config.creds, AzureClientCreds):
            self._set_client_creds_tenant_id(config.creds, config.tenant_id)
        try:
            credential, creds_tenant_id = auth.authenticate(config.creds)
        except BackendAuthError:
            if is_core_model_instance(config.creds, AzureClientCreds):
                raise_invalid_credentials_error(
                    fields=[
                        ["creds", "tenant_id"],
                        ["creds", "client_id"],
                        ["creds", "client_secret"],
                    ]
                )
            else:
                raise_invalid_credentials_error(fields=[["creds"]])
        config_values.tenant_id = self._get_tenant_id_element(
            credential=credential,
            selected=config.tenant_id or creds_tenant_id,
        )
        if config_values.tenant_id.selected is None:
            return config_values
        config_values.subscription_id = self._get_subscription_id_element(
            credential=credential,
            selected=config.subscription_id,
        )
        if config_values.subscription_id.selected is None:
            return config_values
        config_values.locations = self._get_locations_element(
            selected=config.locations or DEFAULT_LOCATIONS
        )
        self._check_config(config=config, credential=credential)
        return config_values

    def create_backend(
        self, project: ProjectModel, config: AzureConfigInfoWithCreds
    ) -> BackendModel:
        if config.locations is None:
            config.locations = DEFAULT_LOCATIONS
        if is_core_model_instance(config.creds, AzureClientCreds):
            self._set_client_creds_tenant_id(config.creds, config.tenant_id)
        credential, _ = auth.authenticate(config.creds)
        if config.resource_group is None:
            config.resource_group = self._create_resource_group(
                credential=credential,
                subscription_id=config.subscription_id,
                location=MAIN_LOCATION,
                project_name=project.name,
            )
        self._create_network_resources(
            credential=credential,
            subscription_id=config.subscription_id,
            resource_group=config.resource_group,
            locations=config.locations,
            create_default_network=config.vpc_ids is None,
        )
        return BackendModel(
            project_id=project.id,
            type=self.TYPE.value,
            config=AzureStoredConfig(
                **AzureConfigInfo.__response__.parse_obj(config).dict(),
            ).json(),
            auth=DecryptedString(plaintext=AzureCreds.parse_obj(config.creds).__root__.json()),
        )

    def get_config_info(self, model: BackendModel, include_creds: bool) -> AnyAzureConfigInfo:
        config = self._get_backend_config(model)
        if include_creds:
            return AzureConfigInfoWithCreds.__response__.parse_obj(config)
        return AzureConfigInfo.__response__.parse_obj(config)

    def get_backend(self, model: BackendModel) -> AzureBackend:
        config = self._get_backend_config(model)
        return AzureBackend(config=config)

    def _get_backend_config(self, model: BackendModel) -> AzureConfig:
        return AzureConfig.__response__(
            **json.loads(model.config),
            creds=AzureCreds.parse_raw(model.auth.get_plaintext_or_error()).__root__,
        )

    def _set_client_creds_tenant_id(
        self,
        creds: AzureClientCreds,
        tenant_id: Optional[str],
    ):
        if creds.tenant_id is not None:
            return
        if tenant_id is None:
            raise_invalid_credentials_error(
                fields=[
                    ["creds", "tenant_id"],
                    ["tenant_id"],
                ]
            )
        creds.tenant_id = tenant_id

    def _get_tenant_id_element(
        self,
        credential: auth.AzureCredential,
        selected: Optional[str] = None,
    ) -> ConfigElement:
        subscription_client = subscription_mgmt.SubscriptionClient(credential=credential)
        element = ConfigElement(selected=selected)
        tenant_ids = []
        for tenant in subscription_client.tenants.list():
            tenant_ids.append(tenant.tenant_id)
            element.values.append(
                ConfigElementValue(value=tenant.tenant_id, label=tenant.tenant_id)
            )
        if selected is not None and selected not in tenant_ids:
            raise ServerClientError(
                "Invalid tenant_id",
                fields=[["tenant_id"]],
            )
        if len(tenant_ids) == 1:
            element.selected = tenant_ids[0]
        return element

    def _get_subscription_id_element(
        self,
        credential: auth.AzureCredential,
        selected: Optional[str] = None,
    ) -> ConfigElement:
        subscription_client = subscription_mgmt.SubscriptionClient(credential=credential)
        element = ConfigElement(selected=selected)
        subscription_ids = []
        for subscription in subscription_client.subscriptions.list():
            subscription_ids.append(subscription.subscription_id)
            element.values.append(
                ConfigElementValue(
                    value=subscription.subscription_id,
                    label=f"{subscription.display_name} ({subscription.subscription_id})",
                )
            )
        if selected is not None and selected not in subscription_ids:
            raise ServerClientError(
                "Invalid subscription_id",
                fields=[["subscription_id"]],
            )
        if len(subscription_ids) == 1:
            element.selected = subscription_ids[0]
        if len(subscription_ids) == 0:
            # Credentials without granted roles don't see any subscriptions
            raise ServerClientError(
                msg="No Azure subscriptions found for provided credentials. Ensure the account has enough permissions.",
            )
        return element

    def _get_locations_element(self, selected: List[str]) -> ConfigMultiElement:
        element = ConfigMultiElement()
        for loc in LOCATION_VALUES:
            element.values.append(ConfigElementValue(value=loc, label=loc))
        element.selected = selected
        return element

    def _create_resource_group(
        self,
        credential: auth.AzureCredential,
        subscription_id: str,
        location: str,
        project_name: str,
    ) -> str:
        resource_manager = ResourceManager(
            credential=credential,
            subscription_id=subscription_id,
        )
        return resource_manager.create_resource_group(
            name=_get_resource_group_name(project_name),
            location=location,
        )

    def _create_network_resources(
        self,
        credential: auth.AzureCredential,
        subscription_id: str,
        resource_group: str,
        locations: List[str],
        create_default_network: bool,
    ):
        def func(location: str):
            network_manager = NetworkManager(
                credential=credential, subscription_id=subscription_id
            )
            if create_default_network:
                network_manager.create_virtual_network(
                    resource_group=resource_group,
                    location=location,
                    name=azure_utils.get_default_network_name(resource_group, location),
                    subnet_name=azure_utils.get_default_subnet_name(resource_group, location),
                )
            network_manager.create_network_security_group(
                resource_group=resource_group,
                location=location,
                name=azure_utils.get_default_network_security_group_name(resource_group, location),
            )
            network_manager.create_gateway_network_security_group(
                resource_group=resource_group,
                location=location,
                name=azure_utils.get_gateway_network_security_group_name(resource_group, location),
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            for location in locations:
                executor.submit(func, location)

    def _check_config(
        self, config: AzureConfigInfoWithCredsPartial, credential: auth.AzureCredential
    ):
        self._check_tags_config(config)
        self._check_resource_group(config=config, credential=credential)
        self._check_vpc_config(config=config, credential=credential)

    def _check_tags_config(self, config: AzureConfigInfoWithCredsPartial):
        if not config.tags:
            return
        if len(config.tags) > TAGS_MAX_NUM:
            raise ServerClientError(
                f"Maximum number of tags exceeded. Up to {TAGS_MAX_NUM} tags is allowed."
            )
        try:
            resources.validate_tags(config.tags)
        except BackendError as e:
            raise ServerClientError(e.args[0])

    def _check_resource_group(
        self, config: AzureConfigInfoWithCredsPartial, credential: auth.AzureCredential
    ):
        if config.resource_group is None:
            return
        resource_manager = ResourceManager(
            credential=credential,
            subscription_id=config.subscription_id,
        )
        if not resource_manager.resource_group_exists(config.resource_group):
            raise ServerClientError(f"Resource group {config.resource_group} not found")

    def _check_vpc_config(
        self, config: AzureConfigInfoWithCredsPartial, credential: auth.AzureCredential
    ):
        if config.subscription_id is None:
            return None
        allocate_public_ip = config.public_ips if config.public_ips is not None else True
        if config.public_ips is False and config.vpc_ids is None:
            raise ServerClientError(msg="`vpc_ids` must be specified if `public_ips: false`.")
        locations = config.locations
        if locations is None:
            locations = DEFAULT_LOCATIONS
        if config.vpc_ids is not None:
            vpc_ids_locations = list(config.vpc_ids.keys())
            not_configured_locations = [loc for loc in locations if loc not in vpc_ids_locations]
            if len(not_configured_locations) > 0:
                if config.locations is None:
                    raise ServerClientError(
                        f"`vpc_ids` not configured for regions {not_configured_locations}. "
                        "Configure `vpc_ids` for all regions or specify `regions`."
                    )
                raise ServerClientError(
                    f"`vpc_ids` not configured for regions {not_configured_locations}. "
                    "Configure `vpc_ids` for all regions specified in `regions`."
                )
            network_client = network_mgmt.NetworkManagementClient(
                credential=credential,
                subscription_id=config.subscription_id,
            )
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = []
                for location in locations:
                    future = executor.submit(
                        compute.get_resource_group_network_subnet_or_error,
                        network_client=network_client,
                        resource_group=None,
                        vpc_ids=config.vpc_ids,
                        location=location,
                        allocate_public_ip=allocate_public_ip,
                    )
                    futures.append(future)
                for future in as_completed(futures):
                    try:
                        future.result()
                    except BackendError as e:
                        raise ServerClientError(e.args[0])


def _get_resource_group_name(project_name: str) -> str:
    return f"dstack-{project_name}"


class ResourceManager:
    def __init__(self, credential: TokenCredential, subscription_id: str):
        self.resource_client = resource_mgmt.ResourceManagementClient(
            credential=credential, subscription_id=subscription_id
        )

    def create_resource_group(
        self,
        name: str,
        location: str,
    ) -> str:
        resource_group: ResourceGroup = self.resource_client.resource_groups.create_or_update(
            resource_group_name=name,
            parameters=ResourceGroup(
                location=location,
            ),
        )
        return resource_group.name

    def resource_group_exists(
        self,
        name: str,
    ) -> bool:
        try:
            self.resource_client.resource_groups.get(
                resource_group_name=name,
            )
        except azure.core.exceptions.ResourceNotFoundError:
            return False
        return True


class NetworkManager:
    def __init__(self, credential: TokenCredential, subscription_id: str):
        self.network_client = network_mgmt.NetworkManagementClient(
            credential=credential, subscription_id=subscription_id
        )

    def create_virtual_network(
        self,
        resource_group: str,
        name: str,
        subnet_name: str,
        location: str,
    ) -> Tuple[str, str]:
        network: VirtualNetwork = self.network_client.virtual_networks.begin_create_or_update(
            resource_group_name=resource_group,
            virtual_network_name=name,
            parameters=VirtualNetwork(
                location=location,
                address_space=AddressSpace(address_prefixes=["10.0.0.0/16"]),
                subnets=[
                    Subnet(
                        name=subnet_name,
                        address_prefix="10.0.0.0/20",
                    )
                ],
            ),
        ).result()
        return network.name, subnet_name

    def create_network_security_group(
        self,
        resource_group: str,
        location: str,
        name: str,
    ):
        self.network_client.network_security_groups.begin_create_or_update(
            resource_group_name=resource_group,
            network_security_group_name=name,
            parameters=NetworkSecurityGroup(
                location=location,
                security_rules=[
                    SecurityRule(
                        name="runner_ssh",
                        protocol=SecurityRuleProtocol.TCP,
                        source_address_prefix="Internet",
                        source_port_range="*",
                        destination_address_prefix="*",
                        destination_port_range="22",
                        access=SecurityRuleAccess.ALLOW,
                        priority=100,
                        direction=SecurityRuleDirection.INBOUND,
                    ),
                ],
            ),
        ).result()

    def create_gateway_network_security_group(
        self,
        resource_group: str,
        location: str,
        name: str,
    ):
        self.network_client.network_security_groups.begin_create_or_update(
            resource_group_name=resource_group,
            network_security_group_name=name,
            parameters=NetworkSecurityGroup(
                location=location,
                security_rules=[
                    SecurityRule(
                        name="gateway_all",
                        protocol=SecurityRuleProtocol.TCP,
                        source_address_prefix="Internet",
                        source_port_range="*",
                        destination_address_prefix="*",
                        destination_port_ranges=["22", "80", "443"],
                        access=SecurityRuleAccess.ALLOW,
                        priority=101,
                        direction=SecurityRuleDirection.INBOUND,
                    )
                ],
            ),
        ).result()

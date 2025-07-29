import json

from dstack._internal.core.backends.base.configurator import (
    BackendRecord,
    Configurator,
    raise_invalid_credentials_error,
)
from dstack._internal.core.backends.hotaisle.backend import HotaisleBackend
from dstack._internal.core.backends.hotaisle.models import (
    AnyHotaisleBackendConfig,
    AnyHotaisleCreds,
    HotaisleBackendConfig,
    HotaisleBackendConfigWithCreds,
    HotaisleConfig,
    HotaisleCreds,
    HotaisleStoredConfig,
)
from dstack._internal.core.models.backends.base import (
    BackendType,
)


class HotaisleConfigurator(Configurator):
    TYPE = BackendType.HOTAISLE
    BACKEND_CLASS = HotaisleBackend

    def validate_config(self, config: HotaisleBackendConfigWithCreds, default_creds_enabled: bool):
        self._validate_creds(config.creds, config.team_handle)
        # TODO: If possible, validate config.regions and any other config parameters

    def create_backend(
        self, project_name: str, config: HotaisleBackendConfigWithCreds
    ) -> BackendRecord:
        return BackendRecord(
            config=HotaisleStoredConfig(
                **HotaisleBackendConfig.__response__.parse_obj(config).dict()
            ).json(),
            auth=HotaisleCreds.parse_obj(config.creds).json(),
        )

    def get_backend_config(
        self, record: BackendRecord, include_creds: bool
    ) -> AnyHotaisleBackendConfig:
        config = self._get_config(record)
        if include_creds:
            return HotaisleBackendConfigWithCreds.__response__.parse_obj(config)
        return HotaisleBackendConfig.__response__.parse_obj(config)

    def get_backend(self, record: BackendRecord) -> HotaisleBackend:
        config = self._get_config(record)
        return HotaisleBackend(config=config)

    def _get_config(self, record: BackendRecord) -> HotaisleConfig:
        return HotaisleConfig.__response__(
            **json.loads(record.config),
            creds=HotaisleCreds.parse_raw(record.auth),
        )

    def _validate_creds(self, creds: AnyHotaisleCreds, team_handle: str):
        """Validate Hotaisle API credentials by testing API access and team membership"""
        import requests

        try:
            # Test the API key and team handle by validating user and team access
            url = "https://admin.hotaisle.app/api/user/"
            headers = {
                "accept": "application/json",
                "Authorization": creds.api_key,
            }

            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 401:
                raise_invalid_credentials_error(fields=[["creds", "api_key"]])
            elif response.status_code != 200:
                # Other error codes might indicate service issues, not necessarily invalid creds
                return

            # Validate team handle
            user_data = response.json()
            teams = user_data.get("teams", [])

            if not teams:
                raise_invalid_credentials_error(
                    fields=[["team_handle"]], message="No Hotaisle teams found for this user"
                )

            # Verify the user provided team exists
            available_teams = [team["handle"] for team in teams]
            if team_handle not in available_teams:
                raise_invalid_credentials_error(
                    fields=[["team_handle"]],
                    message=f"Hotaisle Team '{team_handle}' not found. Available teams: {', '.join(available_teams)}",
                )

        except requests.RequestException:
            # Network errors shouldn't fail validation
            return
        except Exception as e:
            if "raise_invalid_credentials_error" in str(e):
                raise  # Re-raise our own validation errors
            # Other exceptions shouldn't fail validation
            return

#!/usr/bin/env python3

"""
Test script to verify Hotaisle backend configuration structure
matches the YAML format shown in .dstack/server/config.yml
"""

import os
import sys

import yaml

# Add src to path so we can import dstack modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dstack._internal.core.backends.hotaisle.models import (
    HotaisleAPIKeyCreds,
    HotaisleBackendConfigWithCreds,
)


def test_config_structure():
    """Test that our model matches the expected YAML structure"""

    # This matches the structure from the user's config.yml
    expected_yaml_structure = {
        "type": "hotaisle",
        "team_handle": "dstackai",
        "creds": {"type": "api_key", "api_key": "api_key"},
    }

    # Create our model instance
    config = HotaisleBackendConfigWithCreds(
        team_handle="dstackai", creds=HotaisleAPIKeyCreds(api_key="api_key")
    )

    # Convert to dict for comparison
    config_dict = config.dict()

    print("Expected YAML structure:")
    print(yaml.dump(expected_yaml_structure, default_flow_style=False))
    print()
    print("Actual model structure:")
    print(yaml.dump(config_dict, default_flow_style=False))
    print()

    # Check key fields match
    assert config_dict["type"] == expected_yaml_structure["type"], "Type mismatch"
    assert config_dict["team_handle"] == expected_yaml_structure["team_handle"], (
        "Team handle mismatch"
    )
    assert config_dict["creds"]["type"] == expected_yaml_structure["creds"]["type"], (
        "Creds type mismatch"
    )
    assert config_dict["creds"]["api_key"] == expected_yaml_structure["creds"]["api_key"], (
        "API key mismatch"
    )

    print("✅ Configuration structure matches expected YAML format!")

    # Test that regions is optional and defaults to None
    assert config_dict.get("regions") is None, "Regions should default to None"
    print("✅ Optional regions field works correctly!")


if __name__ == "__main__":
    test_config_structure()

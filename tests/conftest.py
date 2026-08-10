# Shared fixtures for the aeroopt test suite.
# Paths are built with os.path so the tests also run on Windows.

import os

import pytest

# Project root; the template configuration lives in aeroopt/template_settings.json
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_SETTINGS_PATH = os.path.join(_ROOT, "aeroopt", "template_settings.json")


@pytest.fixture(scope="session")
def template_settings_path() -> str:
    """Path to the packaged template settings file."""
    assert os.path.exists(TEMPLATE_SETTINGS_PATH), (
        f"template_settings.json not found at {TEMPLATE_SETTINGS_PATH}")
    return TEMPLATE_SETTINGS_PATH

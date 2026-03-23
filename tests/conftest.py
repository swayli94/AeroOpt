# Tests for AeroOpt.core, using template_settings.json
# Windows-compatible paths via os.path

import os
import pytest

# 项目根目录，模板配置位于 AeroOpt/template_settings.json
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_SETTINGS_PATH = os.path.join(_ROOT, "AeroOpt", "template_settings.json")


# core 中 Individual 使用 SettingsData.source_dict 与 Individual.source_dict，
# 而 SettingsData 仅有 data_source_dict，在此做别名以兼容测试
@pytest.fixture(scope="session", autouse=True)
def _patch_source_dict():
    from AeroOpt.core.settings import SettingsData
    from AeroOpt.core.individual import Individual
    if not hasattr(SettingsData, "source_dict"):
        SettingsData.source_dict = SettingsData.data_source_dict
    if not hasattr(Individual, "source_dict"):
        Individual.source_dict = SettingsData.data_source_dict
    yield

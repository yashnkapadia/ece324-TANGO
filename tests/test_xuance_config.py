from xuance.common import get_configs

from ece324_tango.asce.trainers.xuance_backend import XuanceBackend


def test_xuance_custom_config_exists_and_targets_sumo():
    config_path = XuanceBackend._CONFIG_PATH
    assert config_path.exists()
    cfg = get_configs(str(config_path))
    assert cfg["env_name"] == "sumo_custom"
    assert cfg["env_id"] == "grid4x4"

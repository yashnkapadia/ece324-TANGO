from ece324_tango.asce.trainers.xuance_env import register_xuance_sumo_env


def test_register_xuance_sumo_env():
    name = register_xuance_sumo_env("sumo_custom_test")
    assert name == "sumo_custom_test"

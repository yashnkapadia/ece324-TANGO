from types import SimpleNamespace

from ece324_tango.asce.baselines import MaxPressureController


class DummyLane:
    def __init__(self, halting):
        self._halting = halting

    def getLastStepHaltingNumber(self, lane):
        return self._halting.get(lane, 0)


class DummyTrafficlight:
    def __init__(self, controlled_links):
        self._controlled_links = controlled_links

    def getControlledLinks(self, ts_id):
        return self._controlled_links[ts_id]


def test_max_pressure_returns_valid_action_index():
    controlled_links = {
        "A0": [
            (("in0", "out0", "via0"),),
            (("in1", "out1", "via1"),),
        ]
    }
    halting = {"in0": 10, "out0": 2, "in1": 1, "out1": 7}

    ts = SimpleNamespace(
        green_phases=[SimpleNamespace(state="Gr"), SimpleNamespace(state="rG")]
    )
    env = SimpleNamespace(
        traffic_signals={"A0": ts},
        sumo=SimpleNamespace(
            trafficlight=DummyTrafficlight(controlled_links),
            lane=DummyLane(halting),
        ),
    )

    ctrl = MaxPressureController(action_size_by_agent={"A0": 2})
    actions = ctrl.actions(observations={"A0": [0.0]}, env=env)

    assert "A0" in actions
    assert actions["A0"] in {0, 1}
    # phase 0 should win: (10-2)=8 vs phase 1: (1-7)=-6
    assert actions["A0"] == 0

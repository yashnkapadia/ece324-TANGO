from ece324_tango.asce.kpi import occupancy_for_vehicle_type


def test_occupancy_for_vehicle_type():
    assert occupancy_for_vehicle_type("passenger") == 1.3
    assert occupancy_for_vehicle_type("BUS") == 30.0
    assert occupancy_for_vehicle_type("ttc_bus_type") == 30.0

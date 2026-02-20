from ece324_tango.asce.schema import ASCE_DATASET_COLUMNS


def test_asce_schema_columns_stable():
    expected = {
        "intersection_id",
        "time_step",
        "queue_ns",
        "queue_ew",
        "arrivals_ns",
        "arrivals_ew",
        "avg_speed_ns",
        "avg_speed_ew",
        "current_phase",
        "time_of_day",
        "action_phase",
        "action_green_dur",
        "delay",
        "queue_total",
        "throughput",
        "scenario_id",
    }
    assert set(ASCE_DATASET_COLUMNS) == expected


def test_schema_column_order_starts_with_id_and_time():
    assert ASCE_DATASET_COLUMNS[0] == "intersection_id"
    assert ASCE_DATASET_COLUMNS[1] == "time_step"

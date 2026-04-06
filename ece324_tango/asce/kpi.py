from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from ece324_tango.asce.runtime import jain_index
from ece324_tango.error_reporting import report_exception


def occupancy_for_vehicle_type(type_id: str) -> float:
    tid = str(type_id).lower()
    if any(token in tid for token in ("bus", "tram", "streetcar", "ttc")):
        return 30.0
    return 1.3


@dataclass
class EpisodeKPI:
    time_loss_s: float
    person_time_loss_s: float
    avg_trip_time_s: float
    arrived_vehicles: int
    vehicle_delay_jain: float


class KPITracker:
    """Track proposal-aligned KPI values from SUMO TraCI state."""

    def __init__(self):
        self._last_time_loss_by_vehicle: Dict[str, float] = {}
        self._depart_time_by_vehicle: Dict[str, float] = {}
        self._arrived_time_losses: List[float] = []
        self.total_time_loss_s = 0.0
        self.total_person_time_loss_s = 0.0
        self.total_trip_time_s = 0.0
        self.arrived_vehicles = 0

    def update(self, env) -> None:
        sim_time = float(env.sumo.simulation.getTime())
        active_ids = list(env.sumo.vehicle.getIDList())

        for vid in active_ids:
            try:
                current_tl = float(env.sumo.vehicle.getTimeLoss(vid))
            except Exception as exc:
                report_exception(
                    context="kpi.time_loss_lookup_failed",
                    exc=exc,
                    details={"vehicle_id": vid},
                    once_key=f"kpi_time_loss:{vid}",
                )
                continue
            prev_tl = self._last_time_loss_by_vehicle.get(vid, current_tl)
            delta_tl = max(0.0, current_tl - prev_tl)
            self._last_time_loss_by_vehicle[vid] = current_tl

            try:
                vtype = env.sumo.vehicle.getTypeID(vid)
            except Exception as exc:
                report_exception(
                    context="kpi.vehicle_type_lookup_failed",
                    exc=exc,
                    details={"vehicle_id": vid},
                    once_key=f"kpi_vehicle_type:{vid}",
                )
                vtype = ""
            occ = occupancy_for_vehicle_type(vtype)
            self.total_time_loss_s += delta_tl
            self.total_person_time_loss_s += delta_tl * occ

            if vid not in self._depart_time_by_vehicle:
                try:
                    self._depart_time_by_vehicle[vid] = float(
                        env.sumo.vehicle.getDeparture(vid)
                    )
                except Exception as exc:
                    report_exception(
                        context="kpi.departure_lookup_failed",
                        exc=exc,
                        details={"vehicle_id": vid},
                        once_key=f"kpi_departure:{vid}",
                    )
                    self._depart_time_by_vehicle[vid] = sim_time

        for vid in list(env.sumo.simulation.getArrivedIDList()):
            depart_time = self._depart_time_by_vehicle.pop(vid, None)
            final_tl = self._last_time_loss_by_vehicle.pop(vid, None)
            if final_tl is not None:
                self._arrived_time_losses.append(final_tl)
            if depart_time is None:
                continue
            trip_t = max(0.0, sim_time - float(depart_time))
            self.total_trip_time_s += trip_t
            self.arrived_vehicles += 1

    def summary(self) -> EpisodeKPI:
        avg_trip = (
            self.total_trip_time_s / self.arrived_vehicles
            if self.arrived_vehicles > 0
            else 0.0
        )
        return EpisodeKPI(
            time_loss_s=float(self.total_time_loss_s),
            person_time_loss_s=float(self.total_person_time_loss_s),
            avg_trip_time_s=float(avg_trip),
            arrived_vehicles=int(self.arrived_vehicles),
            vehicle_delay_jain=jain_index(self._arrived_time_losses),
        )

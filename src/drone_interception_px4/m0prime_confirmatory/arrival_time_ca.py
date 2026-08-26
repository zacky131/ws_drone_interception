"""Arrival-time counterpart of the frozen source-time delayed CA estimator."""

from __future__ import annotations

from dapcs_mpc.delay_aware_imm import TelemetryPacket
from paper_completion.delay_aware_ca import DelayAwareCA


class ArrivalTimeCA(DelayAwareCA):
    """Apply an accepted delayed packet to the current state at arrival time.

    All prediction, covariance, measurement, initialization, history, delay
    acceptance and forecast behavior is inherited from ``DelayAwareCA``.  The
    only changed operation is the temporal placement of the measurement update.
    """

    def process_packet(self, packet: TelemetryPacket) -> None:
        arrival = float(packet.arrival_timestamp_s)
        source = float(packet.source_timestamp_s)
        if arrival < self.last_arrival_timestamp_s - 1e-12:
            raise ValueError("arrival timestamps cannot move backward")
        if source > arrival + 1e-12:
            raise ValueError("source timestamp cannot be after arrival")

        self.predict_to(arrival)
        self.last_arrival_timestamp_s = arrival
        self.last_repropagation_steps = 0
        if packet.drop or not packet.valid:
            self.dropped_packets += 1
            return
        if arrival - source > self.config.maximum_accepted_delay_s + 1e-12:
            self.rejected_packets += 1
            return
        if source < self.history[0].timestamp_s - 1e-12:
            self.rejected_packets += 1
            return

        # Intentional and sole algorithmic difference from DelayAwareCA:
        # update the already-predicted current state rather than rolling back.
        self._measurement_update(packet.measurement)
        if self.history and abs(self.history[-1].timestamp_s - arrival) <= 1e-12:
            self.history[-1] = self._snapshot()
        else:
            self.history.append(self._snapshot())
        self._trim_history()
        self.accepted_updates += 1
        # This inherited field denotes the state timestamp at which the update
        # was applied.  It is source time for M1 and arrival time for M0-prime.
        self.last_update_source_timestamp_s = arrival

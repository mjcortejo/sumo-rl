"""Max-pressure agent class."""

import numpy as np

from sumo_rl.agents.clm_agent import phase_movements


class MaxPressureAgent:
    """Picks the phase whose movements have the largest (incoming - outgoing) vehicle count."""

    def __init__(self, ts):
        """Initialize max-pressure agent for a TrafficSignal."""
        self.ts = ts
        self.movements = phase_movements(ts)
        self.action = None

    def act(self):
        """Choose the phase with maximum pressure."""
        lane = self.ts.sumo.lane
        pressures = [
            sum(lane.getLastStepVehicleNumber(m.in_lane) - lane.getLastStepVehicleNumber(m.out_lane) for m in movements)
            for movements in self.movements
        ]
        self.action = int(np.argmax(pressures))
        return self.action

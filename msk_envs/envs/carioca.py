import torch

from msk_envs.geometry import FWD_IDX, SIDE_IDX, build_axis
from .lanes import LanesEnv


class CariocaEnv(LanesEnv):
    """
    Carioca / grapevine drill environment.
    The agent must move laterally and alternate crossing one foot
    in front of and behind the other, with a deliberate reset
    (feet side-by-side) between each crossing.
    Phase cycle (0 -> 1 -> 2 -> 3 -> 0 -> ...):
      0  NEUTRAL_1     – feet side-by-side
      1  CROSS_FRONT   – trailing foot crosses in front
      2  NEUTRAL_2     – feet side-by-side again
      3  CROSS_BEHIND  – trailing foot crosses behind
    """
    target_dir = build_axis(SIDE_IDX, 1.0)
    angle_tolerance = 45.0  # need more swivel room for carioca
    lane_width = 1.0  # Bit wider for room

    TRAVEL_IDX = FWD_IDX
    SAGITTAL_IDX = SIDE_IDX

    NEUTRAL_1 = 0
    CROSS_FRONT = 1
    NEUTRAL_2 = 2
    CROSS_BEHIND = 3
    NUM_PHASES = 4

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Current phase index per world (0-3)
        self.phase = torch.zeros(self.num_worlds, dtype=torch.long, device=self.device)

        # Steps spent in the current phase without completing it
        self.cross_timer = torch.zeros(self.num_worlds, dtype=torch.long, device=self.device)

        # How far the trailing foot must cross past the lead foot (m)
        self.cross_threshold = 0.05
        # How much behind the trailing foot must be to be considered in neutral
        self.forward_threshold = 0.1
        # Steps allowed per phase before termination
        self.max_phase_steps = 20

        self.left_foot_collider_ids = [
            self.collider_id_lookup[collider_key] for collider_key in self.collider_id_lookup.keys()
            if "left_foot" in collider_key
        ]
        self.right_foot_collider_ids = [
            self.collider_id_lookup[collider_key] for collider_key in self.collider_id_lookup.keys()
            if "right_foot" in collider_key
        ]

    def _upon_reset_pre_sim(self, reset_mask: torch.Tensor) -> None:
        self.phase[reset_mask] = 0
        self.cross_timer[reset_mask] = 0

    def _get_obs(self) -> torch.Tensor:
        lanes_obs = super()._get_obs()

        # One-hot encode the current phase: (num_worlds, NUM_PHASES)
        phase_one_hot = torch.zeros(
            self.num_worlds, self.NUM_PHASES,
            dtype=torch.float32, device=self.device
        )
        phase_one_hot.scatter_(1, self.phase.unsqueeze(1), 1.0)

        obs = torch.cat([
            lanes_obs,
            phase_one_hot,
            self.cross_timer.float().view(self.num_worlds, 1),
        ], dim=1)
        return obs.detach().clone()

    def _get_touchdown(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns whether left/right foot is touching the ground
        """
        collider_forces = torch.abs(self.collider_forces)
        grf_nonzero = torch.linalg.norm(self.grf, dim=1) > 0
        left_touchdown = (collider_forces[:, self.left_foot_collider_ids] > 0).any(dim=1)
        right_touchdown = (collider_forces[:, self.right_foot_collider_ids] > 0).any(dim=1)
        return left_touchdown & grf_nonzero, right_touchdown & grf_nonzero

    def _get_cross_deltas(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        fwd_delta:  right_toe − left_toe along FWD_IDX  (>0 = right foot in front)
        side_delta: right_toe − left_toe along SIDE_IDX (>0 = right foot laterally ahead)
        """
        left_toe_pos = self.body_positions[:, self.toes_ids[0], :]
        right_toe_pos = self.body_positions[:, self.toes_ids[1], :]
        travel_delta = right_toe_pos[:, self.TRAVEL_IDX] - left_toe_pos[:, self.TRAVEL_IDX]
        sagittal_delta = right_toe_pos[:, self.SAGITTAL_IDX] - left_toe_pos[:, self.SAGITTAL_IDX]
        return travel_delta, sagittal_delta

    def _update_carioca_state(self) -> torch.Tensor:
        """
        Returns a wrong_cross mask: True for any world that achieved
        the opposite crossing condition to the one currently expected.
        """
        travel_delta, sagittal_delta = self._get_cross_deltas()
        left_touchdown, right_touchdown = self._get_touchdown()

        # Right foot overtook the left foot and touchdown
        crossed_travel = (travel_delta > self.forward_threshold) & right_touchdown

        # Right foot is behind the left foot (neutral)
        neutral = (travel_delta < -self.forward_threshold)

        # Sagittal position relationship
        right_foot_laterally_ahead = sagittal_delta > self.cross_threshold
        right_foot_laterally_behind = sagittal_delta < -self.cross_threshold

        # Whether the right foot crossed in front AND is laterally ahead/behind
        right_foot_crossed_front = crossed_travel & right_foot_laterally_ahead
        right_foot_crossed_behind = crossed_travel & right_foot_laterally_behind

        # Completion condition per phase
        phase_done = torch.stack([
            right_foot_crossed_front,  # NEUTRAL -> RIGHT FOOT CROSSED AHEAD
            neutral,  # RIGHT FOOT CROSSED AHEAD -> NEUTRAL
            right_foot_crossed_behind,  # NEUTRAL -> RIGHT FOOT CROSSED BEHIND
            neutral,  # RIGHT FOOT CROSSED BEHIND -> NEUTRAL
        ], dim=1)  # (num_worlds, 4)

        completed = phase_done.gather(1, self.phase.unsqueeze(1)).squeeze(1)

        # Wrong cross: achieved the opposite crossing while in a crossing phase
        expected_cross_front = (self.phase == self.NEUTRAL_1)
        expected_cross_behind = (self.phase == self.NEUTRAL_2)
        wrong_cross = (expected_cross_front & right_foot_crossed_behind) | (
                expected_cross_behind & right_foot_crossed_front)

        # Transition to next phase, start the timer again
        self.phase = torch.where(completed, (self.phase + 1) % self.NUM_PHASES, self.phase, )
        self.cross_timer = torch.where(completed, torch.zeros_like(self.cross_timer), self.cross_timer + 1, )
        return wrong_cross

    def _get_terminated(self) -> torch.Tensor:
        # NOTE: this also advances the phase state machine, which the next observation depends on
        terminated_lanes = self._get_lane_terminated()
        stuck_phase = (self.cross_timer >= self.max_phase_steps)
        wrong_cross = self._update_carioca_state()

        return (terminated_lanes | stuck_phase | wrong_cross).float().detach()

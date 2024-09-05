import numpy as np
from scipy.stats import vonmises
from typing import List

class PeriodicClock:

    """
    Class for keeping track of clock values.
    """

    def __init__(self, cycle_time: float, phase_add: float, swing_ratios: List[float], period_shifts: List[float]):
        assert len(swing_ratios) == 2, \
               f"PeriodicClock got swing_ratios input of length {len(swing_ratios)}, but should " \
               f"be of length 2."
        assert len(period_shifts) == 2, \
               f"PeriodicClock got period_shifts input of length {len(period_shifts)}, but should " \
               f"be of length 2."
        self._cycle_time = cycle_time
        self._phase_add = phase_add
        # Assume that swing ratios and period shifts are in order of [left, right]
        self._swing_ratios = swing_ratios
        self._period_shifts = period_shifts
        self._von_mises_buf = None
        self._phase = np.random.uniform(0, self._cycle_time)

    def increment(self, phase_add: float=None):
        """Increment phase by phase_add

        Args:
            phase_add (float, optional): Use externally defined phase_add if given. Defaults to None.
        """
        if phase_add:
            self._phase += phase_add
        else:
            self._phase += self._phase_add
        if self._phase > self._cycle_time:
            self._phase -= self._cycle_time

    def input_clock(self):
        """Clock represents absolute phase change on unit cycle.
        """
        clock = [np.sin(2 * np.pi * (self._phase / self._cycle_time)),
                 np.cos(2 * np.pi * (self._phase / self._cycle_time))]
        return clock

    def input_sine_only_clock(self):
        """Clock only in two sines functions, representing two seperate legs. This clock empirically
        need LSTM to learn.
        """
        clock = [np.sin(2 * np.pi * ((self._phase/self._cycle_time)+s)) for s in self._period_shifts]
        return clock

    def input_full_clock(self):
        """Clock in two sine/cosine pairs, one for each leg.
        """
        t = self._phase / self._cycle_time
        clock = [np.sin(2 * np.pi * ((t) + self._period_shifts[0])),
                 np.cos(2 * np.pi * ((t) + self._period_shifts[0])),
                 np.sin(2 * np.pi * ((t) + self._period_shifts[1])),
                 np.cos(2 * np.pi * ((t) + self._period_shifts[1]))]
        return clock

    def linear_clock(self, percent_transition: float = 0.2):
        """
        Implements piecewise linear clock function used for cyclic foot reward components. Uses the
        current swing ratio and period shift values to calculate the left and right foot swing clock
        values (0 meaning foot should be in stance, 1 meaning foot should be in swing, with smooth
        transitions between them). The input percent_transition determines what portion of swing
        time should be used to transition to and from swing.

        So for example, if cycle time is 1s, and swing ratio is 0.4, then the foot should be in
        swing for 0.4s. But in reality, this linear clock will return 0 from 0s to 0.6s, then
        linearly increase from 0 to 1 from 0.6s to 0.68s, then return 1 from 0.68s to 0.92s, the
        linearly decrease from 1 to 0 from 0.92s to 1s.

        Note that swing clock values are returned in the order (left, right). The corresponding
        stance values can be obtained from just 1 - swing_value.

        Arguments:
        percent_transition (float): What percentage of swing time to use to linearly transition
                                    between stance and swing. By default is 0.2 (20%).
        """
        y_clock = []
        x_clock = []
        phases = []
        for i in range(2):
            actual_phase = self._phase + self._period_shifts[i] * self._cycle_time
            swing_time = self._cycle_time * self._swing_ratios[i]
            stance_time = self._cycle_time * (1 - self._swing_ratios[i])
            trans_time = swing_time * percent_transition
            swing_time -= trans_time
            y_clock.append([0, 0, 1, 1, 0])
            x_clock.append([0, stance_time, stance_time + trans_time / 2,
                    stance_time + trans_time / 2 + swing_time, self._cycle_time])
            if actual_phase > self._cycle_time:
                phases.append(actual_phase - self._cycle_time)
            else:
                phases.append(actual_phase)
        return np.interp(phases[0], x_clock[0], y_clock[0]), np.interp(phases[1], x_clock[1], y_clock[1])

    def von_mises(self, std: float = 0.1):
        # Von Mises clock function, but with hard coded coeff values of [0, 1]. This is chosen to
        # line up with the linear clock convention of left stance starting at t = 0. Because of this
        # coefficient choice, we actually need to "flip" the swing/stance ratio. If coeff was [1, 0]
        # then swing ratio would be as defined, i.e. if self._swing_ratio[0] = 0.4 then left foot
        # would have 40% swing, but left SWING would start at t = 0. Since we use coeff of [0, 1]
        # instead, swing ratio because stance ratio and vice versa.
        assert std != 0, \
            f"von_mises received std of zero. std must be non-zero."
        kappa = 1 / (std ** 2)

        out = []
        for i in range(2):
            x = (self._phase / self._cycle_time + self._period_shifts[i]) * 2 * np.pi
            # Use `1 - self._swing_ratios[i]` here to flip swing and stance ratios.
            mid = (1 - self._swing_ratios[i]) * 2 * np.pi
            end = 2 * np.pi
            p2 = vonmises.cdf(x, kappa=kappa, loc=mid, scale=1)
            p3 = vonmises.cdf(x, kappa=kappa, loc=end, scale=1)
            out.append(p2 - p3)

        return out[0], out[1]

    def von_mises_full(self, coeff: List[float] = [0.0, 1.0], std: float  = 0.2):
        # This is an alternate version of the von mises clock function. This will do the full
        # computation and allow for different coefficient values which will change the shape of the
        # clock. I think for most cases the coeffs we would use are [0, 1], which simplifies the
        # computation. In this case, use the regular `von_mises` function which should be faster.
        kappa = 1 / (std ** 2)

        out = []
        for i in range(2):
            x = (self._phase / self._cycle_time + self._period_shifts[i]) * 2 * np.pi
            start = 0
            mid = (1 - self._swing_ratios[i]) * 2 * np.pi
            end = 2 * np.pi
            p1 = vonmises.cdf(x, kappa=kappa, loc=start, scale=1)
            p2 = vonmises.cdf(x, kappa=kappa, loc=mid, scale=1)
            p3 = vonmises.cdf(x, kappa=kappa, loc=end, scale=1)
            out.append(coeff[0] * (p1 - p2) + coeff[1] * (p2 - p3))

        return out[0], out[1]

    def precompute_von_mises(self, num_points: int = 200, std = 0.1):
        # Von mises clock can take a while to compute since it has to compute the cdf multiple times
        # so during training for an env it can be useful to precompute the clock values just once
        # beforehand. Note that this function has to be called again if _swing_ratios or
        # _period_shifts change, the values have to be recomputed
        xs = np.linspace(0, self._cycle_time, num_points)
        self._von_mises_buf = np.zeros((num_points, 2))
        orig_phase = self._phase
        for i in range(len(xs)):
            self._phase = xs[i]
            self._von_mises_buf[i, :] = self.von_mises()
        self._phase = orig_phase

    def get_von_mises_values(self):
        assert self._von_mises_buf is not None, \
            f"Von Mises clock buffer is None, can not get value. Call `precompute_von_mises` first."
        xs = np.linspace(0, self._cycle_time, self._von_mises_buf.shape[0])
        return np.interp(self._phase, xs, self._von_mises_buf[:, 0]), np.interp(self._phase, xs, self._von_mises_buf[:, 1])

    # Getters/Setters for class variables
    def get_phase(self):
        return self._phase

    def set_phase(self, phase: float):
        self._phase = phase

    def get_cycle_time(self):
        return self._cycle_time

    def set_cycle_time(self, cycle_time: float):
        self._cycle_time = cycle_time
        # If Von Mises buffer already exists, recompute it to update values
        if self._von_mises_buf is not None:
            self.precompute_von_mises()

    def get_phase_add(self):
        return self._phase_add

    def set_phase_add(self, phase_add: float):
        self._phase_add = phase_add
        # If Von Mises buffer already exists, recompute it to update values
        if self._von_mises_buf is not None:
            self.precompute_von_mises()

    def get_swing_ratios(self):
        return self._swing_ratios

    def set_swing_ratios(self, swing_ratios: List[float]):
        """
        Sets the swing ratios. Note that the input should be a list of length 2, the first element
        being the left swing ratio and the second element being the right swing ratio. Only swing
        ratio is needed since stance ratio is just 1 - swing_ratio

        Arguments:
        swing_ratios (List[float]): List of swing ratios to set to, in format [left, right]
        """
        assert len(swing_ratios) == 2, \
            f"set_swing_ratios got list of length {len(swing_ratios)}, but should be length 2."
        self._swing_ratios = swing_ratios
        # If Von Mises buffer already exists, recompute it to update values
        if self._von_mises_buf is not None:
            self.precompute_von_mises()

    def get_period_shifts(self):
        return self._period_shifts

    def set_period_shifts(self, period_shifts: List[float]):
        """
        Sets the period shifts. Note that the input should be a list of length 2, the first element
        being the left period shift and the second element being the right period shift.

        Arguments:
        period_shifts (List[float]): List of period shifts to set to, in format [left, right]
        """
        assert len(period_shifts) == 2, \
            f"set_swing_ratios got list of length {len(period_shifts)}, but should be length 2."
        self._period_shifts = period_shifts
        # If Von Mises buffer already exists, recompute it to update values
        if self._von_mises_buf is not None:
            self.precompute_von_mises()

    def is_stance(self, threshold=0.05):
        return [i < threshold for i in self.linear_clock(percent_transition=0.01)]

    def is_swing(self):
        return [not i for i in self.is_stance()]

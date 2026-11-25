"""
ocp.py
======
The open-circuit potentials the model uses: a PyBaMM parameter set's half-cell curves (the only thing
the identification takes from the literature besides the temperature laws), optionally with a bounded,
smooth correction on one electrode fitted to the cell's own low-rate OCV test (srcs/window.py).

    ocp = OCPSet('OKane2022')
    ocp.U_n(x), ocp.U_p(y)                      vectorised, clipped to the tabulated range
    ocp2 = ocp.corrected('n', knots, dU)        U_n(x) + piecewise-linear dU on the knots [V]

The correction is a piecewise-linear function on fixed knots, zero outside them, so it can only bend
the curve locally; its amplitude bound and smoothness weight are part of the study config.

A correction {"electrode": "p", "table": {"y": [...], "U": [...]}} instead REPLACES that electrode's
curve by the table (the cathode by subtraction from the cell's own OCV, scripts/step1_window.py); a
"U_n_table": {"x": [...], "U": [...]} next to it replaces the anode curve (the one step 1 subtracted).
"""
from __future__ import annotations

import numpy as np

from srcs import physics


class OCPSet:
    def __init__(self, pm: str, correction: dict | None = None):
        self.pm = pm
        self._U_n, self._U_p = physics.ocps(pm)
        self.dU_n, self.dU_p = physics.entropic(pm)
        self.correction = correction
        if correction and 'U_n_table' in correction:      # the anode curve step 1 used (an analytic literature fit)
            xn, Un = np.asarray(correction['U_n_table']['x'], float), np.asarray(correction['U_n_table']['U'], float)
            self._U_n = lambda v: np.interp(np.clip(v, xn[0], xn[-1]), xn, Un)
        if correction and 'table' in correction:
            s, U = np.asarray(correction['table']['y'], float), np.asarray(correction['table']['U'], float)
            f = lambda v: np.interp(np.clip(v, s[0], s[-1]), s, U)
            if correction['electrode'] == 'p':
                self._U_p = f
            else:
                self._U_n = f
        elif correction:
            self._knots = np.asarray(correction['knots'], float)
            self._dU = np.asarray(correction['dU'], float)

    def _corr(self, s, electrode):
        if not self.correction or 'table' in self.correction or self.correction['electrode'] != electrode:
            return 0.0
        return np.interp(s, self._knots, self._dU, left=0.0, right=0.0)

    def U_n(self, x):
        return self._U_n(x) + self._corr(np.asarray(x, float), 'n')

    def U_p(self, y):
        return self._U_p(y) + self._corr(np.asarray(y, float), 'p')

    def corrected(self, electrode: str, knots, dU) -> 'OCPSet':
        return OCPSet(self.pm, dict(electrode=electrode, knots=list(map(float, knots)), dU=list(map(float, dU))))

    def describe(self) -> dict:
        return dict(pm=self.pm, correction=self.correction)


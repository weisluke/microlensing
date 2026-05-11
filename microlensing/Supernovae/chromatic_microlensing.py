import numpy as np
from scipy.interpolate import RegularGridInterpolator
from sncosmo import PropagationEffect


class ChromaticMicrolensing(PropagationEffect):
    _param_names = []
    _parameters = np.array([])

    def __init__(self, mu, phases, wavelengths=None, **kwargs):
        """
        :param mu: If 2D, array of magnification as a function of
                   phase (axis=0) and wavelength (axis=1)
                   If 1D, array of magnification as a function of phase
        :param phases: Array of phases in days
        :param wavelengths: Array of wavelengths in Angstroms. If passed, mu
                            must be a 2D array
        """
        mu = np.asanyarray(mu)

        phases = np.asanyarray(phases)
        if phases.ndim != 1:
            raise ValueError("phases is not a 1D array")
        self._minphase = np.min(phases)
        self._maxphase = np.max(phases)

        # no wavelength provided, achromatic microlensing that is the same for
        # all wavelengths
        if wavelengths is None:
            wavelengths = np.array([0, np.inf])
            # mu should be a 1D array if no wavelengths were provided...
            if mu.ndim != 1:
                raise ValueError("mu is not a 1D array")
            # ...but we do then need to make it 2D for interpolation
            else:
                mu = np.repeat(mu[:, None], 2, axis=1)
        else:
            wavelengths = np.asanyarray(wavelengths)
            if wavelengths.ndim != 1:
                raise ValueError("wavelengths is not a 1D array")
            if mu.ndim != 2:
                raise ValueError("mu is not a 2D array")
        self._minwave = np.min(wavelengths)
        self._maxwave = np.max(wavelengths)

        if (mu.shape[0] != phases.shape[0]
                or mu.shape[1] != wavelengths.shape[0]):
            raise ValueError("Dimensions of mu do not match provided phases and wavelengths")

        self.mu = RegularGridInterpolator((phases, wavelengths), mu)

    def propagate(self, wave, flux, phase):
        """
        Propagate the magnification onto the model's flux output.
        """
        wave = np.atleast_1d(wave)
        phase = np.atleast_1d(phase)

        mu = self.mu((phase[:, None], wave[None, :]))
        return flux * mu

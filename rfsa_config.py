from RSA_API import *
from RSA_API_funclib_usercopy import search_connect, config_spectrum, acquire_spectrum 
import os
from numpy.typing import NDArray
import numpy as np

class RFSA:
    """
    A class to interface with the Tektronix Real-Time Spectrum Analyzer (RSA)
    using the RSA API.

    Provides methods for configuring, acquiring, and managing spectrum analyzer
    operations including device connection and disconnection.

    Parameters
    ----------
    dll_path : str, optional, default="C:\\Tektronix\\RSA_API\\lib\\x64"
        Directory path to the RSA API DLL library.

    Notes
    -----
    This class loads the Tektronix RSA_API.dll dynamically and wraps commonly
    used RSA spectrum acquisition operations for automation workflows.

    Examples
    --------
    >>> rfsa = RFSA()
    >>> rfsa.configure_rfsa(
    ...     center_freq=2.45e9,
    ...     ref_level=0,
    ...     span=40e6,
    ...     rbw=100e3,
    ...     trace_pts=801
    ... )
    >>> freq = rfsa.create_obs_freq_arr()
    >>> spectrum = rfsa.capture_spectrum()
    >>> rfsa.disconnect()
    """

    def __init__(self, dll_path: str = "C:\\Tektronix\\RSA_API\\lib\\x64"):
        """
        Initialize the RFSA class by adding the RSA API DLL directory and
        loading the shared library.

        Parameters
        ----------
        dll_path : str, optional, default="C:\\Tektronix\\RSA_API\\lib\\x64"
            Directory path containing the RSA_API.dll.

        Returns
        -------
        None

        Notes
        -----
        Uses ctypes to load the RSA_API.dll dynamically at runtime. The DLL path
        must be valid for proper operation.
        """
        os.add_dll_directory(dll_path)
        self.rfsa = cdll.LoadLibrary("RSA_API.dll")

    def configure_rfsa(self, **specParams) -> None:
        """
        Configure the Real-Time Spectrum Analyzer (RSA) with given spectrum
        acquisition parameters.

        Parameters
        ----------
        **specParams : dict
            Dictionary containing the following keys:
                - center_freq : float
                    Center frequency in Hz.
                - ref_level : float
                    Reference level in dBm.
                - span : float
                    Frequency span in Hz.
                - rbw : float
                    Resolution bandwidth in Hz.
                - trace_pts : int
                    Number of trace points to acquire.

        Returns
        -------
        None

        Notes
        -----
        Establishes connection with the RSA device and configures spectrum
        acquisition parameters. The resulting configuration object is stored
        in `self.specSet`.

        Examples
        --------
        >>> rfsa.configure_rfsa(
        ...     center_freq=2.45e9,
        ...     ref_level=-10,
        ...     span=20e6,
        ...     rbw=100e3,
        ...     trace_pts=801
        ... )
        """
        search_connect()

        center_freq = specParams['center_freq']
        ref_level = specParams['ref_level']
        span = specParams['span']
        rbw = specParams['rbw']
        trace_pts = specParams['trace_pts']

        self.specSet = config_spectrum(cf=center_freq, refLevel=ref_level,
                                span=span, rbw=rbw, tracepts=trace_pts)

    def create_obs_freq_arr(self) -> NDArray[np.float64]:
        """
        Generate an array of observed frequencies corresponding to the spectrum trace.

        Parameters
        ----------
        None

        Returns
        -------
        obs_freq : NDArray[np.float64]
            Array of frequency points (in Hz) for the acquired spectrum.

        Notes
        -----
        The frequency array is derived from the actual start frequency, frequency
        step size, and trace length stored in the configured spectrum parameters.

        Examples
        --------
        >>> freqs = rfsa.create_obs_freq_arr()
        >>> print(freqs[:5])
        [2.44e9, 2.44005e9, 2.4401e9, 2.44015e9, 2.4402e9]
        """
        obs_freq = np.arange(self.specSet.actualStartFreq, self.specSet.actualStartFreq
                     + self.specSet.actualFreqStepSize * self.specSet.traceLength,
                     self.specSet.actualFreqStepSize)

        return obs_freq

    def capture_spectrum(self)-> NDArray[np.float64]:
        """
        Capture and return a single spectrum trace from the configured RSA.

        Parameters
        ----------
        None

        Returns
        -------
        spectrum : NDArray[np.float64]
            Measured power spectrum in dBm across the configured frequency range.

        Notes
        -----
        Requires prior configuration using `configure_rfsa()`. The function
        triggers spectrum acquisition and retrieves the measured trace.

        Examples
        --------
        >>> spectrum = rfsa.capture_spectrum()
        >>> print(spectrum.shape)
        (801,)
        """
        spectrum  = acquire_spectrum(self.specSet)
        return spectrum

    def disconnect(self) -> None:
        """
        Disconnect from the RSA device and release hardware resources.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        Stops active data acquisition, disconnects from the RSA device,
        and releases DLL ownership safely.

        Examples
        --------
        >>> rfsa.disconnect()
        """
        if self.rfsa:
            print('Disconnecting RFSA...')
            self.rfsa.DEVICE_Stop()
            self.rfsa.DEVICE_Disconnect()
            self.rfsa = None


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Units
    dbm = 1

    khz = 1e03
    megahz = 1e06
    ghz = 1e09
    
    f_center = 2200 * megahz
    rfsa = RFSA()
    spec_params = {
        'center_freq': f_center,
        'ref_level': -30 * dbm,
        'span': 400 * megahz,
        'rbw': 4 * khz,
        'trace_pts': 4001
    }

    rfsa.configure_rfsa(**spec_params)
    f_obs = rfsa.create_obs_freq_arr() / megahz
    spdata = rfsa.capture_spectrum()

    plt.figure("Test")
    plt.plot(f_obs, spdata)
    plt.axvline(f_center / megahz, color="k", alpha=0.5, linestyle="--")
    plt.axvline(2150, color="k", alpha=0.5, linestyle=":")
    plt.show()
from __future__ import annotations
import numpy as np
import time
import serial
import sys
import pandas as pd
import h5py
import clr  # pythonnet

clr.AddReference('mcl_RUDAT_NET45')  # Reference the DLL

from mcl_RUDAT_NET45 import USB_RUDAT
from rfgen_config import RFGEN
from rfsa_config import RFSA

from pathlib import Path
from itertools import product
from datetime import date, datetime
from typing import Tuple, Any
from numpy.typing import NDArray
from tqdm import tqdm


class SpectrumTestwProgAttn:
    """
    Class to capture spectra while varying input power and frequency. 
    It automates instrument connection, waveform generation, data acquisition, 
    and structured data storage in HDF5 format.

    Parameters
    ----------
    pwr_in : numpy.ndarray
        Input rf generator power levels in dBm.
    f_in : numpy.ndarray
        Input frequencies in Hz.
    spec_params : dict
        Dictionary containing RF spectrum analyzer configuration parameters.

    Notes
    -----
    - Provides a unified interface for conducting automated spectrum acquisition experiments.
    - Instrument communication must be initialized using `connect()` before acquisition.
    - Data and metadata are stored in compressed HDF5 format for reproducibility and efficient access.

    Example
    -------
    >>> test = SpectrumTest(pwr_in=np.linspace(-10, 10, 5),
    ...                     f_in=np.linspace(1e9, 1.2e9, 3),
    ...                     spec_params={'center_freq': 1e9,
    ...                                  'ref_level': 0,
    ...                                  'span': 100e6,
    ...                                  'rbw': 10e3,
    ...                                  'trace_pts': 801})
    >>> test.repeat_collect_and_save_spectra(
    ...     savepath="dataset/",
    ...     savefilename="run_batch",
    ...     num_iters=5,
    ...     delay=2
    ... )
    >>> test.disconnect()
    """

    def __init__(
            self,
            pwr_in: NDArray[np.float64],
            f_in: NDArray[np.float64],
            g_2: NDArray[np.float64],
            spec_params: dict
    ) -> None:
        """
        Initialize the spectrum recording test configuration.

        Parameters
        ----------
        pwr_in : numpy.ndarray
            Array of input oscillator power values in dBm.
        f_in : numpy.ndarray
            Array of input frequency values in Hz.
        g_2 : numpy.ndarray
            Array of attenuation values in dB.
        spec_params : dict
            Configuration parameters for the spectrum analyzer:
                - center_freq : float
                    Center frequency in Hz.
                - ref_level : float
                    Reference level in dBm.
                - span : float
                    Frequency span in Hz.
                - rbw : float
                    Resolution bandwidth in Hz.
                - trace_pts : int
                    Number of trace points per spectrum.

        Returns
        -------
        None

        Notes
        -----
        - Instruments are not connected until `connect()` is explicitly called.
        - The attributes `rfsa` and `rfgen` are initialized as `None` to avoid AttributeError.
        - If `pwr_in` exceeds 17 dBm or `f_in` exceeds 13.6 GHz, the program terminates.

        Example
        -------
        >>> test = SpectrumTest(pwr_in=np.array([0, 5, 10]),
        ...                     f_in=np.linspace(1e9, 1.2e9, 5),
        ...                     g_2=np.array([30, 25, 20]),
        ...                     spec_params={'center_freq': 1e9, 'ref_level': 0,
        ...                                  'span': 100e6, 'rbw': 10e3, 'trace_pts': 801})
        """
        self.pwr_in = pwr_in if pwr_in.max() < 17 else sys.exit("Maximum power limit is crossed.")
        self.f_in = f_in if f_in.max() < 13.6e09 else sys.exit("Maximum frequency limit is crossed.")
        self.g_2 = g_2 if g_2.max() < 30 else sys.exit("Maximum attenuation limit is crossed")
        self.spec_params = spec_params

        # prevent AttributeError in disconnect()
        self.rfsa = None
        self.rfgen = None
        self.attn = None

    def connect(self) -> None:
        """
        Connect to the RFSA and RF generator and apply initial configurations.

        Returns
        -------
        None

        Notes
        -----
        - Configures the RF Spectrum Analyzer (RFSA) with parameters in `spec_params`.
        - Initializes the RF Generator (RFGEN) via serial COM port.
        - Must be called before any data collection methods.

        Example
        -------
        >>> test.connect()
        """
        self.rfsa = RFSA()
        self.rfsa.configure_rfsa(**self.spec_params)

        self.rfgen = RFGEN(port="COM3", baudrate=9600)
        self.rfgen.connect()

        self.attn = USB_RUDAT()  # Create an instance of the USB control class
        self.attn.Connect()
        Responses = self.attn.Send_SCPI(":SN?", "")  # Read serial number
        print('Serial number: ' + str(Responses[
                                          1]))  # Python interprets the response as a tuple [function return (0 or 1), command parameter, response parameter]
        Responses = self.attn.Send_SCPI(":MN?", "")  # Read model name
        print('Model name: ' + str(Responses[1]))

    def disconnect(self) -> None:
        """
        Safely disconnect all connected instruments.

        Returns
        -------
        None

        Notes
        -----
        - Ensures that both RFSA and RFGEN sessions are closed cleanly.
        - Checks if each instrument handle exists before disconnection.
        - Should be called in a `finally` block for reliability.

        Example
        -------
        >>> test.disconnect()
        """
        if self.rfsa:
            self.rfsa.disconnect()
        if self.rfgen:
            self.rfgen.disconnect()
        if self.attn:
            self.attn.Disconnect()

    def collect_spectra(self) -> Tuple[float, NDArray[np.float64]]:
        """
        Collect spectra by iterating over input power and frequency arrays.

        Returns
        -------
        duration : float
            Total acquisition time in seconds.
        data : numpy.ndarray
            3D array of recorded spectra with shape 
            (len(g_2), len(pwr_in), len(f_in), trace_pts).

        Notes
        -----
        - Sets the attenuation value in dB.
        - Calls the RF generator to set each power and frequency pair.
        - Uses the RFSA to capture spectra for each configuration.
        - Timing is measured to assess acquisition performance.

        Example
        -------
        >>> duration, data = test.collect_spectra()
        >>> print(f"Collected {data.shape} spectra in {duration:.2f} s")
        """
        # RF lock time = 4 ms for RF generator
        lock_time = 10e-03  # 10 ms

        data = []
        start = time.time()

        for g_2 in self.g_2:
            Status = self.attn.Send_SCPI(f":SETATT={g_2}", "")  # Set attenuation
            Responses = self.attn.Send_SCPI(":ATT?", "")  # Read attenuation
            print(f"Set attenuation value: {Responses[1]} dB")
            outer = []
            for pwr_in in self.pwr_in:
                self.rfgen.set_pwr(pwr=pwr_in)
                inner = []
                for f_in in self.f_in:
                    self.rfgen.set_freq(freq=f_in)
                    time.sleep(lock_time)
                    inner.append(self.rfsa.capture_spectrum())
                outer.append(inner)   # ← wrap row inside a list for the middle dimension
            data.append(outer)

        end = time.time()
        data = np.array(data, dtype=np.float64)  # shape: (len(g_2), len(pwr_in), len(f_in), trace_pts)
        return end - start, data

    def _write_metadata_to_h5(self, f: h5py.File) -> None:
        """
        Write instrument configuration and experiment metadata to an open HDF5 file.

        Parameters
        ----------
        f : h5py.File
            Open HDF5 file handle to which metadata is written.

        Returns
        -------
        None

        Notes
        -----
        - Global attributes include experiment description, creator, and date.
        - Instrument configurations are stored in separate groups:
          `prog_attn_settings`, `rfsa_settings`, and `rfgen_settings`.
        - Uses gzip compression for storage efficiency.

        Example
        -------
        >>> with h5py.File("data/run1.h5", "a") as f:
        ...     test._write_metadata_to_h5(f)
        """
        # --- Global attributes ---
        f.attrs["creator"] = "Anirban Mukhopadhyay"
        f.attrs["description"] = (
            "Recording spectra at different input RF power and frequency values."
        )
        f.attrs["date"] = str(date.today())

        # --- RFSA settings ---
        grp = f.create_group("rfsa_settings")
        self.f_obs = self.rfsa.create_obs_freq_arr()
        dset = grp.create_dataset("observation_frequency",
                                  data=np.asarray(self.f_obs),
                                  shape=len(self.f_obs),
                                  dtype=np.float64,
                                  compression="gzip",
                                  compression_opts=9,
                                  shuffle=True)
        dset.attrs["unit"] = "Hz"

        for key, val in self.spec_params.items():
            if key == "ref_level":
                unit = " dBm"
                grp.attrs[key] = str(val) + unit
            elif key == "trace_pts":
                grp.attrs[key] = str(val)
            else:
                unit = " Hz"
                grp.attrs[key] = str(val) + unit

        # --- RFGEN settings ---
        grp = f.create_group("rfgen_settings")
        dset = grp.create_dataset("input_power",
                                  data=np.asarray(self.pwr_in),
                                  shape=len(self.pwr_in),
                                  dtype=np.float64,
                                  compression="gzip",
                                  compression_opts=9,
                                  shuffle=True)
        dset.attrs["unit"] = "dBm"

        dset = grp.create_dataset("input_frequency",
                                  data=np.asarray(self.f_in),
                                  shape=len(self.f_in),
                                  dtype=np.float64,
                                  compression="gzip",
                                  compression_opts=9,
                                  shuffle=True)
        dset.attrs["unit"] = "Hz"

        # --- Prog Attn settings ---
        grp = f.create_group("prog_attn_settings")
        dset = grp.create_dataset("set_attenuation",
                                  data=np.asarray(self.g_2),
                                  shape=len(self.g_2),
                                  dtype=np.float64,
                                  compression="gzip",
                                  compression_opts=9,
                                  shuffle=True)
        dset.attrs["unit"] = "dB"

    def repeat_collect_and_save_spectra(
            self,
            savepath: str | Path,
            savefilename: str | None,
            num_iters: int = 10,
            delay: int = 60
    ) -> None:
        """
        Perform repeated spectrum acquisitions and save all results into one HDF5 file.

        Parameters
        ----------
        savepath : str or Path
            Directory path where the HDF5 file will be stored.
        savefilename : str or None
            Output filename (".h5" extension is appended automatically).
        num_iters : int, optional
            Number of repeated acquisitions (default: 10).
        delay : int, optional
            Delay between acquisitions in seconds (default: 60).

        Returns
        -------
        None

        Notes
        -----
        - Instruments are connected once for the entire experiment.
        - First acquisition determines the data array shape for preallocation.
        - Datasets are compressed using gzip for efficiency.
        - Each iteration is timestamped for traceability.
        - Automatically handles existing filenames by appending timestamps.

        Example
        -------
        >>> test.repeat_collect_and_save_spectra(
        ...     savepath="dataset/",
        ...     savefilename="run_batch",
        ...     num_iters=5,
        ...     delay=30
        ... )
        """
        try:
            # connect instruments once for the whole run
            self.connect()

            # --- First measurement to discover array shape ---
            duration, first_data = self.collect_spectra()
            arr_shape = first_data.shape  # shape: (len(g_2), len(pwr_in), len(f_in), trace_pts)

            savepath = Path(savepath)
            savepath.mkdir(parents=True, exist_ok=True)

            if not savefilename:
                savefilename = savepath.name
            if not savefilename.lower().endswith(".h5"):
                savefilename += ".h5"
            output_h5_file = savepath / savefilename

            # --- Always prepend timestamp to output filename ---
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
            output_h5_file = output_h5_file.with_name(
                f"{timestamp}{output_h5_file.name}"
            )
            print(f"💾 Saving output as: {output_h5_file.name}")

            with h5py.File(output_h5_file, "a") as f:
                self._write_metadata_to_h5(f)

                # --- Preallocate main dataset ---
                grp = f.create_group("spectra")

                dset1 = grp.create_dataset(
                    "output_spectra",
                    shape=(num_iters,) + arr_shape,
                    dtype=np.float64,
                    compression="gzip",
                    compression_opts=9,
                    shuffle=True,
                )
                dset1.attrs["unit"] = "dBm"

                # --- Preallocate metadata datasets ---
                durations = f.create_dataset("durations", shape=(num_iters,), dtype="float32")
                timestamps = f.create_dataset("timestamps", shape=(num_iters,), dtype="S32")

                # --- Store first iteration ---
                dset1[0] = first_data
                durations[0] = duration
                timestamps[0] = datetime.now().strftime("%Y-%m-%d %H:%M:%S").encode("utf-8")
                f.flush()

                # --- Loop over remaining iterations ---
                for rep in tqdm(range(1, num_iters)):
                    time.sleep(delay)
                    duration, data = self.collect_spectra()
                    dset1[rep] = data
                    durations[rep] = duration
                    timestamps[rep] = datetime.now().strftime("%Y-%m-%d %H:%M:%S").encode("utf-8")
                    f.flush()

        except (Exception, KeyboardInterrupt) as e:
            print(f"[ERROR] Measurement stopped: {e}")

        finally:
            try:
                self.disconnect()
                print("[INFO] Instruments disconnected safely.")
            except Exception as e:
                print(f"[WARNING] Disconnect failed: {e}")


if __name__ == "__main__":

    # Units
    dbm = 1

    khz = 1e03
    megahz = 1e06
    ghz = 1e09

    secs = 1

    # Input power variation
    input_pwr = np.array([-30]) * dbm

    # Input frequency variation
    input_freq = np.linspace(2000, 2150 + 200, 11) * megahz

    g2 = np.array([25, 20, 15, 10, 5, 0, 5, 10, 15, 20, 25])

    spec_params = {
        'center_freq': np.mean(input_freq),
        'ref_level': -10 * dbm,
        'span': input_freq[-1] - input_freq[0],
        'rbw': 100 * khz,
        'trace_pts': 4001
    }

    if 1:
        spectrumtest = SpectrumTestwProgAttn(pwr_in=input_pwr,
                                             f_in=input_freq,
                                             spec_params=spec_params,
                                             g_2=g2,
                                             )

        spectrumtest.repeat_collect_and_save_spectra(
            savepath="C:/Users/DELL/Documents/spin_wave_expts/sept_25/spectrum_test_yig_s18",
            savefilename=f"spectrum_test_yig_s18_w_ampl_prog_attn_port1_to_2",
            num_iters=50,
            delay=2 * secs)


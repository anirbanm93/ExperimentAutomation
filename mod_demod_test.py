from __future__ import annotations
import numpy as np
import time
import serial
import sys
import pandas as pd
import h5py

from dso_config import DSO
from rfgen_config import RFGEN
from awg_config import AWG
from pulse_lib import gauss_pulse

from pathlib import Path
from itertools import product
from datetime import date, datetime
from typing import Tuple, Any
from numpy.typing import NDArray
from tqdm import tqdm


class ModDemodTest:
    """
    Class to perform modulation/demodulation testing using an Arbitrary Waveform Generator (AWG),
    a Digital Storage Oscilloscope (DSO), and an RF signal generator. It automates instrument
    connection, waveform generation, data acquisition, and structured data storage in HDF5 format.

    Notes
    -----
    - This class provides a unified interface for performing RF modulation-demodulation experiments.
    - Instrument communication must be established before data acquisition using `connect()`.
    - Results are automatically saved with metadata for reproducibility.

    Example
    -------
    >>> test = ModDemodTest(
    ...     pwr_lo=-5,
    ...     fc=1e9,
    ...     vpp=np.array([0.5, 1.0]),
    ...     sig=np.sin(np.linspace(0, 2*np.pi, 1000)),
    ...     time_periods=np.array([1e-6, 2e-6]),
    ...     dead_cyc=10,
    ...     dead_volt=0.0,
    ...     time_div=5e-9,
    ...     num_points=None,
    ...     avg_count=16,
    ...     volt_divs=[0.1, 0.2],
    ...     pos=[0, 0],
    ...     on_chans=[1, 2]
    ... )
    >>> test.repeat_collect_and_save_waveform(
        ...     savepath="dataset/",
        ...     savefilename="run_batch",
        ...     num_iters=5,
        ...     delay=30
        ... )
    >>> test.disconnect()
    """

    def __init__(
            self,
            pwr_lo: float,
            fc: float,
            vpp: NDArray[np.float64],
            sig: NDArray[np.float64],
            time_periods: NDArray[np.float64],
            dead_cyc: int,
            dead_volt: float,
            time_div: float,
            num_points: int | None,
            avg_count: int,
            volt_divs: list[float],
            pos: list[float],
            on_chans: list[int]
    ) -> None:
        """
        Initialize the modulation-demodulation test configuration.

        Parameters
        ----------
        pwr_lo : float
            Local oscillator (LO) power in dBm.
        fc : float
            LO carrier frequency in Hz.
        vpp : numpy.ndarray
            Array of peak-to-peak voltage amplitudes (V) to sweep.
        sig : numpy.ndarray
            Baseband modulation signal waveform.
        time_periods : numpy.ndarray
            Array of time periods (s) corresponding to pulse repetition.
        dead_cyc : int
            Number of dead cycles inserted between active waveform cycles.
        dead_volt : float
            Dead voltage level (V) for inactive pulse regions.
        time_div : float
            DSO time division setting (s/div).
        num_points : int or None
            Number of sampling points for DSO acquisition (None = auto).
        avg_count : int
            Number of waveform averages to perform on DSO.
        volt_divs : list of float
            Vertical scale per channel in volts/div.
        pos : list of float
            Vertical position offset for each channel.
        on_chans : list of int
            List of DSO channels to enable during acquisition.

        Notes
        -----
        - Attributes are initialized but instruments are not connected until `connect()` is called.
        - AWG, DSO, and RF generator objects are set to `None` initially to avoid AttributeError.
        """
        self.pwr_lo = pwr_lo if 7 <= pwr_lo <= 10 else sys.exit("LO power should be between 7 and 10 dBm.")
        self.fc = fc if fc < 13.6e9 else sys.exit("Maximum LO frequency limit is crossed.")
        self.vpp = vpp if vpp < 2 else sys.exit("Maximum peak to peak voltage limit is crossed.")
        self.sig = sig
        self.time_periods = time_periods
        self.dead_cyc = dead_cyc
        self.dead_volt = dead_volt
        self.time_div = time_div
        self.num_points = num_points
        self.avg_count = avg_count
        self.volt_divs = volt_divs
        self.pos = pos
        self.on_chans = on_chans

        # prevent AttributeError in disconnect()
        self.awg = None
        self.dso = None
        self.rfgen = None

    def connect(self) -> None:
        """
        Connect to the AWG, DSO, and RF generator and apply initial configurations.

        Notes
        -----
        - The AWG is initialized via a unique serial number.
        - The DSO channel states, timebase, and scaling parameters are configured automatically.
        - The RF generator is initialized via COM port and programmed with LO power and frequency.

        Example
        -------
        >>> test.connect()
        """
        self.awg = AWG(serial=65542)
        self.awg.connect()

        self.dso = DSO()
        self.dso.set_channel_state(on_chans=self.on_chans)
        self.dso.configure_channel(
            chans=self.on_chans,
            time_div=self.time_div,
            volt_divs=self.volt_divs,
            pos=self.pos,
            num_points=self.num_points,
            avg_count=self.avg_count,
            trig_level=0.5 * self.vpp.min(),
        )

        self.rfgen = RFGEN(port="COM3", baudrate=9600)
        self.rfgen.connect()
        self.rfgen.set_pwr_freq(self.pwr_lo, self.fc)
        print(f"LO power: {self.pwr_lo:0.2f} dBm")
        print(f"LO frequency: {self.fc * 1e-06:0.4f} MHz")

    def disconnect(self) -> None:
        """
        Disconnect all connected instruments safely.

        Notes
        -----
        - Ensures that AWG, DSO, and RF generator sessions are properly terminated.
        - Checks if each instrument handle exists before attempting disconnection.

        Example
        -------
        >>> test.disconnect()
        """
        if self.awg:
            self.awg.disconnect()
        if self.dso:
            self.dso.disconnect()
        if self.rfgen:
            self.rfgen.disconnect()

    def collect_waveform(self) -> Tuple[float, NDArray[np.float64]]:
        """
        Generate modulated waveforms, acquire oscilloscope data,
        and return both acquisition duration and waveform array.

        Returns
        -------
        duration : float
            Total acquisition time in seconds.
        data : numpy.ndarray
            Recorded waveform array of shape
            (len(vpp), len(time_periods), len(on_chans), num_points).

        Notes
        -----
        - The AWG is programmed for each voltage and time-period combination.
        - DSO trigger level is set dynamically to 40% of the current Vpp.
        - Returns data for all enabled DSO channels.

        Example
        -------
        >>> duration, data = test.collect_waveform()
        """
        data = []
        start = time.time()

        # Step 1: Configure the waveform once
        self.awg.setup_waveform_template(
            sample_rate="Auto",
            lut_data=list(self.sig),
            strict=False,
            dead_cycles=self.dead_cyc,
            dead_voltage=self.dead_volt
        )

        for vpp in self.vpp:
            row = []
            for tp in self.time_periods:
                freq = 1.0 / tp
                self.awg.gen_pulsed_wave(channels=[1, 2], frequency=freq, amplitude=vpp)
                row.append(self.dso.get_waveform(on_chans=self.on_chans))
            data.append(row)

        end = time.time()
        data = np.array(data, dtype=np.float32)  # shape: (len(vpp), len(time_periods), len(on_chans), num_points)
        return end - start, data

    def _write_metadata_to_h5(self, f: h5py.File, num_time_samples: int) -> None:
        """
        Write instrument configuration and experiment metadata to an open HDF5 file.

        Parameters
        ----------
        f : h5py.File
            Open HDF5 file object to write metadata into.
        num_time_samples : int
            Number of time-domain samples in each waveform.

        Notes
        -----
        - Global attributes include experiment description, creator, and date.
        - Instrument settings are stored in separate HDF5 groups: `awg_settings`, `dso_settings`, and `rfgen_settings`.
        - Datasets are compressed using gzip for efficient storage.

        Example
        -------
        >>> with h5py.File("data/run1.h5", "a") as f:
        ...     test._write_metadata_to_h5(f, num_time_samples=2000)
        """
        # --- Global attributes ---
        f.attrs["creator"] = "Anirban Mukhopadhyay"
        f.attrs["description"] = (
            "A modulation-domain test was conducted on the RF components. "
            "A Gaussian-modulated GHz signal was applied to the component, "
            "and the demodulated output was recorded using a digital storage oscilloscope (DSO)."
        )
        f.attrs["date"] = str(date.today())

        # --- AWG settings ---
        grp = f.create_group("awg_settings")
        dset = grp.create_dataset("peak_to_peak_input_voltage", 
                                data=np.asarray(self.vpp),
                                shape=len(self.vpp),
                                dtype=np.float64,
                                compression="gzip", 
                                compression_opts=9, 
                                shuffle=True)
        dset.attrs["unit"] = "V"

        dset2 = grp.create_dataset("time_periods", 
                                data=np.asarray(self.time_periods),
                                shape=len(self.time_periods),
                                dtype=np.float64,
                                compression="gzip", 
                                compression_opts=9, 
                                shuffle=True)
        dset2.attrs["unit"] = "s"

        grp.attrs["dead_volt"] = f"{self.dead_volt} V"
        grp.attrs["dead_cycle"] = f"{self.dead_cyc} cycles"

        # --- DSO settings ---
        grp = f.create_group("dso_settings")
        dset = grp.create_dataset("observation_time",
                                data=np.linspace(0, 12 * self.time_div, num=num_time_samples, endpoint=False),
                                shape=num_time_samples,
                                dtype=np.float64,
                                compression="gzip", 
                                compression_opts=9, 
                                shuffle=True)
        dset.attrs["unit"] = "s"
        grp.attrs["voltage_division"] = [f"{v} V" for v in self.volt_divs]
        grp.attrs["position"] = self.pos
        grp.attrs["average_count"] = self.avg_count

        # --- RFGEN settings ---
        grp = f.create_group("rfgen_settings")
        grp.attrs["LO_power"] = f"{self.pwr_lo} dBm"
        grp.attrs["LO_frequency"] = f"{self.fc} Hz"

    def repeat_collect_and_save_waveform(
            self,
            savepath: str | Path,
            savefilename: str | None,
            num_iters: int = 10,
            delay: int = 60
    ) -> None:
        """
        Perform repeated waveform acquisition and save all results into one HDF5 file.

        Parameters
        ----------
        savepath : str or Path
            Directory where the output HDF5 file will be stored.
        savefilename : str or None
            Output filename (".h5" will be appended automatically).
        num_iters : int, optional, default=10
            Number of repeated acquisitions.
        delay : int, optional, default=60
            Delay between acquisitions in seconds.

        Notes
        -----
        - Connects instruments once for the entire test sequence.
        - Preallocates all HDF5 datasets to minimize write overhead.
        - Each iteration stores duration and timestamp for traceability.

        Example
        -------
        >>> test.repeat_collect_and_save_waveform(
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
            duration, first_data = self.collect_waveform()
            arr_shape = first_data.shape  # (len(vpp), len(time_periods), len(on_chans), num_points)

            savepath = Path(savepath)
            savepath.mkdir(parents=True, exist_ok=True)

            if not savefilename:
                savefilename = savepath.name
            if not savefilename.lower().endswith(".h5"):
                savefilename += ".h5"
            output_h5_file = savepath / savefilename

            # --- Check if file already exists ---
            if output_h5_file.exists():
                timestamp = datetime.now().strftime("_%Y%m%d_%H%M%S")
                output_h5_file = output_h5_file.with_name(
                    f"{output_h5_file.stem}{timestamp}{output_h5_file.suffix}"
                )
                print(f"⚠️ File already exists. Saving as: {output_h5_file.name}")

            with h5py.File(output_h5_file, "a") as f:
                self._write_metadata_to_h5(f, arr_shape[-1])

                # --- Preallocate main dataset ---
                grp = f.create_group("waveforms_all_iteration")

                dset1 = grp.create_dataset(
                    "reference_modulation_waveforms",
                    shape=(num_iters, arr_shape[0], arr_shape[1], arr_shape[-1]),
                    dtype=np.float32,
                    compression="gzip",
                    compression_opts=9,
                    shuffle=True,
                )
                dset1.attrs["unit"] = "V"

                dset2 = grp.create_dataset(
                    "demodulated_waveforms",
                    shape=(num_iters, arr_shape[0], arr_shape[1], arr_shape[2] - 1, arr_shape[-1]),
                    dtype=np.float32,
                    compression="gzip",
                    compression_opts=9,
                    shuffle=True,
                )
                dset2.attrs["unit"] = "V"

                # --- Preallocate metadata datasets ---
                durations = f.create_dataset("durations", shape=(num_iters,), dtype="float32")
                timestamps = f.create_dataset("timestamps", shape=(num_iters,), dtype="S32")

                # --- Store first iteration ---
                dset1[0] = first_data[:, :, 0, :]
                dset2[0] = first_data[:, :, 1:, :]
                durations[0] = duration
                timestamps[0] = datetime.now().strftime("%Y-%m-%d %H:%M:%S").encode("utf-8")
                f.flush()

                # --- Loop over remaining iterations ---
                for rep in tqdm(range(1, num_iters)):
                    time.sleep(delay)
                    duration, data = self.collect_waveform()
                    dset1[rep] = data[:, :, 0, :]
                    dset2[rep] = data[:, :, 1:, :]
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

    megahz = 1e06
    ghz = 1e09

    secs = 1

    v = 1
    mv = 1e-03

    mus = 1e-06
    ns = 1e-09

    # Vpp variation
    pw, _, pulse = gauss_pulse(FWHM=10 * ns, sample_rate=int(1e9), plot=False)

    if 1:
        peak_2_peak_volts = np.linspace(20, 63.2, 21) * mv  # * 10

        pw = np.array([10 * ns])

        moddemodtest = ModDemodTest(pwr_lo=10 * dbm,
                                    fc=2088.5 * megahz,
                                    vpp=peak_2_peak_volts,
                                    sig=pulse,
                                    time_periods=pw,
                                    dead_cyc=80,
                                    dead_volt=0.0 * v,
                                    time_div=200 * ns,
                                    num_points=None,
                                    avg_count=16,
                                    # volt_divs=[10 * mv, 5 * mv, 5 * mv],
                                    # pos=[-3, 3, 4],
                                    # on_chans=[1, 2, 3],
                                    volt_divs=[10 * mv, 5 * mv],
                                    pos=[-3, 3],
                                    on_chans=[1, 3]
                                    )

        moddemodtest.repeat_collect_and_save_waveform(
            savepath="C:/Users/DELL/Documents/spin_wave_expts/sept_25/mod_demod_test_closed_loop",
            savefilename=f"20251029_mod_demod_test_closed_loop_G2_-23dB_vpp_var_detc_1_to_ps_port1_pw_10ns",
            num_iters=50,
            delay=2 * secs)

    # PW variation
    if 0:
        time.sleep(10)

        FWHM = np.arange(4, 12) * ns
        STD = FWHM / np.sqrt(8 * np.log(2))
        pw = 6 * STD

        peak_2_peak_volts = np.array([60]) * mv * 10

        moddemodtest = ModDemodTest(pwr_lo=10 * dbm,
                                    fc=2088.5 * megahz,
                                    vpp=peak_2_peak_volts,
                                    sig=pulse,
                                    time_periods=pw,
                                    dead_cyc=80,
                                    dead_volt=0.0 * v,
                                    time_div=800 * ns,
                                    num_points=None,
                                    avg_count=16,
                                    volt_divs=[50 * mv, 10 * mv],
                                    pos=[-3, 4],
                                    on_chans=[1, 4]
                                    )

        moddemodtest.repeat_collect_and_save_waveform(
            savepath="C:/Users/DELL/Documents/spin_wave_expts/sept_25/mod_demod_test_no_dut",
            savefilename=f"20251011_mod_demod_test_no_dut_detc_3_pw_var",
            num_iters=38,
            delay=2 * secs)

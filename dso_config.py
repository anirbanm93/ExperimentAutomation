from __future__ import annotations

import numpy as np
from RsInstrument import *
from numpy.typing import NDArray
import time


class DSO:
    """
    Class to control Rohde & Schwarz RTM3004 DSO.

    Parameters
    ----------
    resource_string : str, optional
        VISA resource string identifying the instrument.
        If None, defaults to 'USB::0x0AAD::0x01D6::102243::INSTR'.

    Notes
    -----
    Requires RsInstrument version >= 1.70.0.

    Examples
    --------
    >>> dso = DSO()  # or DSO("TCPIP::169.254.253.36::INSTR")
    >>> dso.set_channel_state([1, 2])
    >>> dso.configure_channel([1], 1e-3, 0.5, 0, num_points=1000, avg_count=8)
    >>> data = dso.get_waveform([1])
    >>> result = dso.run_measurement([1], [1], ["FREQuency"])
    >>> dso.disconnect()
    """

    def __init__(self, resource_string: str | None = None):
        """
        Initialize the DSO instrument connection.

        Parameters
        ----------
        resource_string : str, optional
            VISA resource string identifying the instrument.
            Defaults to USB connection string if None.

        Returns
        -------
        None

        Notes
        -----
        Requires RsInstrument version >= 1.70.0.
        """
        RsInstrument.assert_minimum_version("1.70.0")

        if resource_string is None:
            resource_string = "USB::0x0AAD::0x01D6::102243::INSTR"
            # resource_string = "TCPIP::192.168.1.168::INSTR"

        self.instr = RsInstrument(resource_string)
        self.instr.visa_timeout = 50000  # 50 seconds
        self.instr.opc_timeout = 50000  # 50 seconds

    def configure_channel(
            self,
            chans: list[int],
            time_div: float,
            volt_divs: float | list[float],
            pos: float | list[float],
            num_points: int | None = None,
            avg_count: int = 100,
            trig_level: float | None = None
    ) -> None:
        """
        Configure acquisition settings for specified channels.

        Parameters
        ----------
        chans : list[int]
            List of channel numbers to configure (1-4).
        time_div : float
            Time scale (seconds per division).
        volt_divs : float or list of float
            Voltage scale (volts per division).
        pos : float or list of float
            Vertical position(s) of trace(s), in divisions.
        num_points : int, optional
            Number of acquisition points.
        avg_count : int, optional
            Number of waveforms averaged.
        trig_level : float, optional
            Trigger level for the first channel in chans.

        Returns
        -------
        None
        """

        commands = ([f"CHAN{chan}:COUP DC" for chan in chans] +
                    [f"CHAN{chan}:BANDwidth FULL" for chan in chans] +
                    [f"TIM:SCAL {time_div}"] +
                    [f"CHAN{chan}:SCAL {volt_div}" for chan, volt_div in zip(chans, volt_divs)] +
                    [f"CHAN{chan}:POS {pos}" for chan, pos in zip(chans, pos)] +
                    [f"ACQuire:AVERage:COUNt {avg_count}"] +
                    [f"CHAN{chan}:ARIThmetics AVERage" for chan in chans] +
                    ["FORM REAL,32"] +
                    ["FORM:BORD LSBF"]  # Set transfer format once (32-bit float, little-endian)
                    )

        if num_points is not None:
            commands.append(f"ACQuire:POINts:VALue {num_points}")

        if trig_level is not None:
            commands.extend([f"TRIG:A:SOUR CH{chans[0]}",
                             "TRIG:A:TYPE EDGE",
                             "TRIG:A:EDGE:SLOP POS",
                             f"TRIG:A:LEV{chans[0]} {trig_level}"
                             ])

        for cmd in commands:
            self.instr.write_str_with_opc(cmd)

    def update_trig_level(self, val: float, chan: int = 1) -> None:
        """
        Update the trigger level for a specified channel.

        Parameters
        ----------
        val : float
            New trigger level voltage.
        chan : int, optional
            Channel number (default 1).

        Returns
        -------
        None
        """
        commands = [f"TRIG:A:SOUR CH{chan}",
                    "TRIG:A:TYPE EDGE",
                    "TRIG:A:EDGE:SLOP POS",
                    f"TRIG:A:LEV{chan} {val}"
                    ]
        for cmd in commands:
            self.instr.write_str_with_opc(cmd)

    def update_time_div(self, val: float) -> None:
        """
        Update the time scale (seconds per division).

        Parameters
        ----------
        val : float
            New time scale value.

        Returns
        -------
        None
        """
        self.instr.write_str_with_opc(f"TIM:SCAL {val}")

    def update_volt_div(self, val: float | list[float], chans: list[int]) -> None:
        """
        Update voltage scale(s) for specified channels.

        Parameters
        ----------
        val : float or list of float
            Voltage scale(s) in volts per division.
        chans : list[int]
            List of channels to update.

        Returns
        -------
        None
        """
        for chan, _v in zip(chans, val):
            self.instr.write_str_with_opc(f"CHAN{chan}:SCAL {_v}")

    def set_channel_state(self, on_chans: list[int]) -> None:
        """
        Turn channels ON or OFF by specifying which to turn ON.

        Parameters
        ----------
        on_chans : list[int]
            List of channels to enable (1-4).

        Raises
        ------
        ValueError
            If channel number is out of bounds.

        Returns
        -------
        None
        """
        commands = [f"CHAN{chan}:STAT {'ON' if chan in on_chans else 'OFF'}" for chan in [1, 2, 3, 4]]
        for cmd in commands:
            self.instr.write_str_with_opc(cmd)

    def disconnect(self) -> None:
        """
        Close the instrument connection.

        Notes
        -----
        If the instrument is already disconnected, nothing happens.

        Returns
        -------
        None
        """
        if self.instr:
            print("Disconnecting DSO...")
            self.instr.close()
            self.instr = None

    def get_waveform(self, on_chans: list[int]) -> NDArray[float]:
        """
        Retrieve voltage waveform data from specified channels.

        Parameters
        ----------
        on_chans : list[int]
            List of channels to acquire waveforms from (1-4).

        Returns
        -------
        numpy.ndarray
            Array of shape (n_channels, n_samples) with waveform data as floats.
        """
        data = []

        for on_chan in on_chans:
            time.sleep(0.01)  # optional delay 10 ms
            # --- Query waveform data (raw float values) ---
            raw = self.instr.query_bin_or_ascii_float_list_with_opc(f"CHAN{on_chan}:DATA?")
            data.append(np.array(raw, dtype=np.float32))

        return np.array(data, dtype=np.float32)

    def run_measurement(self, meas_place: list[int], on_chans: list[int], meas_type: list[str]) -> dict:
        """
        Run a single measurement on the oscilloscope and return its result.

        Parameters
        ----------
        meas_place : int
            Measurement slot (1-6).
        on_chans : list[int]
            Channel number to measure (1-4).
        meas_type : str
            Measurement type, e.g., "FREQuency", "PERiod",
            "UPEakvalue", "LPEakvalue", "AMPLitude",
            "MEAN", "RMS", "STDDev", etc.

        Returns
        -------
        float
            The measurement result as returned by the instrument.

        Returns
        -------
        dict
            Dictionary mapping 'chan-xMeasurementType' to measured float values.

        Raises
        ------
        ValueError
            If any measurement place or channel number is out of valid range.
        """
        if not any(1 <= x <= 6 for x in meas_place):
            raise ValueError("measurement place must be between 1 and 6.")
        if not any(1 <= x <= 4 for x in on_chans):
            raise ValueError("channel must be between 1 and 4.")

        commands = []

        for on_chan in on_chans:
            for mp, mt in zip(meas_place, meas_type):
                commands.append(f"MEASurement{mp}:SOURce CH{on_chan}")
                commands.append(f"MEASurement{mp}:ENABle ON")
                commands.append(f"MEASurement{mp}:MAIN {mt}")

        for cmd in commands:
            self.instr.write_str_with_opc(cmd)

        result = {}
        for on_chan in on_chans:
            for mp, mt in zip(meas_place, meas_type):
                result["chan-" + str(on_chan) + mt] = float(self.instr.query_str_with_opc(
                    f"MEASurement{mp}:RESult:ACTual?"
                ))
        return result


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from time import sleep

    on_chans = [4]

    # Connect to scope
    dso = DSO()

    # Turn channel states ON/OFF
    dso.set_channel_state(on_chans)

    # Configure channel
    dso.configure_channel(
        chans=on_chans,
        time_div=200e-06,
        volt_divs=[50e-03] * 2,
        pos=[0] * 2,  # position in divisions
        num_points=None,
        avg_count=100,  # averaging count
        trig_level=75e-03
    )

    sleep(10)

    dso.update_trig_level(chan=on_chans[0], val=80e-03)

    # Acquire waveform data
    data = dso.get_waveform(on_chans)
    print(f"Retrieved {len(data)} points, first 10: {data[:10]}")

    # Plot
    # plt.figure()
    # plt.plot(data, marker='o', linestyle='--')
    # plt.xlabel("Sample")
    # plt.title(f"On channel--{on_chan}")
    # plt.ylabel("Amplitude")

    # Run measurement
    result = dso.run_measurement(
        meas_place=[1, 2],
        on_chans=on_chans,
        meas_type=["FREQuency", "PERiod"]
    )
    print(result)

    # Disconnect
    dso.disconnect()

    plt.show()

from moku.instruments import ArbitraryWaveformGenerator


class AWG:
    """
    Wrapper class for controlling the Moku:Lab Arbitrary Waveform Generator (AWG).

    Parameters
    ----------
    serial : int, optional
        The serial number of the Moku:Lab device.
    ip : str, optional
        The IP address of the Moku:Lab device.

    Returns
    -------
    AWG
        An initialized wrapper object for controlling the AWG.

    Notes
    -----
    Either 'serial' or 'ip' must be provided to connect.
    Configures both output channels with 50 Ohm load during connection.

    Examples
    --------
    >>> awg = AWG(serial=65542)
    >>> awg.connect()
    >>> awg.gen_wave(channel=1, sample_rate='Auto', lut_data=list(sig), frequency=1/T, amplitude=Vpp)
    >>> awg.disconnect()
    """

    def __init__(self, serial: int = None, ip: str = None):
        """
        Initialize the AWG instance.

        Parameters
        ----------
        serial : int, optional
            The serial number of the Moku:Lab device.
        ip : str, optional
            The IP address of the Moku:Lab device.

        Returns
        -------
        None

        Examples
        --------
        >>> awg = AWG(serial=65542)
        """
        self.serial = serial
        self.ip = ip
        self.awg = None

    def connect(self):
        """
        Establish connection to the Moku:Lab AWG.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        Both channels are set to 50 Ohm.
        Raises ValueError if neither 'serial' nor 'ip' is provided.

        Examples
        --------
        >>> awg = AWG(serial=65542)
        >>> awg.connect()
        """
        if self.serial is not None:
            self.awg = ArbitraryWaveformGenerator(serial=self.serial, force_connect=True)
        elif self.ip is not None:
            self.awg = ArbitraryWaveformGenerator(self.ip, force_connect=True)
        else:
            raise ValueError("Must provide either serial or IP to connect to Moku:Lab AWG")

        # Configure channel loads to 50 Ohm
        self.awg.set_output_load(1, "50Ohm")
        self.awg.set_output_load(2, "50Ohm")

    def disconnect(self):
        """
        Disconnect from the Moku:Lab AWG by disabling outputs and releasing ownership.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        Disables outputs on both channels and relinquishes device ownership.

        Examples
        --------
        >>> awg.disconnect()
        """
        if self.awg:
            print("Disconnecting Moku:Lab AWG...")
            self.awg.enable_output(1, False)
            self.awg.enable_output(2, False)
            self.awg.relinquish_ownership()
            self.awg = None

    def gen_wave(self, channel: int, sample_rate: str, lut_data: list,
                 frequency: float, amplitude: float, offset: float = 0.0,
                 strict: bool = False):
        """
        Generate a standard waveform on the specified channel.

        Parameters
        ----------
        channel : int
            AWG output channel (1 or 2).
        sample_rate : str
            Sampling rate ("Auto", "1Gs", "500Ms", "250Ms", "125Ms").
        lut_data : list
            List of waveform data points (LUT).
        frequency : float
            Frequency in Hz.
        amplitude : float
            Peak-to-peak amplitude in volts.
        offset : float, optional
            DC offset voltage.
        strict : bool, optional
            If True, enforce strict waveform generation.

        Returns
        -------
        None

        Notes
        -----
        Calls the AWG's 'generate_waveform' method.

        Examples
        --------
        >>> awg.gen_wave(channel=1, sample_rate='Auto', lut_data=list(sig), frequency=10e3, amplitude=1.0)
        """
        self.awg.generate_waveform(channel=channel,
                                   sample_rate=sample_rate,
                                   lut_data=list(lut_data),
                                   frequency=float(frequency),
                                   amplitude=amplitude,
                                   offset=offset,
                                   strict=strict)


    def setup_waveform_template(self, sample_rate: str, lut_data: list,
                            strict: bool = False, offset_flag: bool = False,
                            dead_cycles: int = 2, dead_voltage: float = 0.0):
        """
        Set up the waveform template with constant parameters for waveform generation.

        Parameters
        ----------
        sample_rate : str
            Sampling rate used for waveform generation (e.g., "Auto", "1Gs", "500Ms", "250Ms", "125Ms").
        lut_data : list
            List containing the lookup table (LUT) data points that define the waveform.
        strict : bool, optional, default=False
            If True, enforces strict validation of waveform parameters before use.
        offset_flag : bool, optional, default=False
            If True, applies a DC offset correction to the waveform.
        dead_cycles : int, optional, default=2
            Number of zero-valued cycles inserted between waveform repetitions.
        dead_voltage : float, optional, default=0.0
            Voltage level used during the dead cycles (in volts).

        Returns
        -------
        None

        Notes
        -----
        Stores the waveform configuration parameters in an internal dictionary 
        (`self._waveform_template`) for later use in waveform generation or upload.

        Examples
        --------
        >>> awg.setup_waveform_template(
        ...     sample_rate="1 GSa/s",
        ...     lut_data=[0.0, 0.5, 1.0, 0.5, 0.0],
        ...     strict=True,
        ...     offset_flag=False,
        ...     dead_cycles=4,
        ...     dead_voltage=0.0
        ... )
        """
        self._waveform_template = {
            "sample_rate": sample_rate,
            "lut_data": list(lut_data),
            "strict": strict,
            "offset_flag": offset_flag,
            "dead_cycles": dead_cycles,
            "dead_voltage": dead_voltage
        }


    def gen_pulsed_wave(self, channels, frequency: float, amplitude: float,
                    sample_rate: str = None, lut_data: list = None,
                    strict: bool = None, offset_flag: bool = None,
                    dead_cycles: int = None, dead_voltage: float = None):
        """
        Generate a pulsed waveform on the specified channel.

        Parameters
        ----------
        channel : int
            AWG output channel (1 or 2).
        sample_rate : str
            Sampling rate ("Auto", "1Gs", "500Ms", "250Ms", "125Ms").
        lut_data : list
            List of waveform data points (LUT).
        frequency : float
            Frequency in Hz.
        amplitude : float
            Peak-to-peak amplitude in volts.
        strict : bool, optional
            If True, enforce strict waveform generation.
        offset_flag : bool, optional
            If True, use dead_voltage as offset.
        dead_cycles : int, optional
            Number of dead cycles.
        dead_voltage : float, optional
            Dead voltage in volts.

        Returns
        -------
        None

        Notes
        -----
        Generates waveform and applies pulse modulation via the AWG.

        Examples
        --------
        >>> awg.gen_pulsed_wave(channel=1, sample_rate='Auto', lut_data=list(sig), frequency=10e3,
        ...                     amplitude=1.0, dead_cycles=5, dead_voltage=0)
        """
        # Ensure channels is a list
        if isinstance(channels, int):
            channels = [channels]

        # Load defaults from stored template
        params = self._waveform_template.copy() if hasattr(self, "_waveform_template") else {}
        sample_rate = sample_rate or params.get("sample_rate", "Auto")
        lut_data = lut_data or params.get("lut_data", [])
        strict = strict if strict is not None else params.get("strict", False)
        offset_flag = offset_flag if offset_flag is not None else params.get("offset_flag", False)
        dead_cycles = dead_cycles or params.get("dead_cycles", 2)
        dead_voltage = dead_voltage or params.get("dead_voltage", 0.0)
        
        # Loop through all channels
        for ch in channels:
            # Generate waveform
            self.awg.generate_waveform(
                channel=ch,
                sample_rate=sample_rate,
                lut_data=list(lut_data),
                frequency=float(frequency),
                amplitude=amplitude,
                offset=dead_voltage if offset_flag else 0,
                strict=strict
            )
            # Apply pulse modulation
            self.awg.pulse_modulate(
                channel=ch,
                dead_cycles=dead_cycles,
                dead_voltage=dead_voltage
            )


# Example usage at the end of the file
if __name__ == "__main__":
    from pulse_lib import gauss_pulse
    from time import sleep
    from dso_config import DSO
    import matplotlib.pyplot as plt
    import numpy as np

    # Units
    dbm = 1

    megahz = 1e06
    ghz = 1e09

    v = 1
    mv = 1e-03

    mus = 1e-06
    ns = 1e-09

    Vpp = 200e-03

    awg = AWG(serial=65542)
    awg.connect()

    pw, _, pulse = gauss_pulse(FWHM=10*ns, sample_rate=int(100e9), plot=False)

    awg.gen_pulsed_wave(channel=1, sample_rate='Auto',
                        lut_data=pulse, frequency=1 / pw,
                        amplitude=Vpp/2, strict=False,
                        dead_cycles=80, dead_voltage=0*v)

    # awg.gen_pulsed_wave(channel=2, sample_rate='Auto',
    #                     lut_data=pulse, frequency=1 / pw,
    #                     amplitude=Vpp/2, strict=False,
    #                     dead_cycles=80, dead_voltage=0*v)

    # Least amount of delay between two consecutive measurements
    on_chans = [1]

    # Connect to scope
    dso = DSO()

    # Turn channel states ON/OFF
    for i in range(1, 5):
        dso.set_channel_state(i, "ON" if i in on_chans else "OFF")

    # Configure channel
    data = []
    for on_chan in on_chans:
        dso.configure_channel(
            chan=on_chan,
            time_div=0.4*mus,
            volt_div=25*mv,
            pos=-3,  # position in divisions
            num_points=None,
            avg_count=100,  # averaging count
        )

    dso.update_trig_level(chan=1, val=50*mv)

    for _ in range(2):
        for on_chan in on_chans:
            # Acquire waveform data
            data.append(dso.get_waveform(on_chan))
            sleep(10)

    awg.disconnect()
    dso.disconnect()

    # Plot
    data = np.array(data).reshape((2, len(on_chans), -1))
    print(data.shape)
    plt.figure()
    plt.plot(data[0, 0, :], marker='o', linestyle='--')
    plt.plot(data[-1, 0, :], marker='o', linestyle='--')
    plt.xlabel("Sample")
    plt.title(f"On channel-{on_chans[0]}")
    plt.ylabel("Amplitude")
    #
    # plt.figure()
    # plt.plot(data[1, 1, :], marker='o', linestyle='--')
    # plt.xlabel("Sample")
    # plt.title(f"On channel-{on_chans[1]}")
    # plt.ylabel("Amplitude")

    plt.show()

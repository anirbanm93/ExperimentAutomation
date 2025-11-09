import serial


class RFGEN:
    """
    Class to control the Windfreak Synth HD Pro RF Generator via serial connection.

    Parameters
    ----------
    port : str, optional
        Serial port to which the RF generator is connected. Default is "COM3".
    baudrate : int, optional
        Baud rate for serial communication. Default is 9600.

    Methods
    -------
    connect()
        Establish serial connection to the RF generator and configure defaults.
    set_pwr_freq(pwr, freq)
        Set generator output power (dBm) and frequency (Hz).
    disconnect()
        Disable RF output and close the serial connection.

    Notes
    -----
    - Make sure the RF generator is powered on before calling connect().
    - Frequency is set in Hz but printed in MHz.
    - Power and frequency values must be within device limits.

    Example
    -------
    try:
        rf = RFGEN("COM3")   # Update port if needed
        rf.connect()
        rf.set_pwr_freq(10, 60e6)       # 10 dBm, 60 MHz
    except Exception as e:
        print(f"Error: {e}")
    finally:
        rf.disconnect()
    """

    def __init__(self, port: str = "COM3", baudrate: int = 9600):
        """
        Initialize the RF generator class with serial port parameters.

        Parameters
        ----------
        port : str, optional
            Serial port to which the RF generator is connected. Default is "COM3".
        baudrate : int, optional
            Baud rate for serial communication. Default is 9600.

        Returns
        -------
        None

        Notes
        -----
        - Initializes internal serial handle to None.
        """
        self.port = port
        self.baudrate = baudrate
        self.rfgen = None

    def connect(self) -> None:
        """
        Establish serial connection to the RF generator and configure defaults.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        serial.SerialException
            If the connection to the specified port cannot be established.

        Notes
        -----
        - Configures internal reference to 10 MHz, selects channel 1, and enables RF output.

        Example
        -------
        rf = RFGEN("COM3")
        rf.connect()
        """
        if self.rfgen and self.rfgen.is_open:
            self.rfgen.close()

        self.rfgen = serial.Serial(
            self.port,
            self.baudrate,
            timeout=None,
            parity=serial.PARITY_NONE,
            bytesize=serial.EIGHTBITS
        )

        # Configure RF generator defaults
        self.rfgen.write(str.encode("x2"))  # Internal reference to 10 MHz
        self.rfgen.write(str.encode("C1"))  # Select channel 1
        self.rfgen.write(str.encode("E1r1"))  # Enable RF output
        print(f"Connected to RF generator on {self.port}")

    def set_pwr_freq(self, pwr: float, freq: float) -> None:
        """
        Set generator output power and frequency.

        Parameters
        ----------
        pwr : float
            Output power in dBm (e.g., 10 for +10 dBm).
        freq : float
            Frequency in Hz to set the RF generator.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `pwr` or `freq` are not valid ranges supported by device.
        serial.SerialException
            If the serial connection is not established.

        Notes
        -----
        - Frequency is converted to MHz for generator commands.
        - Flushes input buffer after setting parameters.

        Example
        -------
        rf.set_pwr_freq(10, 60e6)  # Set 10 dBm at 60 MHz
        """
        if not self.rfgen or not self.rfgen.is_open:
            raise serial.SerialException("RF generator is not connected. Call connect() first.")

        self.rfgen.write(str.encode(f"W{pwr}"))  # Set power
        # print(f"Power input: {pwr:0.2f} dBm")

        self.rfgen.write(str.encode(f"f{freq * 1e-06}"))  # Set frequency
        # print(f"Frequency: {freq * 1e-06:0.4f} MHz")

        self.rfgen.flushInput()

    def set_pwr(self, pwr: float) -> None:
        """
        Set generator output power.

        Parameters
        ----------
        pwr : float
            Output power in dBm (e.g., 10 for +10 dBm).

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `pwr` are not valid ranges supported by device.
        serial.SerialException
            If the serial connection is not established.

        Notes
        -----
        - Set power in dBm.
        - Set power before setting the frequency.

        Example
        -------
        rf.set_pwr(10)  # Set 10 dBm
        """
        if not self.rfgen or not self.rfgen.is_open:
            raise serial.SerialException("RF generator is not connected. Call connect() first.")

        self.rfgen.write(str.encode(f"W{pwr}"))  # Set power
        # print(f"Power input: {pwr:0.2f} dBm")

    def set_freq(self, freq: float) -> None:
        """
        Set generator output frequency.

        Parameters
        ----------
        freq : float
            Frequency in Hz to set the RF generator.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `freq` are not valid ranges supported by device.
        serial.SerialException
            If the serial connection is not established.

        Notes
        -----
        - Frequency is converted to MHz for generator commands.
        - Flushes input buffer after setting parameters.

        Example
        -------
        rf.set_freq(60e6)  # Set 60 MHz
        """
        if not self.rfgen or not self.rfgen.is_open:
            raise serial.SerialException("RF generator is not connected. Call connect() first.")

        self.rfgen.write(str.encode(f"f{freq * 1e-06}"))  # Set frequency
        # print(f"Frequency: {freq * 1e-06:0.4f} MHz")

        self.rfgen.flushInput()

    def set_pwr_freq(self, pwr: float, freq: float) -> None:
        """
        Set generator output power and frequency.

        Parameters
        ----------
        pwr : float
            Output power in dBm (e.g., 10 for +10 dBm).
        freq : float
            Frequency in Hz to set the RF generator.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `pwr` or `freq` are not valid ranges supported by device.
        serial.SerialException
            If the serial connection is not established.

        Notes
        -----
        - Frequency is converted to MHz for generator commands.
        - Flushes input buffer after setting parameters.

        Example
        -------
        rf.set_pwr_freq(10, 60e6)  # Set 10 dBm at 60 MHz
        """
        if not self.rfgen or not self.rfgen.is_open:
            raise serial.SerialException("RF generator is not connected. Call connect() first.")

        self.rfgen.write(str.encode(f"W{pwr}"))  # Set power
        # print(f"Power input: {pwr:0.2f} dBm")

        self.rfgen.write(str.encode(f"f{freq * 1e-06}"))  # Set frequency
        # print(f"Frequency: {freq * 1e-06:0.4f} MHz")

        self.rfgen.flushInput()

    def disconnect(self) -> None:
        """
        Disable RF output and close the serial connection.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        serial.SerialException
            If the serial connection cannot be closed.

        Notes
        -----
        - Sends command to disable RF output before closing the port.
        - Sets internal serial handle to None.

        Example
        -------
        rf.disconnect()
        """
        if self.rfgen and self.rfgen.is_open:
            print("Disconnecting RF generator...")
            self.rfgen.write(str.encode("E0r0"))  # Disable RF output
            self.rfgen.close()
            self.rfgen = None


# Example usage at the end of the file
if __name__ == "__main__":
    from time import sleep

    # Unit
    megahz = 1e06
    dbm = 1

    try:
        rf = RFGEN(port="COM3", baudrate=9600)  # Update port if needed
        rf.connect()
        rf.set_pwr_freq(-10*dbm, 2088.5*megahz)  # 10 dBm, 60e06 Hz
    except Exception as e:
        print(f"Error: {e}")
    finally:
        sleep(10)
        rf.disconnect()

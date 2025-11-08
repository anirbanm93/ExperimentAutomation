# ExperimentAutomation
This repository contains Python scripts for automating experimental control and data acquisition in an RF laboratory setup.
The automation framework interfaces with multiple test and measurement instruments, including:
* Arbitrary Waveform Generator (Moku:Lab)
* Digital Storage Oscilloscope (Rohde & Schwarz RTM3004)
* RF Signal Generator (Windfreak SynthHD Pro V1.4)
* RF Spectrum Analyzer (Tektronix RSA306B)

The core modules, ModDemodTest and SpectrumTest, provide a unified workflow that:
* Initialize and power on all connected instruments.
* Configure each device according to the specified experimental parameters.
* Automate waveform generation, temporal trace measurement, and spectrum acquisition.
* Store all acquired data in a structured HDF5 format to ensure reproducibility and consistency.
* Safely power down all instruments upon completion or in the event of an error.

This framework streamlines the execution of RF experiments and data management, enabling repeatable characterization of magnonic systems, particularly spin-wave active ring oscillators, as well as general RF devices.

## License
This project is distributed under a Custom Academic License. Free for academic and research use, provided proper citation is given. Commercial use requires a separate license agreement. See the [LICENSE](./LICENSE) file for full terms.

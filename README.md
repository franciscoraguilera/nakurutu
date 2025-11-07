# NAKURUTU I Rocket Simulation

High-fidelity flight simulation for the NAKURUTU I rocket using RocketPy. This simulation incorporates real-world atmospheric data, Mach-dependent aerodynamics, and a reefing parachute recovery system.

## Features

- **Real Weather Data**: Uses GFS forecast data with automatic fallback to standard atmosphere
- **Mach-Dependent Aerodynamics**: Loads drag coefficient curves from CSV for both powered and unpowered flight phases
- **Reefing Recovery**: Two-stage parachute deployment (drogue at apogee, main at 500m AGL)
- **Multiple Output Formats**:
  - Google Earth KML trajectory file
  - Static 3D trajectory plot (PNG)
  - Interactive 3D trajectory (HTML with Plotly)
  - Comprehensive 6-page PDF flight report

## Requirements

   ```bash
pip install numpy rocketpy matplotlib plotly
```

## Rocket Specifications

- **Motor**: AeroTech M2000R RSE
- **Dry Mass**: 13.194 kg (without motor)
- **Diameter**: 156 mm
- **Nose Cone**: LV-Haack profile, 560 mm length
- **Fins**: 4x trapezoidal fins with 43.4° sweep angle
- **Recovery**: Reefing system with drogue (Cd*S = 1.012 m²) and main (Cd*S = 2.976 m²)

## File Structure

```
.
├── simple_flight.py           # Main simulation script
├── docs/
│   └── AeroTech_M2000R.rse    # Motor thrust curve data
├── data/
│   └── rockets/
│       └── nakurutu/
│           ├── powerOFF_CD.csv # Drag coefficient (motor off)
│           └── powerON_CD.csv  # Drag coefficient (motor on)
├── results/                    # Generated outputs
└── logs/                       # Simulation logs
```

## Usage

Run the simulation:

```bash
python simple_flight.py
```

The script will automatically:
1. Download current GFS weather forecast (or use standard atmosphere if unavailable)
2. Parse motor data from the RSE file
3. Load Mach-dependent drag curves
4. Simulate the full flight trajectory
5. Generate all output files in the `results/` directory

## Launch Configuration

Default launch parameters:
- **Location**: -21.938328°, -48.949887° at 493.1 m elevation
- **Rail Length**: 6.0 m
- **Launch Angle**: 87° (3° from vertical)
- **Heading**: 40° (northeast)

## Outputs

After running, you'll find in the `results/` directory:

1. **nakurutu_trajectory.kml** - Import into Google Earth to visualize the flight path
2. **nakurutu_trajectory_3d.png** - Static 3D plot of the trajectory
3. **nakurutu_interactive_3d_[timestamp].html** - Interactive 3D visualization (opens automatically in browser)
4. **nakurutu_flight_report_[timestamp].pdf** - Comprehensive 6-page report with:
   - Flight summary and key metrics
   - Rocket and motor specifications
   - Altitude, velocity, and acceleration plots
   - Mach number progression
   - 3D trajectory visualization
   - Complete flight data table

## Key Flight Metrics

The simulation tracks and reports:
- Apogee altitude and time
- Maximum velocity and Mach number
- Total flight time
- Impact velocity and drift distance
- Landing coordinates
- Acceleration profile
- Static margin and thrust-to-weight ratio

## Aerodynamics

The simulation uses realistic drag modeling:
- Separate Cd curves for powered and unpowered flight
- Linear interpolation between Mach data points
- Constant extrapolation outside data range
- Automatic data validation and cleaning
- Cd values clamped to [0.02, 2.5] for safety

## Recovery System

Two-stage reefing system:
1. **Drogue deployment**: Triggered at apogee (reefed configuration)
2. **Main deployment**: Triggered at 500 m AGL during descent only
   - Includes descent check (vz < 0) to prevent premature deployment

## Notes

- The interactive HTML file requires a modern web browser with JavaScript enabled
- Large trajectory datasets are downsampled to 1000 points for performance
- All atmospheric data is fetched automatically - no manual downloads needed
- The simulation includes realistic noise and lag parameters for parachute deployment

## Troubleshooting

**GFS Forecast Fails**: The script automatically falls back to standard atmosphere. Check your internet connection if you need real weather data.

**Missing CSV Files**: Ensure the drag coefficient CSV files are present in `data/rockets/nakurutu/`. The script expects columns: angle, Mach, Cd.

**Plotly Not Available**: Install with `pip install plotly` to enable interactive 3D visualization.

## License

This simulation uses RocketPy, an open-source rocket trajectory simulation library.

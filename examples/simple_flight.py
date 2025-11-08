"""
NAKURUTU I – Advanced RocketPy Simulation
- AeroTech M2000R RSE thrust curve
- Mach-dependent drag from CSV (powerON / powerOFF) with robust preprocessing
- Original geometry restored (using sweep instead of large fin cant)
- Reefing recovery: drogue @ apogee, main @ 500 m AGL (descent only)
- GFS weather forecast with safe fallback
"""

import os
import sys
import csv
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import numpy as np
from rocketpy import Environment, SolidMotor, Rocket, Flight, Function

# Plotting libraries
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec

# Try to import plotly for interactive 3D
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("[INFO] Plotly not available. Install with: pip install plotly")


# ---------------------- Paths / Folders ----------------------
SCRIPT_DIR = os.path.dirname(__file__)
WORKSPACE_ROOT = os.path.join(SCRIPT_DIR, "..")
RESULTS_DIR = os.path.join(WORKSPACE_ROOT, "results")
LOGS_DIR = os.path.join(WORKSPACE_ROOT, "logs")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

RSE_PATH = os.path.join(WORKSPACE_ROOT, "docs", "AeroTech_M2000R.rse")
POWER_OFF_CSV = os.path.join(WORKSPACE_ROOT, "data", "rockets", "nakurutu", "powerOFF_CD.csv")
POWER_ON_CSV  = os.path.join(WORKSPACE_ROOT, "data", "rockets", "nakurutu", "powerON_CD.csv")


# ---------------------- Helpers ----------------------
def parse_rse_file(rse_path):
    """Return (thrust_curve, motor_params) from an .rse file."""
    tree = ET.parse(rse_path)
    root = tree.getroot()
    engine = root.find(".//engine")
    if engine is None:
        raise ValueError("No <engine> element in RSE")

    def _f(attr, default=0.0):
        v = engine.get(attr)
        return float(v) if v is not None else float(default)

    motor_params = {
        "total_mass":      _f("initWt") / 1000.0,  # g→kg
        "propellant_mass": _f("propWt") / 1000.0,  # g→kg
        "burn_time":       _f("burn-time"),
        "diameter":        _f("dia") / 1000.0,     # mm→m
        "length":          _f("len") / 1000.0,     # mm→m
        "Itot":            _f("Itot"),             # N·s
        "avgThrust":       _f("avgThrust"),
        "peakThrust":      _f("peakThrust"),
        "mfg":             engine.get("mfg", "Unknown"),
        "code":            engine.get("code", "Unknown"),
    }
    motor_params["dry_mass"] = motor_params["total_mass"] - motor_params["propellant_mass"]

    thrust_curve = []
    for e in engine.findall(".//eng-data"):
        t = float(e.get("t", 0.0))
        f = float(e.get("f", 0.0))
        thrust_curve.append((t, f))
    thrust_curve.sort(key=lambda x: x[0])
    return thrust_curve, motor_params


def load_mach_cd_curve(csv_path):
    """
    CSV expected: angle, Mach, Cd  (we ignore angle)
    Returns a robust, sorted list of unique (Mach, Cd) pairs:
      - Non-finite rows removed
      - Duplicate Mach values averaged
      - Mach >= 0 ensured
      - Cd clipped to a sane [0.02, 2.5] range
      - Endpoints extended slightly for stable constant extrapolation
    """
    pairs_raw = []
    with open(csv_path, "r", newline="") as f:
        r = csv.reader(f)
        for row in r:
            if len(row) < 3:
                continue
            try:
                mach = float(row[1])
                cd   = float(row[2])
                if not np.isfinite(mach) or not np.isfinite(cd):
                    continue
                if mach < 0:
                    continue
                cd = float(np.clip(cd, 0.02, 2.5))
                pairs_raw.append((mach, cd))
            except Exception:
                continue

    if not pairs_raw:
        raise ValueError(f"No valid Mach/Cd data in {csv_path}")

    # Sort by Mach and merge duplicates
    pairs_raw.sort(key=lambda x: x[0])
    uniq = []
    eps = 1e-9
    accum_cd = []
    current_m = None
    for m, c in pairs_raw:
        if current_m is None or abs(m - current_m) > eps:
            if accum_cd:
                uniq.append((current_m, float(np.mean(accum_cd))))
            current_m = m
            accum_cd = [c]
        else:
            accum_cd.append(c)
    if accum_cd:
        uniq.append((current_m, float(np.mean(accum_cd))))

    # Ensure we start at Mach 0 and extend the upper end a bit
    first_m, first_c = uniq[0]
    last_m, last_c   = uniq[-1]
    if first_m > 0.0:
        uniq.insert(0, (0.0, first_c))
    uniq.append((last_m + 0.2, last_c))

    return uniq


def cd_function_from_pairs(pairs):
    """Create a linear-interpolated Function(Mach -> Cd) with constant extrapolation."""
    return Function(
        source=pairs,           # list[(Mach, Cd)]
        inputs=["Mach"],
        outputs=["Cd"],
        interpolation="linear",   # robust vs. spline
        extrapolation="constant", # clamp outside range
    )


def export_kml_compat(flight, file_name, **kwargs):
    """Use new exporter if available; else fall back to Flight.export_kml."""
    try:
        from rocketpy.simulation.flight_data_exporter import FlightDataExporter
        FlightDataExporter(flight).export_kml(file_name=file_name, **kwargs)
        print(f"File  {os.path.abspath(file_name)}  saved with success! (new exporter)")
    except Exception:
        # Fallback (deprecated in v1.12+)
        flight.export_kml(file_name=file_name, **kwargs)
        print(f"File  {os.path.abspath(file_name)}  saved with success!")


def generate_interactive_3d_html(flight, sim_folder, timestamp, env):
    """
    Generate an interactive 3D trajectory plot using Plotly.
    Creates an HTML file that can be opened in a browser for full interactivity.
    """
    if not PLOTLY_AVAILABLE:
        print("  [WARN] Plotly not available. Skipping interactive 3D HTML.")
        return None
    
    print("  [5/10] Generating interactive 3D HTML...")
    
    try:
        # Extract trajectory data directly as 1D numpy arrays and flatten
        time = np.atleast_1d(np.array(flight.time)).flatten()
        x = np.atleast_1d(np.array(flight.x)).flatten()
        y = np.atleast_1d(np.array(flight.y)).flatten()
        z_asl = np.atleast_1d(np.array(flight.z)).flatten()
        z = z_asl - env.elevation  # AGL
        
        # Calculate additional data for tooltips
        vx = np.atleast_1d(np.array(flight.vx)).flatten()
        vy = np.atleast_1d(np.array(flight.vy)).flatten()
        vz = np.atleast_1d(np.array(flight.vz)).flatten()
        velocity = np.sqrt(vx**2 + vy**2 + vz**2)
        
        # Create hover text with detailed information (use list comprehension for safety)
        hover_text = [
            (
                f"<b>Time:</b> {time[i]:.2f} s<br>"
                f"<b>Altitude:</b> {z[i]:.2f} m ({z[i]*3.28084:.0f} ft)<br>"
                f"<b>Velocity:</b> {velocity[i]:.2f} m/s ({velocity[i]*2.23694:.1f} mph)<br>"
                f"<b>Position X:</b> {x[i]:.2f} m<br>"
                f"<b>Position Y:</b> {y[i]:.2f} m<br>"
                f"<b>Downrange:</b> {np.sqrt(x[i]**2 + y[i]**2):.2f} m"
            )
            for i in range(min(len(time), 1000))  # Limit to first 1000 points for performance
        ]
        
        # Limit data points for performance
        n_points = min(len(time), 1000)
        x_plot = x[:n_points]
        y_plot = y[:n_points]
        z_plot = z[:n_points]
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        # Add trajectory line
        fig.add_trace(go.Scatter3d(
            x=x_plot, y=y_plot, z=z_plot,
            mode='lines+markers',
            name='Trajectory',
            line=dict(color=z_plot, colorscale='Viridis', width=4,
                     colorbar=dict(title="Altitude (m)", x=1.1)),
            marker=dict(size=3, color=z_plot, colorscale='Viridis', showscale=False),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            showlegend=True
        ))
        
        # Mark key events
        # Launch point
        fig.add_trace(go.Scatter3d(
            x=[float(x_plot[0])], y=[float(y_plot[0])], z=[float(z_plot[0])],
            mode='markers', name='Launch',
            marker=dict(size=10, color='green', symbol='diamond'),
            hovertemplate='<b>Launch Point</b><br>Time: 0.0 s<extra></extra>',
            showlegend=True
        ))
        
        # Apogee (use full data to find actual apogee)
        apogee_idx = int(np.argmax(z))
        if apogee_idx < len(x) and apogee_idx < len(y) and apogee_idx < len(z) and apogee_idx < len(time):
            apogee_x = float(x[apogee_idx])
            apogee_y = float(y[apogee_idx])
            apogee_z = float(z[apogee_idx])
            apogee_t = float(time[apogee_idx])
            fig.add_trace(go.Scatter3d(
                x=[apogee_x], y=[apogee_y], z=[apogee_z],
                mode='markers', name='Apogee',
                marker=dict(size=12, color='red', symbol='diamond'),
                hovertemplate=(
                    f'<b>Apogee</b><br>'
                    f'Time: {apogee_t:.2f} s<br>'
                    f'Altitude: {apogee_z:.2f} m ({apogee_z*3.28084:.0f} ft)<extra></extra>'
                ),
                showlegend=True
            ))
        
        # Landing point (use full data for actual landing)
        if len(x) > 0 and len(y) > 0 and len(z) > 0 and len(time) > 0:
            landing_x = float(x[-1])
            landing_y = float(y[-1])
            landing_z = float(z[-1])
            landing_t = float(time[-1])
            landing_dist = float(np.sqrt(landing_x**2 + landing_y**2))
            fig.add_trace(go.Scatter3d(
                x=[landing_x], y=[landing_y], z=[landing_z],
                mode='markers', name='Landing',
                marker=dict(size=10, color='blue', symbol='diamond'),
                hovertemplate=(
                    f'<b>Landing Point</b><br>'
                    f'Time: {landing_t:.2f} s<br>'
                    f'Distance: {landing_dist:.2f} m<extra></extra>'
                ),
                showlegend=True
            ))
        
        # Update layout
        try:
            apogee_val = float(flight.apogee - env.elevation)
            max_speed_val = float(flight.max_speed)
            t_final_val = float(flight.t_final)
        except:
            apogee_val = float(apogee_z)
            max_speed_val = float(np.max(velocity))
            t_final_val = float(time[-1])
        
        fig.update_layout(
            title=dict(
                text=f'<b>NAKURUTU I - Interactive 3D Flight Trajectory</b><br>'
                     f'<sub>Apogee: {apogee_val:.2f} m | '
                     f'Max Speed: {max_speed_val:.2f} m/s | '
                     f'Flight Time: {t_final_val:.2f} s</sub>',
                x=0.5, xanchor='center'
            ),
            scene=dict(
                xaxis_title='East-West (m)',
                yaxis_title='North-South (m)',
                zaxis_title='Altitude AGL (m)',
                aspectmode='data',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
                bgcolor='rgba(240, 240, 240, 0.9)'
            ),
            hoverlabel=dict(bgcolor="white", font_size=12),
            legend=dict(x=0.02, y=0.98, bgcolor='rgba(255, 255, 255, 0.8)'),
            width=1400, height=900,
            template='plotly_white'
        )
        
        # Add instructions
        fig.add_annotation(
            text=(
                "<b>Controls:</b><br>"
                "Left-click + drag: Rotate<br>"
                "Right-click + drag: Pan<br>"
                "Scroll: Zoom<br>"
                "Hover: View data"
            ),
            xref="paper", yref="paper",
            x=0.02, y=0.02,
            xanchor='left', yanchor='bottom',
            showarrow=False,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black', borderwidth=1,
            font=dict(size=10)
        )
        
        # Save to HTML file
        html_path = os.path.join(sim_folder, f'nakurutu_interactive_3d_{timestamp}.html')
        fig.write_html(html_path, config={
            'toImageButtonOptions': {
                'format': 'png',
                'filename': f'nakurutu_trajectory_3d_{timestamp}',
                'height': 1080, 'width': 1920, 'scale': 2
            },
            'displayModeBar': True,
            'displaylogo': False
        })
        
        print(f"  [OK] Interactive 3D HTML saved: {html_path}")
        
        # Try to open in browser
        try:
            import webbrowser
            webbrowser.open('file://' + os.path.abspath(html_path))
            print(f"  [OK] Opening in browser...")
        except:
            pass
        
        return html_path
        
    except Exception as e:
        print(f"  [ERROR] Failed to generate interactive 3D HTML: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_motor_plots(motor, sim_folder, timestamp):
    """Generate and save motor visualization plots."""
    print("  [6/10] Generating motor plots...")
    try:
        # Motor representation
        fig1_path = os.path.join(sim_folder, f'motor_representation_{timestamp}.png')
        motor.plots.draw(filename=fig1_path)
        print(f"    [OK] Motor representation saved")
        
        # Thrust curve
        fig2_path = os.path.join(sim_folder, f'motor_thrust_curve_{timestamp}.png')
        motor.plots.thrust(filename=fig2_path)
        print(f"    [OK] Thrust curve saved")
        
        return fig1_path, fig2_path
    except Exception as e:
        print(f"    [WARN] Could not generate motor plots: {e}")
        return None, None


def generate_rocket_plots(rocket, sim_folder, timestamp):
    """Generate and save rocket visualization."""
    print("  [7/10] Generating rocket schematic...")
    try:
        fig_path = os.path.join(sim_folder, f'rocket_schematic_{timestamp}.png')
        rocket.draw(filename=fig_path)
        print(f"    [OK] Rocket schematic saved")
        return fig_path
    except Exception as e:
        print(f"    [WARN] Could not generate rocket schematic: {e}")
        return None


def generate_atmospheric_plots(env, sim_folder, timestamp):
    """Generate and save atmospheric condition plots."""
    print("  [8/10] Generating atmospheric plots...")
    try:
        # Wind conditions
        fig1_path = os.path.join(sim_folder, f'atmospheric_conditions_{timestamp}.png')
        env.plots.atmospheric_model(filename=fig1_path)
        print(f"    [OK] Atmospheric conditions saved")
        return fig1_path
    except Exception as e:
        print(f"    [WARN] Could not generate atmospheric plots: {e}")
        return None


def generate_pdf_report(flight, rocket, motor, env, sim_folder, timestamp, motor_params):
    """
    Generate a comprehensive PDF flight report with all graphs and data.
    Similar to NDRT 2020 style report with motor, rocket, and atmospheric visualizations.
    """
    print("  [9/10] Generating comprehensive PDF report...")
    
    pdf_path = os.path.join(sim_folder, f'nakurutu_flight_report_{timestamp}.pdf')
    
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
        
        with PdfPages(pdf_path) as pdf:
            # ========== PAGE 1: TITLE PAGE ==========
            fig = plt.figure(figsize=(8.5, 11))
            fig.patch.set_facecolor('white')
            ax = fig.add_subplot(111)
            ax.axis('off')
            
            # Title
            fig.text(0.5, 0.85, 'NAKURUTU I', ha='center', fontsize=36, weight='bold',
                    fontfamily='sans-serif')
            fig.text(0.5, 0.80, 'Flight Simulation Report', ha='center', fontsize=20,
                    fontfamily='sans-serif', style='italic')
            
            # Date and location
            fig.text(0.5, 0.70, f'Launch Date: {datetime.now().strftime("%B %d, %Y")}', 
                    ha='center', fontsize=14, fontfamily='sans-serif')
            fig.text(0.5, 0.67, f'Location: {env.latitude:.6f}°, {env.longitude:.6f}° | Elevation: {env.elevation:.1f} m',
                    ha='center', fontsize=12, fontfamily='sans-serif')
            
            # Key results box
            try:
                apogee_agl = flight.apogee - env.elevation
                fig.text(0.5, 0.55, 'KEY FLIGHT RESULTS', ha='center', fontsize=16,
                        weight='bold', fontfamily='sans-serif')
                
                results_text = f"""
Apogee Altitude: {apogee_agl:.1f} m ({apogee_agl*3.28084:.0f} ft) AGL
Apogee Time: {flight.apogee_time:.2f} s
Maximum Velocity: {flight.max_speed:.1f} m/s ({flight.max_speed*2.23694:.1f} mph)
Maximum Mach Number: {flight.max_mach_number:.3f}
Flight Time: {flight.t_final:.1f} s
Impact Velocity: {flight.impact_velocity:.2f} m/s
Drift Distance: {np.sqrt(flight.x_impact**2 + flight.y_impact**2):.1f} m ({np.sqrt(flight.x_impact**2 + flight.y_impact**2)*3.28084:.0f} ft)
"""
                fig.text(0.5, 0.35, results_text, ha='center', fontsize=11,
                        fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            except Exception as e:
                fig.text(0.5, 0.45, f'Error extracting results: {e}', ha='center', fontsize=10)
            
            # Footer
            fig.text(0.5, 0.1, 'Generated by RocketPy', ha='center', fontsize=10,
                    fontfamily='sans-serif', style='italic', color='gray')
            fig.text(0.95, 0.02, 'Page 1', ha='right', fontsize=9, color='gray')
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # ========== PAGE 2: MOTOR VISUALIZATION ==========
            fig = plt.figure(figsize=(8.5, 11))
            fig.suptitle('MOTOR VISUALIZATION', fontsize=18, weight='bold', y=0.98)
            
            gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.3, height_ratios=[1, 1])
            
            # Motor diagram
            try:
                ax1 = fig.add_subplot(gs[0, 0])
                motor.plots.draw(ax=ax1)
                ax1.set_title('Solid Motor Representation', fontsize=14, weight='bold', pad=10)
            except Exception as e:
                ax1 = fig.add_subplot(gs[0, 0])
                ax1.axis('off')
                ax1.text(0.5, 0.5, f'Motor diagram unavailable: {e}', ha='center', transform=ax1.transAxes)
            
            # Thrust curve
            try:
                ax2 = fig.add_subplot(gs[1, 0])
                time_array = np.linspace(0, motor_params['burn_time'], 100)
                thrust_array = [motor.thrust(t) for t in time_array]
                ax2.plot(time_array, thrust_array, 'b-', linewidth=2)
                ax2.set_xlabel('Time (s)', fontsize=12)
                ax2.set_ylabel('Thrust (N)', fontsize=12)
                ax2.set_title('Thrust Curve', fontsize=14, weight='bold', pad=10)
                ax2.grid(True, alpha=0.3)
                ax2.axhline(y=motor_params['avgThrust'], color='r', linestyle='--', 
                          alpha=0.5, label=f"Avg: {motor_params['avgThrust']:.1f} N")
                ax2.legend(fontsize=10)
            except Exception as e:
                ax2 = fig.add_subplot(gs[1, 0])
                ax2.axis('off')
                ax2.text(0.5, 0.5, f'Thrust curve unavailable: {e}', ha='center', transform=ax2.transAxes)
            
            fig.text(0.95, 0.02, 'Page 2', ha='right', fontsize=9, color='gray')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # ========== PAGE 3: ROCKET CONFIGURATION ==========
            fig = plt.figure(figsize=(8.5, 11))
            fig.suptitle('ROCKET CONFIGURATION', fontsize=18, weight='bold', y=0.98)
            
            gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
            
            # Motor specifications
            ax1 = fig.add_subplot(gs[0, :])
            ax1.axis('off')
            ax1.text(0.5, 0.9, 'MOTOR SPECIFICATIONS', ha='center', fontsize=14,
                    weight='bold', transform=ax1.transAxes)
            motor_text = f"""
Motor: {motor_params.get('code', 'AeroTech M2000R')} ({motor_params.get('mfg', 'Aerotech')})
Total Impulse: {motor_params.get('Itot', 0):.1f} N·s
Average Thrust: {motor_params.get('avgThrust', 0):.1f} N
Peak Thrust: {motor_params.get('peakThrust', 0):.1f} N
Burn Time: {motor_params.get('burn_time', 0):.2f} s
Propellant Mass: {motor_params.get('propellant_mass', 0):.3f} kg
Dry Mass: {motor_params.get('dry_mass', 0):.3f} kg
Diameter: {motor_params.get('diameter', 0)*1000:.1f} mm
Length: {motor_params.get('length', 0)*1000:.1f} mm
"""
            ax1.text(0.1, 0.4, motor_text, fontsize=10, fontfamily='monospace',
                    transform=ax1.transAxes, verticalalignment='top')
            
            # Rocket specifications
            ax2 = fig.add_subplot(gs[1, :])
            ax2.axis('off')
            ax2.text(0.5, 0.9, 'ROCKET SPECIFICATIONS', ha='center', fontsize=14,
                    weight='bold', transform=ax2.transAxes)
            try:
                rocket_text = f"""
Dry Mass (without motor): {rocket.mass:.3f} kg
Total Mass (at liftoff): {rocket.total_mass(0):.3f} kg
Radius: {rocket.radius*1000:.1f} mm
Center of Mass (without motor): {rocket.center_of_mass_without_motor:.3f} m from nose
Static Margin (at t=0): {rocket.static_margin(0):.3f} calibers
Thrust-to-Weight Ratio: {motor.thrust(0.10) / (rocket.total_mass(0.0) * 9.81):.2f}

AERODYNAMIC SURFACES:
- Nose Cone: LV-Haack, Length 560 mm
- Fins: 4x Trapezoidal, Root: 361 mm, Tip: 60 mm, Span: 156 mm
- Sweep Angle: 43.40 degrees
"""
                ax2.text(0.1, 0.4, rocket_text, fontsize=10, fontfamily='monospace',
                        transform=ax2.transAxes, verticalalignment='top')
            except Exception as e:
                ax2.text(0.5, 0.5, f'Error: {e}', ha='center', transform=ax2.transAxes)
            
            # Recovery system
            ax3 = fig.add_subplot(gs[2, :])
            ax3.axis('off')
            ax3.text(0.5, 0.9, 'RECOVERY SYSTEM', ha='center', fontsize=14,
                    weight='bold', transform=ax3.transAxes)
            recovery_text = """
REEFING PARACHUTE SYSTEM:
- Drogue (Reefed): Cd*S = 1.012 m^2, deploys at apogee
- Main (Full): Cd*S = 2.976 m^2, deploys at 500 m AGL (descent only)
- Sampling Rate: 100 Hz
- Deployment Lag: Drogue 0.3s, Main 0.2s
"""
            ax3.text(0.1, 0.4, recovery_text, fontsize=10, fontfamily='monospace',
                    transform=ax3.transAxes, verticalalignment='top')
            
            fig.text(0.95, 0.02, 'Page 3', ha='right', fontsize=9, color='gray')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # ========== PAGE 4: ROCKET SCHEMATIC & ATMOSPHERIC CONDITIONS ==========
            fig = plt.figure(figsize=(8.5, 11))
            fig.suptitle('ROCKET SCHEMATIC & ATMOSPHERIC CONDITIONS', fontsize=16, weight='bold', y=0.98)
            
            gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3, height_ratios=[1.2, 1])
            
            # Rocket schematic
            try:
                ax1 = fig.add_subplot(gs[0, :])
                rocket.draw(ax=ax1)
                ax1.set_title('Rocket Representation', fontsize=14, weight='bold')
            except Exception as e:
                ax1 = fig.add_subplot(gs[0, :])
                ax1.axis('off')
                ax1.text(0.5, 0.5, f'Rocket schematic unavailable: {e}', ha='center', transform=ax1.transAxes)
            
            # Atmospheric conditions - Wind profile
            try:
                ax2 = fig.add_subplot(gs[1, 0])
                altitudes = np.linspace(0, 6000, 50)
                wind_speeds = []
                wind_dirs = []
                for alt in altitudes:
                    wind_x = env.wind_velocity_x(alt)
                    wind_y = env.wind_velocity_y(alt)
                    speed = np.sqrt(wind_x**2 + wind_y**2)
                    direction = np.degrees(np.arctan2(wind_x, wind_y)) % 360
                    wind_speeds.append(speed)
                    wind_dirs.append(direction)
                
                ax2_twin = ax2.twiny()
                ax2.plot(wind_speeds, altitudes, 'orange', linewidth=2, label='Wind Speed')
                ax2_twin.plot(wind_dirs, altitudes, 'blue', linewidth=2, label='Wind Direction')
                ax2.set_xlabel('Wind Speed (m/s)', fontsize=11, color='orange')
                ax2_twin.set_xlabel('Wind Direction (°)', fontsize=11, color='blue')
                ax2.set_ylabel('Height Above Sea Level (m)', fontsize=11)
                ax2.grid(True, alpha=0.3)
                ax2.tick_params(axis='x', labelcolor='orange')
                ax2_twin.tick_params(axis='x', labelcolor='blue')
            except Exception as e:
                ax2 = fig.add_subplot(gs[1, 0])
                ax2.axis('off')
                ax2.text(0.5, 0.5, f'Wind data unavailable', ha='center', transform=ax2.transAxes)
            
            # Atmospheric conditions - Density & Sound Speed
            try:
                ax3 = fig.add_subplot(gs[1, 1])
                altitudes = np.linspace(0, 6000, 50)
                densities = [env.density(alt) for alt in altitudes]
                sound_speeds = [env.speed_of_sound(alt) for alt in altitudes]
                
                ax3_twin = ax3.twiny()
                ax3.plot(densities, altitudes, 'navy', linewidth=2, label='Density')
                ax3_twin.plot(sound_speeds, altitudes, 'orange', linewidth=2, label='Speed of Sound')
                ax3.set_xlabel('Density (kg/m³)', fontsize=11, color='navy')
                ax3_twin.set_xlabel('Speed of Sound (m/s)', fontsize=11, color='orange')
                ax3.set_ylabel('Height Above Sea Level (m)', fontsize=11)
                ax3.grid(True, alpha=0.3)
                ax3.tick_params(axis='x', labelcolor='navy')
                ax3_twin.tick_params(axis='x', labelcolor='orange')
            except Exception as e:
                ax3 = fig.add_subplot(gs[1, 1])
                ax3.axis('off')
                ax3.text(0.5, 0.5, f'Atmospheric data unavailable', ha='center', transform=ax3.transAxes)
            
            fig.text(0.95, 0.02, 'Page 4', ha='right', fontsize=9, color='gray')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # ========== PAGE 5: ALTITUDE & VELOCITY PLOTS ==========
            fig = plt.figure(figsize=(8.5, 11))
            fig.suptitle('FLIGHT TRAJECTORY - ALTITUDE & VELOCITY', fontsize=16, weight='bold', y=0.98)
            
            gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.4)
            
            # Altitude vs Time
            ax1 = fig.add_subplot(gs[0, 0])
            time_data = np.array(flight.time)
            alt_data = np.array(flight.z) - env.elevation
            # Ensure arrays are same length
            min_len = min(len(time_data), len(alt_data))
            time_data = time_data[:min_len]
            alt_data = alt_data[:min_len]
            ax1.plot(time_data, alt_data, 'b-', linewidth=2)
            ax1.axhline(y=apogee_agl, color='r', linestyle='--', alpha=0.5, label=f'Apogee: {apogee_agl:.1f} m')
            ax1.set_xlabel('Time (s)', fontsize=12)
            ax1.set_ylabel('Altitude AGL (m)', fontsize=12)
            ax1.set_title('Altitude Above Ground Level', fontsize=13, weight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=10)
            
            # Velocity vs Time
            ax2 = fig.add_subplot(gs[1, 0])
            vel_data = np.array([np.sqrt(vx**2 + vy**2 + vz**2) 
                                for vx, vy, vz in zip(flight.vx, flight.vy, flight.vz)])
            vel_data = vel_data[:min_len]
            ax2.plot(time_data, vel_data, 'g-', linewidth=2)
            ax2.axhline(y=flight.max_speed, color='r', linestyle='--', alpha=0.5, 
                       label=f'Max: {flight.max_speed:.1f} m/s')
            ax2.set_xlabel('Time (s)', fontsize=12)
            ax2.set_ylabel('Velocity (m/s)', fontsize=12)
            ax2.set_title('Total Velocity', fontsize=13, weight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=10)
            
            # Vertical Velocity vs Time
            ax3 = fig.add_subplot(gs[2, 0])
            vz_data = np.array(flight.vz)
            vz_data = vz_data[:min_len]
            ax3.plot(time_data, vz_data, 'purple', linewidth=2)
            ax3.axhline(y=0, color='k', linestyle='-', alpha=0.2)
            ax3.set_xlabel('Time (s)', fontsize=12)
            ax3.set_ylabel('Vertical Velocity (m/s)', fontsize=12)
            ax3.set_title('Vertical Velocity', fontsize=13, weight='bold')
            ax3.grid(True, alpha=0.3)
            
            fig.text(0.95, 0.02, 'Page 5', ha='right', fontsize=9, color='gray')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # ========== PAGE 6: ACCELERATION & MACH NUMBER ==========
            fig = plt.figure(figsize=(8.5, 11))
            fig.suptitle('FLIGHT DYNAMICS - ACCELERATION & MACH', fontsize=16, weight='bold', y=0.98)
            
            gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.4)
            
            # Acceleration vs Time
            ax1 = fig.add_subplot(gs[0, 0])
            acc_data = np.array(flight.acceleration)
            acc_data = acc_data[:min_len]
            ax1.plot(time_data, acc_data, 'r-', linewidth=2)
            ax1.set_xlabel('Time (s)', fontsize=12)
            ax1.set_ylabel('Acceleration (m/s²)', fontsize=12)
            ax1.set_title('Total Acceleration', fontsize=13, weight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Mach Number vs Time
            ax2 = fig.add_subplot(gs[1, 0])
            mach_data = np.array(flight.mach_number)
            mach_data = mach_data[:min_len]
            ax2.plot(time_data, mach_data, 'orange', linewidth=2)
            ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Mach 1.0')
            ax2.axhline(y=flight.max_mach_number, color='b', linestyle='--', alpha=0.5,
                       label=f'Max: {flight.max_mach_number:.3f}')
            ax2.set_xlabel('Time (s)', fontsize=12)
            ax2.set_ylabel('Mach Number', fontsize=12)
            ax2.set_title('Mach Number vs Time', fontsize=13, weight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=10)
            
            # Drag Coefficient vs Mach
            ax3 = fig.add_subplot(gs[2, 0])
            # Sample drag coefficient from rocket's power_off_drag function
            try:
                mach_range = np.linspace(0, 1.3, 100)
                if hasattr(rocket.power_off_drag, '__call__'):
                    cd_values = [rocket.power_off_drag(m) for m in mach_range]
                    ax3.plot(mach_range, cd_values, 'b-', linewidth=2, label='Power Off')
                    if hasattr(rocket.power_on_drag, '__call__'):
                        cd_on_values = [rocket.power_on_drag(m) for m in mach_range]
                        ax3.plot(mach_range, cd_on_values, 'r-', linewidth=2, label='Power On')
                    ax3.legend(fontsize=10)
                else:
                    ax3.text(0.5, 0.5, 'Constant drag coefficient', ha='center', transform=ax3.transAxes)
                ax3.set_xlabel('Mach Number', fontsize=12)
                ax3.set_ylabel('Drag Coefficient', fontsize=12)
                ax3.set_title('Drag Coefficient vs Mach Number', fontsize=13, weight='bold')
                ax3.grid(True, alpha=0.3)
            except:
                ax3.text(0.5, 0.5, 'Drag data not available', ha='center', transform=ax3.transAxes)
            
            fig.text(0.95, 0.02, 'Page 6', ha='right', fontsize=9, color='gray')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # ========== PAGE 7: 3D TRAJECTORY ==========
            fig = plt.figure(figsize=(8.5, 11))
            fig.suptitle('3D FLIGHT TRAJECTORY', fontsize=16, weight='bold', y=0.96)
            
            ax = fig.add_subplot(111, projection='3d')
            
            x_data = np.array(flight.x)
            y_data = np.array(flight.y)
            z_data = np.array(flight.z) - env.elevation
            
            # Ensure all arrays have same length
            min_len_3d = min(len(x_data), len(y_data), len(z_data))
            x_data = x_data[:min_len_3d]
            y_data = y_data[:min_len_3d]
            z_data = z_data[:min_len_3d]
            
            # Plot trajectory
            scatter = ax.scatter(x_data, y_data, z_data, c=z_data, cmap='viridis',
                               s=10, alpha=0.6)
            ax.plot(x_data, y_data, z_data, 'b-', alpha=0.3, linewidth=1)
            
            # Mark special points
            if len(x_data) > 0:
                ax.scatter([x_data[0]], [y_data[0]], [z_data[0]], 
                          color='green', s=200, marker='^', label='Launch', edgecolors='black', linewidths=2)
                apogee_idx_3d = int(np.argmax(z_data))  # Use truncated z_data
                if apogee_idx_3d < len(x_data):
                    ax.scatter([x_data[apogee_idx_3d]], [y_data[apogee_idx_3d]], [z_data[apogee_idx_3d]],
                              color='red', s=200, marker='*', label='Apogee', edgecolors='black', linewidths=2)
                ax.scatter([x_data[-1]], [y_data[-1]], [z_data[-1]],
                          color='blue', s=200, marker='v', label='Landing', edgecolors='black', linewidths=2)
            
            ax.set_xlabel('East-West (m)', fontsize=11)
            ax.set_ylabel('North-South (m)', fontsize=11)
            ax.set_zlabel('Altitude AGL (m)', fontsize=11)
            ax.legend(fontsize=10)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
            cbar.set_label('Altitude (m)', fontsize=10)
            
            # Add comprehensive stats text box (NDRT style)
            try:
                sm = rocket.static_margin(0)
                stats_text = f"""FLIGHT SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━
Apogee (AGL):      {apogee_agl:.1f} m ({apogee_agl*3.28084:.0f} ft)
Max Velocity:      {flight.max_speed:.1f} m/s ({flight.max_speed*2.23694:.1f} mph)
Max Mach:          {flight.max_mach_number:.3f}
Flight Time:       {flight.t_final:.1f} s
━━━━━━━━━━━━━━━━━━━━━━━━
Drift Distance:    {np.sqrt(flight.x_impact**2 + flight.y_impact**2):.1f} m ({np.sqrt(flight.x_impact**2 + flight.y_impact**2)*3.28084:.0f} ft)
Landing:           ({flight.x_impact:.1f}, {flight.y_impact:.1f}) m
Impact Velocity:   {abs(flight.impact_velocity):.2f} m/s
━━━━━━━━━━━━━━━━━━━━━━━━
Static Margin:     {sm:.3f} calibers
T/W Ratio:         {motor.thrust(0.10) / (rocket.total_mass(0.0) * 9.81):.2f}
"""
            except Exception as e:
                stats_text = f"""FLIGHT SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━
Apogee:      {apogee_agl:.1f} m
Drift:       {np.sqrt(flight.x_impact**2 + flight.y_impact**2):.1f} m
Landing:     ({flight.x_impact:.1f}, {flight.y_impact:.1f}) m
"""
            
            ax.text2D(0.02, 0.02, stats_text, transform=ax.transAxes,
                     fontsize=8, verticalalignment='bottom', family='monospace',
                     bbox=dict(boxstyle='round', facecolor='#F5F5DC', alpha=0.9, 
                              edgecolor='#8B4513', linewidth=2))
            
            fig.text(0.95, 0.02, 'Page 7', ha='right', fontsize=9, color='gray')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # ========== PAGE 8: SUMMARY TABLE ==========
            fig = plt.figure(figsize=(8.5, 11))
            fig.suptitle('FLIGHT DATA SUMMARY', fontsize=18, weight='bold', y=0.96)
            
            ax = fig.add_subplot(111)
            ax.axis('off')
            
            # Create summary table data
            summary_data = []
            try:
                summary_data = [
                    ['LAUNCH CONDITIONS', ''],
                    ['Launch Rail Length', f'{6.0:.1f} m'],
                    ['Launch Inclination', f'{87.0:.1f}°'],
                    ['Launch Heading', f'{40.0:.1f}°'],
                    ['', ''],
                    ['FLIGHT PERFORMANCE', ''],
                    ['Apogee Altitude (AGL)', f'{apogee_agl:.2f} m ({apogee_agl*3.28084:.0f} ft)'],
                    ['Apogee Time', f'{flight.apogee_time:.2f} s'],
                    ['Maximum Velocity', f'{flight.max_speed:.2f} m/s ({flight.max_speed*2.23694:.1f} mph)'],
                    ['Maximum Mach Number', f'{flight.max_mach_number:.3f}'],
                    ['Maximum Acceleration', f'{np.max(flight.acceleration):.2f} m/s² ({np.max(flight.acceleration)/9.81:.2f} g)'],
                    ['Burnout Time', f'{motor_params["burn_time"]:.2f} s'],
                    ['Burnout Altitude', f'{flight.z(motor_params["burn_time"]) - env.elevation:.2f} m'],
                    ['', ''],
                    ['DESCENT & LANDING', ''],
                    ['Flight Time', f'{flight.t_final:.2f} s'],
                    ['Impact Velocity', f'{flight.impact_velocity:.2f} m/s ({flight.impact_velocity*2.23694:.1f} mph)'],
                    ['Drift Distance', f'{np.sqrt(flight.x_impact**2 + flight.y_impact**2):.2f} m ({np.sqrt(flight.x_impact**2 + flight.y_impact**2)*3.28084:.0f} ft)'],
                    ['Impact Coordinates', f'({flight.x_impact:.2f}, {flight.y_impact:.2f}) m'],
                    ['', ''],
                    ['STABILITY', ''],
                    ['Static Margin (t=0)', f'{rocket.static_margin(0):.3f} calibers'],
                    ['Thrust-to-Weight Ratio', f'{motor.thrust(0.10) / (rocket.total_mass(0.0) * 9.81):.2f}'],
                ]
            except Exception as e:
                summary_data = [['Error generating summary', str(e)]]
            
            # Create table
            table = ax.table(cellText=summary_data,
                           colWidths=[0.5, 0.4],
                           cellLoc='left',
                           loc='center',
                           bbox=[0.1, 0.1, 0.8, 0.8])
            
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # Style the table
            for i, row in enumerate(summary_data):
                if row[1] == '':  # Header rows
                    for j in range(2):
                        cell = table[(i, j)]
                        cell.set_facecolor('#4CAF50')
                        cell.set_text_props(weight='bold', color='white', fontsize=11)
            
            fig.text(0.95, 0.02, 'Page 8', ha='right', fontsize=9, color='gray')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # Add PDF metadata
            d = pdf.infodict()
            d['Title'] = 'NAKURUTU I Flight Simulation Report'
            d['Author'] = 'RocketPy'
            d['Subject'] = 'Rocket Flight Simulation'
            d['Keywords'] = 'RocketPy, Flight Simulation, NAKURUTU'
            d['CreationDate'] = datetime.now()
        
        print(f"  [OK] PDF report saved: {pdf_path}")
        print(f"       Report contains 8 pages with comprehensive flight data")
        return pdf_path
        
    except Exception as e:
        print(f"  [ERROR] Failed to generate PDF report: {e}")
        import traceback
        traceback.print_exc()
        return None


# ---------------------- Main ----------------------
def main():
    print("=" * 60)
    print("NAKURUTU I – ADVANCED ROCKETPY SIMULATION")
    print("• GFS Forecast • Mach-dependent Cd • Original Geometry • Reefing Recovery")
    print("=" * 60)

    # 1) Environment with GFS forecast (safe fallback)
    print("\n[1/4] Environment...")
    env = Environment(
        latitude=-21.938328,
        longitude=-48.949887,
        elevation=493.1,
    )
    # Use today at 12:00 UTC (good for GFS cycles)
    today = datetime.now(timezone.utc)
    env.set_date((today.year, today.month, today.day, 12))
    try:
        env.set_atmospheric_model(type="Forecast", file="GFS")
        print("  [OK] GFS forecast loaded")
    except Exception as e:
        print(f"  [WARN] GFS forecast failed ({e}); using StandardAtmosphere")
        env.set_atmospheric_model(type="StandardAtmosphere")

    # 2) Motor from RSE
    print("\n[2/4] Motor (AeroTech M2000R)...")
    try:
        thrust_curve, mp = parse_rse_file(RSE_PATH)
        print(f"  [OK] {mp['code']} ({mp['mfg']}) | Itot={mp['Itot']:.1f} N·s | "
              f"avg={mp['avgThrust']:.1f} N | burn={mp['burn_time']:.2f} s")
    except Exception as e:
        print(f"  [ERROR] Could not parse RSE: {e}")
        sys.exit(1)

    motor = SolidMotor(
        thrust_source=thrust_curve,
        burn_time=mp["burn_time"],
        dry_mass=mp["dry_mass"],
        dry_inertia=(0.55, 0.55, 0.008),
        center_of_dry_mass_position=mp["length"] / 2.0,
        grains_center_of_mass_position=mp["length"] / 2.0,
        grain_number=5,
        grain_separation=0.003,
        grain_density=1750,
        grain_outer_radius=mp["diameter"] / 2.0 - 0.005,
        grain_initial_inner_radius=0.020,
        grain_initial_height=0.120,
        nozzle_radius=mp["diameter"] / 2.0,
        throat_radius=0.022,
        interpolation_method="linear",
        nozzle_position=0.0,
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )
    print("  [OK] Motor built")

    # 3) Mach-dependent Cd + Original Geometry
    print("\n[3/4] Rocket geometry & aerodynamics...")

    # Mach-dependent Cd (robust)
    try:
        pairs_off = load_mach_cd_curve(POWER_OFF_CSV)
        pairs_on  = load_mach_cd_curve(POWER_ON_CSV)
        power_off_drag = cd_function_from_pairs(pairs_off)
        power_on_drag  = cd_function_from_pairs(pairs_on)
        print(f"  [OK] Cd(Mach) loaded: off={len(pairs_off)} pts | on={len(pairs_on)} pts")
        print(f"      Mach ranges: off {pairs_off[0][0]:.3f}–{pairs_off[-1][0]:.3f} | "
              f"on {pairs_on[0][0]:.3f}–{pairs_on[-1][0]:.3f}")
        # Quick sanity samples
        for m_test in (0.00, 0.10, 0.50, 1.00):
            cd_off = power_off_drag(m_test)
            cd_on  = power_on_drag(m_test)
            print(f"      Cd_off(M={m_test:.2f})={cd_off:.3f} | Cd_on(M={m_test:.2f})={cd_on:.3f}")
    except Exception as e:
        print(f"  [WARN] Mach-Cd load failed ({e}); using constants off=0.45 on=0.42")
        power_off_drag, power_on_drag = 0.45, 0.42

    # Original mass/inertia/CoM from the design
    rocket = Rocket(
        radius=0.078,  # m
        mass=13.194,   # kg (without motor)
        inertia=(12.0, 12.0, 0.07),  # kg*m^2
        power_off_drag=power_off_drag,
        power_on_drag=power_on_drag,
        center_of_mass_without_motor=1.10,  # move slightly forward for stability
        coordinate_system_orientation="nose_to_tail",
    )

    # Original placements (nose_to_tail: 0 at nose tip, + toward tail)
    # Rocket body length approximately 2.60 m
    # Motor nozzle extends 10 cm past the tail, so motor positioned at 2.70 m
    # Fin leading edge positioned near tail section at 2.30 m
    # Nose base at 0.56 m from nose tip
    
    rocket.add_nose(
        length=0.560,
        kind="lvhaack",
        base_radius=0.078,
        position=0.0,
    )

    # Use *sweep angle* to reflect the planform; keep cant small to avoid huge roll torque
    rocket.add_trapezoidal_fins(
        n=4,
        root_chord=0.361,
        tip_chord=0.060,
        span=0.156,
        position=2.30,       # Moved closer to tail
        sweep_angle=43.40,   # degrees (planform geometry)
        cant_angle=0.0,      # keep <= 2.0 if you want intentional slow roll
        airfoil=None,
    )
    
    # Motor positioned so nozzle extends 10 cm outside the rocket tail
    rocket.add_motor(motor, position=2.70)

    # Reefing recovery (our areas): drogue at apogee -> main at 500 m on descent
    print("  [OK] Adding reefing recovery: drogue@apogee -> main@500 m (descent only)")
    drogue_cd_s = 1.012  # reefed (reduced drag)
    rocket.add_parachute(
        name="Reefed",
        cd_s=drogue_cd_s,
        trigger="apogee",
        sampling_rate=100,
        lag=0.3,
        noise=(0, 8.3, 0.5),
    )

    full_cd_s = 2.976  # unreefed (full drag)
    def reefing_cut_trigger(pressure, height, state_vec):
        # Trigger only when below 500 m AGL AND descending (vz < 0)
        try:
            vz = state_vec[5]  # [x, y, z, vx, vy, vz, ...]
            return (height <= 500.0) and (vz < 0.0)
        except Exception:
            return height <= 500.0

    rocket.add_parachute(
        name="Main",
        cd_s=full_cd_s,
        trigger=reefing_cut_trigger,
        sampling_rate=100,
        lag=0.2,
        noise=(0, 8.3, 0.5),
    )

    # Diagnostics: static margin and T/W at 0.10 s
    try:
        sm0 = rocket.static_margin(0)
        print(f"  [OK] Static margin @t0: {sm0:.3f} calibers")
        if sm0 < 1.5:
            print("  [WARN] Static margin < 1.5; consider moving CoM forward or adding nose ballast")
    except Exception as e:
        print(f"  [WARN] Could not compute static margin ({e})")

    try:
        m0 = rocket.total_mass(0.0)
        tw = motor.thrust(0.10) / (m0 * 9.81)
        print(f"  [INFO] T/W @0.10 s: {tw:.2f}")
        if tw < 1.0:
            print("  [WARN] T/W < 1.0 → liftoff unlikely")
    except Exception:
        pass

    print("  [OK] Rocket assembled")

    # 4) Flight
    print("\n[4/4] Flight simulation...")
    try:
        flight = Flight(
            rocket=rocket,
            environment=env,
            rail_length=6.0,
            inclination=87.0,
            heading=40.0,
            max_time=600.0,
            terminate_on_apogee=False,
            verbose=False,
        )
        print("\n" + "=" * 60)
        print("[SUCCESS] SIMULATION COMPLETE!")
        print("=" * 60)

        # Key results (guarded prints)
        print("\nFlight Results:")
        try:
            print(f"  Apogee:        {flight.apogee:.1f} m ({flight.apogee*3.28084:.0f} ft)")
        except Exception:
            print("  Apogee:        N/A")
        try:
            print(f"  Apogee Time:   {flight.apogee_time:.2f} s")
        except Exception:
            print("  Apogee Time:   N/A")
        try:
            print(f"  Max Velocity:  {flight.max_speed:.1f} m/s ({flight.max_speed*2.23694:.1f} mph)")
        except Exception:
            print("  Max Velocity:  N/A")
        try:
            print(f"  Flight Time:   {flight.t_final:.1f} s")
        except Exception:
            print("  Flight Time:   N/A")
        try:
            print(f"  Impact Vel:    {flight.impact_velocity:.2f} m/s")
        except Exception:
            print("  Impact Vel:    N/A")

        # Create simulation folder with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sim_folder = os.path.join(RESULTS_DIR, f"simulation_{timestamp}")
        os.makedirs(sim_folder, exist_ok=True)
        print(f"\nCreated simulation folder: {sim_folder}")
        
        # Save basic outputs
        print("\nGenerating output files...")
        # Save KML via non-deprecated path when available
        kml_path = os.path.join(sim_folder, f"nakurutu_trajectory_{timestamp}.kml")
        export_kml_compat(flight, file_name=kml_path, extrude=True, altitude_mode="absolute")
        print(f"  [OK] KML saved")

        # 3D trajectory image
        png_path = os.path.join(sim_folder, f"nakurutu_trajectory_3d_{timestamp}.png")
        flight.plots.trajectory_3d(filename=png_path)
        print(f"  [OK] 3D trajectory PNG saved\n")
        
        # Generate advanced visualizations and reports
        print("Generating advanced visualizations...")
        
        # Generate interactive 3D HTML
        html_path = generate_interactive_3d_html(flight, sim_folder, timestamp, env)
        
        # Generate motor visualization plots
        motor_repr_path, motor_thrust_path = generate_motor_plots(motor, sim_folder, timestamp)
        
        # Generate rocket schematic
        rocket_schematic_path = generate_rocket_plots(rocket, sim_folder, timestamp)
        
        # Generate atmospheric plots
        atmospheric_path = generate_atmospheric_plots(env, sim_folder, timestamp)
        
        # Generate comprehensive PDF report
        pdf_path = generate_pdf_report(flight, rocket, motor, env, sim_folder, timestamp, mp)
        
        print("  [10/10] All visualizations complete!\n")
        
        # Final summary
        print("=" * 60)
        print("ALL OUTPUTS GENERATED")
        print("=" * 60)
        print(f"\nSimulation Folder: {sim_folder}")
        print("\nGenerated Files:")
        print(f"  1. KML (Google Earth): nakurutu_trajectory_{timestamp}.kml")
        print(f"  2. 3D Trajectory PNG:  nakurutu_trajectory_3d_{timestamp}.png")
        if html_path:
            print(f"  3. Interactive 3D HTML: nakurutu_interactive_3d_{timestamp}.html")
        if motor_repr_path:
            print(f"  4. Motor Representation: motor_representation_{timestamp}.png")
        if motor_thrust_path:
            print(f"  5. Motor Thrust Curve: motor_thrust_curve_{timestamp}.png")
        if rocket_schematic_path:
            print(f"  6. Rocket Schematic: rocket_schematic_{timestamp}.png")
        if atmospheric_path:
            print(f"  7. Atmospheric Conditions: atmospheric_conditions_{timestamp}.png")
        if pdf_path:
            print(f"  8. Comprehensive PDF Report (8 pages): nakurutu_flight_report_{timestamp}.pdf")
        print("\n" + "=" * 60)

    except Exception as e:
        print(f"\n[ERROR] Simulation failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
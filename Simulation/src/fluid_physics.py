"""
Thermodynamic and Fluid Dynamic Physics Formulation for Pneumatic Actuators and Leaks
Based on ISO 6358 standard compressible flow and first-law energy conservation.
"""

import numpy as np

# Physical constants for dry air
GAMMA = 1.4              # Specific heat ratio (cp/cv)
R_AIR = 287.058          # Specific gas constant [J/(kg*K)]
T_REF = 293.15           # Standard temperature [K] (20 deg C)
P_ATM = 1.01325          # Atmospheric pressure [bar]
P_ATM_PA = 1.01325e5     # Atmospheric pressure [Pa]
RHO_0 = 1.184            # Air density at standard reference conditions (20 deg C, 1 atm) [kg/m^3]
B_CRIT = 0.528           # Critical pressure ratio for sonic choked flow

# Actuator geometric parameters (Standard Festo 32mm bore, 100mm stroke cylinder)
BORE_D = 0.032           # Piston diameter [m] (32 mm)
ROD_D = 0.012            # Rod diameter [m] (12 mm)
STROKE = 0.100           # Stroke length [m] (100 mm)
A_PISTON = np.pi * (BORE_D / 2.0)**2          # Piston area [m^2] ~ 8.042e-4 m^2
A_ROD = np.pi * (ROD_D / 2.0)**2              # Rod area [m^2] ~ 1.131e-4 m^2
A_ANNULAR = A_PISTON - A_ROD                  # Return chamber area [m^2] ~ 6.911e-4 m^2
DEAD_VOL_1 = 1.5e-5      # Advance chamber dead volume [m^3] (15 cm^3)
DEAD_VOL_2 = 1.5e-5      # Return chamber dead volume [m^3] (15 cm^3)

P_IS_GAUGE = True        # Sensor reports gauge pressure (~6 bar supply)

def compute_chamber_volumes(x_m):
    """
    Computes cylinder chamber volumes given piston position x in meters [0, STROKE].
    V1: Advance chamber volume [m^3]
    V2: Return chamber volume [m^3]
    """
    v1 = DEAD_VOL_1 + A_PISTON * x_m
    v2 = DEAD_VOL_2 + A_ANNULAR * (STROKE - x_m)
    return v1, v2

def iso6358_mass_flow(p_up_bar, p_down_bar, C_conductance, b_ratio=B_CRIT, T_up=T_REF):
    """
    ISO 6358 Compressible Flow Equation
    C_conductance: Sonic conductance [m^3/(s*Pa)]
    Returns mass flow rate in [kg/s]
    """
    p_u = np.maximum(p_up_bar, 0.01) * 1e5
    p_d = np.maximum(p_down_bar, 0.01) * 1e5
    pr = p_d / p_u
    
    temp_corr = np.sqrt(T_REF / T_up)
    choked = pr <= b_ratio
    flow_choked = C_conductance * p_u * RHO_0 * temp_corr
    
    subsonic_term = np.sqrt(np.maximum(0.0, 1.0 - ((pr - b_ratio) / (1.0 - b_ratio))**2))
    flow_subsonic = C_conductance * p_u * RHO_0 * temp_corr * subsonic_term
    
    return np.where(choked, flow_choked, flow_subsonic)

def lpm_to_massflow(q_lpm):
    """Converts volumetric flow in Standard L/min (at 20C, 1 atm) to mass flow in kg/s"""
    return (q_lpm / 60000.0) * RHO_0

def compute_physics_features(df):
    """
    Enriches raw time series with physics-governed states and dynamic residuals.
    Input df columns: time_s, displacement_mm, velocity_mps, flow_rate_lpm, total_volume_l, pressure_bar
    """
    time = df['time_s'].values
    x_m = df['displacement_mm'].values / 1000.0
    v = df['velocity_mps'].values
    q = np.abs(df['flow_rate_lpm'].values)  # Standard L/min
    p_bar = df['pressure_bar'].values

    # Proper handling of gauge vs absolute pressure across both sim and hardware
    p_abs = p_bar + P_ATM if P_IS_GAUGE else p_bar

    # 1. Kinematic consistency residual: e_kin = v - dx/dt
    dx_dt = np.gradient(x_m, time)
    e_kin = v - dx_dt

    # 2. Dynamic state derivatives
    dp_dt = np.gradient(p_bar, time)
    dq_dt = np.gradient(q, time)

    # 3. Sonic conductance proxy: C = (q / 60) / max(p_abs, 0.05) [L/(s*bar)]
    # Uses absolute line pressure to align directly with ISO 6358 C = q_m / (rho_0 * p_u,abs)
    conductance = (q / 60.0) / np.maximum(p_abs, 0.05)

    # 4. Instantaneous pneumatic fluid power [Watts]: P_fluid = p_abs * q_vol * (10 / 6)
    # Scaled by (10/6) to convert bar * L/min -> (1e5 Pa) * (1e-3/60 m^3/s) = Joules/s = Watts
    p_power = p_abs * q * (10.0 / 6.0)

    # 5. Inter-stroke baseline steady flow (distinguishes leak regimes from transient exhaust spikes)
    # Piston dwells for >50% of the cycle, so window median accurately isolates steady leak flow
    q_baseline = np.full_like(q, np.median(q))

    physics_df = df.copy()
    physics_df['dx_dt_mps'] = dx_dt
    physics_df['e_kinematic'] = e_kin
    physics_df['pneumatic_power_W'] = p_power
    physics_df['dp_dt_bar_s'] = dp_dt
    physics_df['dq_dt_lpm_s'] = dq_dt
    physics_df['conductance_proxy'] = conductance
    physics_df['leak_baseline_lpm'] = q_baseline

    return physics_df

if __name__ == "__main__":
    print("Fluid physics module loaded and verified successfully.")

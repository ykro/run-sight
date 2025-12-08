import os
import time
import json
import numpy as np
import pandas as pd
import fitdecode
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.cloud import storage, firestore

# Load environment variables
load_dotenv()

GEMINI_MODEL = 'gemini-3-pro-preview'

# --- GCP CONFIG ---
GCS_BUCKET_NAME = os.getenv('GCS_BUCKET_NAME')
GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID')
FIRESTORE_DB_NAME = os.getenv('FIRESTORE_DB_NAME', '(default)')
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

class AnalysisError(Exception):
    pass

def setup_client(api_key: Optional[str] = None):
    key = api_key or os.getenv('GEMINI_API_KEY')
    if not key:
        raise AnalysisError("GEMINI_API_KEY not found in environment variables.")
    return genai.Client(api_key=key)

SYSTEM_PROMPT_PATH = Path('analysis.md')

def load_system_prompt():
    if not SYSTEM_PROMPT_PATH.exists():
        return "Analyze this run."
    return SYSTEM_PROMPT_PATH.read_text(encoding='utf-8')

# --- CLOUD STORAGE FUNCTIONS ---

def upload_to_gcs(file_path: Path, bucket_name: str = None, destination_blob_name: str = None) -> str:
    """Uploads a file to the bucket."""
    bucket_name = bucket_name or GCS_BUCKET_NAME
    if not bucket_name:
        print("Warning: GCS_BUCKET_NAME not set. Skipping upload.")
        return None

    if destination_blob_name is None:
        destination_blob_name = file_path.name

    try:
        # Client will automatically look for GOOGLE_APPLICATION_CREDENTIALS
        storage_client = storage.Client(project=GCP_PROJECT_ID)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(str(file_path))
        
        return f"gs://{bucket_name}/{destination_blob_name}"
    except Exception as e:
        print(f"Failed to upload to GCS: {e}")
        return None

def _convert_numpy_types(obj):
    """Recursively converts numpy types to native Python types for Firestore compatibility."""
    if isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    return obj

def save_to_firestore(collection: str, document_id: str, data: Dict):
    """Saves data to a Firestore document."""
    try:
        # Convert numpy types to native Python types
        clean_data = _convert_numpy_types(data)
        
        # Client will automatically look for GOOGLE_APPLICATION_CREDENTIALS
        db = firestore.Client(project=GCP_PROJECT_ID, database=FIRESTORE_DB_NAME)
        doc_ref = db.collection(collection).document(document_id)
        doc_ref.set(clean_data)
        return True
    except Exception as e:
        print(f"Failed to save to Firestore: {e}")
        return False

# --- CORE LOGIC ---


def parse_fit_file(file_path: Path) -> tuple[pd.DataFrame, Dict]:
    """Parses a .fit file (or a .zip containing one) and returns a DataFrame of records and a metadata dict."""
    import zipfile
    
    data_points = []
    metadata = {
        'filename': file_path.name,
        'processed_at': datetime.now().isoformat(),
        'session_data': {}
    }

    try:
        # Check if it's a zip file
        if file_path.suffix.lower() == '.zip':
            with zipfile.ZipFile(file_path, 'r') as z:
                # Find first .fit file
                fit_files = [f for f in z.namelist() if f.lower().endswith('.fit')]
                if not fit_files:
                    raise AnalysisError("No .fit file found inside the zip archive.")
                
                # Extract to bytes
                with z.open(fit_files[0]) as f:
                    # FitReader can take a file-like object (bytes)
                    with fitdecode.FitReader(f) as fit_file:
                        _parse_fit_frames(fit_file, data_points, metadata)
                        
        else:
            # Standard .fit file
            with fitdecode.FitReader(file_path) as fit_file:
                _parse_fit_frames(fit_file, data_points, metadata)

    except Exception as e:
        raise AnalysisError(f"Failed to parse file: {e}")

    if not data_points:
        raise AnalysisError("No 'record' frames found in .fit file.")

    df = pd.DataFrame(data_points)
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df, metadata

def _parse_fit_frames(fit_file, data_points, metadata):
    """Helper to iterate frames from an open FitReader"""
    for frame in fit_file:
        if frame.frame_type == fitdecode.FIT_FRAME_DATA:
            if frame.name == 'record':
                point = {}
                for field in frame.fields:
                    if field.value is not None:
                        point[field.name] = field.value
                data_points.append(point)
            elif frame.name == 'session':
                for field in frame.fields:
                    if field.value is not None:
                        metadata['session_data'][field.name] = field.value



def calculate_metrics(df: pd.DataFrame, metadata: Dict, user_intent: Dict = None) -> Dict:
    """Calculates forensic metrics including Time Core, GAP, Zones, and Peaks."""
    metrics = {}
    
    if user_intent:
        metrics['user_intent'] = user_intent
    
    # --- 1. Data Hygiene (Preprocessing) ---
    # Cadence Cleaning: Replace 0 with NaN for averages (avoid underestimation)
    if 'cadence' in df.columns:
        df['cadence_clean'] = df['cadence'].replace(0, np.nan)
    else:
        df['cadence_clean'] = np.nan
    
    # ... (existing Preprocessing)
    # Altitude Smoothing (SMA 10s)
    if 'enhanced_altitude' in df.columns:
        df['alt_smooth'] = df['enhanced_altitude'].rolling(window=10, min_periods=1).mean()
    else:
        df['alt_smooth'] = 0

    # Basic Deltas
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds().fillna(0)
    df['dist_diff'] = df['distance'].diff().fillna(0)
    df['alt_diff'] = df['alt_smooth'].diff().fillna(0)
    
    # Calculate Grade (Rise/Run)
    # Avoid division by zero
    df['grade'] = np.where(df['dist_diff'] > 0, df['alt_diff'] / df['dist_diff'], 0)
    df['grade_pct'] = df['grade'] * 100

    # Ensure speed exists (enhanced_speed is m/s)
    if 'enhanced_speed' not in df.columns:
        df['enhanced_speed'] = np.where(df['time_diff'] > 0, df['dist_diff'] / df['time_diff'], 0)
        
    # ... (existing Time Core)
    # --- 2. Time Core (State Classification) ---
    conditions = [
        (df['time_diff'] > 5), # Pause (gap)
        (df['enhanced_speed'] < 0.5), # Idle
        (df['cadence_clean'] < 130), # Walk/Hike (Power Hiking)
        (df['cadence_clean'] >= 130) # Run
    ]
    choices = ['Pause', 'Idle', 'Walk', 'Run']
    df['state'] = np.select(conditions, choices, default='Idle')
    
    # Time Analysis Aggregation
    time_analysis = df.groupby('state')['time_diff'].sum().to_dict()
    total_duration = df['time_diff'].sum()
    metrics['time_analysis'] = {
        'total_duration': round(total_duration, 1),
        'check': time_analysis
    }
    
    # ... (existing Advanced Metrics A, B, C, D, E) need to be preserved... 
    # I am replacing start of function, need to be careful not to delete sections not included in replacement range.
    # The user asked to modify specific "Legacy" logic which comes LATER in the file.
    # Wait, Step 235 replaced entire function body in thought process but I should target specific lines.
    # I will split this into MULTIPLE replacements to avoid accidentally deleting the middle if I target too wide.
    # OR I target the legacy block specifically for the bugs, and the signature for user_intent.
    
    # REPLACEMENT 1: Signature & user_intent (Lines 140-145 approx)
    
    # REPLACEMENT 2: Uphill Logic (Lines ~300)
    
    # REPLACEMENT 3: Cardiac Cost (Lines ~330)
    
    # I will cancel this single large replacement and use 3 specific ones via multi_replace.

    # Cadence Cleaning: Replace 0 with NaN for averages (avoid underestimation)
    if 'cadence' in df.columns:
        df['cadence_clean'] = df['cadence'].replace(0, np.nan)
    else:
        df['cadence_clean'] = np.nan

    # Altitude Smoothing (SMA 10s)
    if 'enhanced_altitude' in df.columns:
        df['alt_smooth'] = df['enhanced_altitude'].rolling(window=10, min_periods=1).mean()
    else:
        df['alt_smooth'] = 0

    # Basic Deltas
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds().fillna(0)
    df['dist_diff'] = df['distance'].diff().fillna(0)
    df['alt_diff'] = df['alt_smooth'].diff().fillna(0)
    
    # Calculate Grade (Rise/Run)
    # Avoid division by zero
    df['grade'] = np.where(df['dist_diff'] > 0, df['alt_diff'] / df['dist_diff'], 0)
    df['grade_pct'] = df['grade'] * 100

    # Ensure speed exists (enhanced_speed is m/s)
    # --- 2. Time Core (State Classification) ---
    # Logic:
    # Pause: Gap in recording > 5s
    # Run: Cadence >= 125 spm OR Running Dynamics Present
    # Idle: Speed < 0.3 m/s AND Cadence < 30 (Truly stopped)
    # Walk: Everything else (moving slow with cadence, or moving fast without run dynamics)
    
    # Check for Running Dynamics presence (proxy for Run)
    has_run_dyn = pd.Series(False, index=df.index)
    if 'stance_time' in df.columns:
        has_run_dyn = df['stance_time'].notna()
    
    conditions = [
        (df['time_diff'] > 5), # Pause (gap)
        (df['cadence_clean'] >= 125) | (has_run_dyn), # Run (High cadence or has GCT)
        (df['enhanced_speed'] < 0.4) & (df['cadence_clean'].fillna(0) < 30) & (df['dist_diff'] < 0.5) # Idle (Stopped)
    ]
    choices = ['Pause', 'Run', 'Idle']
    # Default to Walk (captures active recovery, hiking, power hiking)
    df['state'] = np.select(conditions, choices, default='Walk')
    
    # Time Analysis Aggregation
    time_analysis = df.groupby('state')['time_diff'].sum().to_dict()
    total_duration = df['time_diff'].sum()
    metrics['time_analysis'] = {
        'total_duration': round(total_duration, 1),
        'check': time_analysis
    }
    
    # --- GLOBAL FILTER FOR METRICS ---
    # Use active segments for performance calculations (Efficiency, Durability, VI, etc.)
    # Exclude Idle or segments where speed is essentially zero but state wasn't caught as Idle
    active_mask = (df['state'] != 'Idle') & (df['enhanced_speed'] > 0.1)
    active_df = df[active_mask].copy()

    # --- 3. Advanced Metrics Implementation ---

    # A. GAP (Grade Adjusted Pace) - Minetti
    # Cost (J/kg/m) = 155.4*i^5 - 30.4*i^4 - 43.3*i^3 + 46.3*i^2 + 19.5*i + 3.6
    # Ratio = Cost(i) / Cost(0) -> Cost(0) = 3.6
    # GAP Speed = Speed * Ratio (Actually, Speed * (Cost(i)/3.6) ? 
    # Wait, Energy Cost is higher uphill. So to traverse same distance takes MORE energy. 
    # Equivalent Flat Speed (GAP) should be HIGHER than real speed uphill.
    # GAP = Speed * (EnergyCost_Gradient / EnergyCost_Flat)
    
    i = df['grade'].clip(-0.45, 0.45) # Clip to valid Minetti range
    energy_cost = 155.4*(i**5) - 30.4*(i**4) - 43.3*(i**3) + 46.3*(i**2) + 19.5*i + 3.6
    df['gap_speed'] = df['enhanced_speed'] * (energy_cost / 3.6)
    
    # B. Peak Performance (Rolling Max)
    # Resample to 1s freq to perform accurate rolling if data is irregular? 
    # FIT files are usually 1s. We'll assume roughly 1s or use rolling on index if 1s.
    # If not 1s, this is an approximation.
    
    # B. Peak Performance (Rolling Max with Location)
    peaks = {}
    start_ts = df['timestamp'].iloc[0]
    
    for window_s in [60, 300, 1200, 3600]: # 1m, 5m, 20m, 60m
        label = f"{window_s//60}m"
        if len(df) > window_s:
            if 'power' in df.columns:
                rolling_power = df['power'].rolling(window=window_s).mean()
                peak_val = rolling_power.max()
                if pd.notna(peak_val):
                    peak_idx = rolling_power.idxmax()
                    time_from_start = (df.loc[peak_idx, 'timestamp'] - start_ts).total_seconds()
                    km_loc = df.loc[peak_idx, 'distance'] / 1000.0
                    peaks[f'peak_power_{label}'] = {
                        "value": int(peak_val),
                        "time_from_start_sec": int(time_from_start),
                        "km_loc": round(km_loc, 1)
                    }
            
            rolling_gap = df['gap_speed'].rolling(window=window_s).mean()
            peak_gap_val = rolling_gap.max()
            if pd.notna(peak_gap_val):
                peak_gap_idx = rolling_gap.idxmax()
                time_from_start = (df.loc[peak_gap_idx, 'timestamp'] - start_ts).total_seconds()
                km_loc = df.loc[peak_gap_idx, 'distance'] / 1000.0
                peaks[f'peak_gap_{label}'] = {
                    "value": round(peak_gap_val, 2),
                    "time_from_start_sec": int(time_from_start),
                    "km_loc": round(km_loc, 1)
                }
            
    metrics['peaks'] = peaks

    # C. Technical Analysis
    tech = {}
    # Use all available data for Tech Metrics, don't strict filter by state (since state is inferred)
    if 'stance_time' in df.columns: # GCT
       val = df['stance_time'].mean()
       tech['avg_gct_ms'] = int(val) if pd.notna(val) else None
    if 'vertical_ratio' in df.columns:
       val = df['vertical_ratio'].mean()
       tech['avg_vert_ratio'] = round(val, 2) if pd.notna(val) else None
    
    # Stability: Calculate on 'Run' segments (now defined more robustly)
    run_df = df[df['state']=='Run']
    if not run_df.empty:
        if 'cadence_clean' in run_df.columns:
             tech['stability_cadence_std'] = round(run_df['cadence_clean'].std(), 1)
        tech['stability_pace_std'] = round(run_df['enhanced_speed'].std(), 2)
        
    metrics['technical'] = tech

    # --- Mechanics by Terrain (AVG Active) ---
    # Define Terrain: Uphill (> 2%), Downhill (< -2%), Flat
    # Use active_df for clean mechanics
    mechanics_terrain = {}
    if not active_df.empty and 'grade_pct' in active_df.columns:
        active_df['terrain_cat'] = pd.cut(active_df['grade_pct'], bins=[-np.inf, -2, 2, np.inf], labels=['downhill', 'flat', 'uphill'])
        
        for t_cat in ['uphill', 'downhill', 'flat']:
            t_df = active_df[active_df['terrain_cat'] == t_cat]
            if not t_df.empty:
                mech_data = {}
                if 'stance_time' in t_df.columns:
                     val = t_df['stance_time'].mean()
                     mech_data['avg_gct'] = int(val) if pd.notna(val) else None
                if 'vertical_ratio' in t_df.columns:
                     val = t_df['vertical_ratio'].mean()
                     mech_data['avg_vert_ratio'] = round(val, 2) if pd.notna(val) else None
                mechanics_terrain[t_cat] = mech_data
    
    metrics['mechanics_by_terrain'] = mechanics_terrain
    
    # D. Efficiency (RE) - Speed / Power
    # Split by terrain: Uphill (>2%), Flat (-2% to 2%), Downhill (<-2%)
    if 'power' in df.columns:
        df['terrain'] = pd.cut(df['grade_pct'], bins=[-np.inf, -2, 2, np.inf], labels=['Downhill', 'Flat', 'Uphill'])
        # Filter for running only for RE
        re_df = df[(df['state']=='Run') & (df['power'] > 0)]
        re_stats = re_df.groupby('terrain', observed=False).apply(
            lambda x: (x['enhanced_speed'] / x['power']).mean() if not x.empty else 0,
            include_groups=False
        ).to_dict()
        # Clean keys
        metrics['running_efficiency'] = {k: round(v, 4) for k, v in re_stats.items() if pd.notna(v)}

    # E. Load & Fluids
    sweat_est = 0
    if 'power' in df.columns:
        # Metabolic Cost Estimation
        # Mechanical Work (kJ) = Avg Power (W) * Duration (s) / 1000
        # Human Gross Efficiency ~21-24%.
        # Converstion: 1 kJ mechanical work ~= 1 kCal metabolic energy expenditure (Rule of Thumb: 4.184 J/cal * 0.24 eff ~ 1)
        avg_watts = df['power'].mean()
        energy_kcal = (avg_watts * total_duration) / 1000
        # Sweat Rate: varies by temp/humidity. Approx 0.8 - 1.2 L / 1000 kCal.
        # Conservative Estimate: 0.7 L / 1000 kCal (User said previous was too hard)
        sweat_est = (energy_kcal / 1000) * 0.7
    else:
        # Time Based Fallback (0.5 L/hr - Conservative)
        sweat_est = (total_duration / 3600) * 0.5
        
    metrics['physiological'] = {
        'sweat_loss_liter_est': round(sweat_est, 2)
    }

    # --- TRIMP (Edwards) & Impact Load ---
    # TRIMP = Sum(Duration_mins * Zone_Weight)
    # Zone Weights: Z1=1, Z2=2, Z3=3, Z4=4, Z5=5
    trimp = 0
    if 'heart_rate' in df.columns:
         # Calculate zones for every record to be precise
         # We already have 'hr_zone' in df from block 4 logic availability check? 
         # Block 4 is later. We need to move logic or duplicate local calc if not present yet.
         # Actually Block 4 (Zones) is distinctly later in the file (Line 505).
         # Let's simple calc here on active_df
         sys_max_hr = 190
         if user_intent and user_intent.get('max_hr'):
             sys_max_hr = int(user_intent['max_hr'])
         elif 'heart_rate' in df.columns:
             mghr = df['heart_rate'].max()
             sys_max_hr = mghr if pd.notna(mghr) else 190
         
         # Vectorized TRIMP
         # Edwards: 50-60% MaxHR = 1, 60-70=2, 70-80=3, 80-90=4, 90-100=5
         # Use active_df for load accumulation? TRIMP technically counts even rest, but user asked for "Active" focus? 
         # TRIMP is physiological load. If HR is high while stopped, it IS load. 
         # But user said "calculate metrics... without taking into account idle". 
         # I will use active_df for TRIMP to consistent with user request "metrics... without data reported as idle"
         if not active_df.empty:
             hrs = active_df['heart_rate']
             z1 = (hrs >= 0.5*sys_max_hr) & (hrs < 0.6*sys_max_hr)
             z2 = (hrs >= 0.6*sys_max_hr) & (hrs < 0.7*sys_max_hr)
             z3 = (hrs >= 0.7*sys_max_hr) & (hrs < 0.8*sys_max_hr)
             z4 = (hrs >= 0.8*sys_max_hr) & (hrs < 0.9*sys_max_hr)
             z5 = (hrs >= 0.9*sys_max_hr)
             
             # Approx 1 sec per record? Use time_diff
             sample_weights = np.zeros(len(active_df))
             sample_weights[z1] = 1
             sample_weights[z2] = 2
             sample_weights[z3] = 3
             sample_weights[z4] = 4
             sample_weights[z5] = 5
             
             # Weighted minutes
             trimp = np.sum((active_df['time_diff'] / 60.0) * sample_weights)
    
    metrics['trimp'] = int(trimp)

    # Impact Load
    # Cumulative Sum of (Step * VertOsc) approx
    impact_load = 0
    if not active_df.empty and 'vertical_oscillation' in active_df.columns:
         # Vert Osc is mm. 
         # If exact step count per record not available, use Cadence (spm) / 60 * duration * VertOsc
         # Impact ~ Steps * Amplitude
         steps_est = (active_df['cadence_clean'].fillna(0) / 60.0) * active_df['time_diff']
         impact_mm = steps_est * active_df['vertical_oscillation'].fillna(0)
         # Arbitrary unit? Or Meters? 
         # Let's sum meters of vertical displacement
         impact_load = impact_mm.sum() / 1000.0 # Meters of vertical checking
    
    metrics['impact_load'] = round(impact_load, 1)
    
    # --- LEGACY METRICS RESTORATION ---
    
    # 1. Context Detection (Trail vs Road)
    total_ascent = df[df['alt_diff'] > 0]['alt_diff'].sum()
    total_dist_km = df['distance'].max() / 1000.0 if df['distance'].max() > 0 else 0.001
    ascent_ratio = total_ascent / total_dist_km
    metrics['ascent_ratio'] = round(ascent_ratio, 2)
    metrics['context'] = 'Trail' if ascent_ratio > 25 else 'Road'

    # 2. Continuity Index (Aligned with Time Analysis)
    total_elapsed_s = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds()
    timer_time_s = df['time_diff'].sum()
    # Logistical Pause: Time the watch was stopped (Elapsed - Recorded)
    pause_time_s = max(0, total_elapsed_s - timer_time_s)
    
    # Idle Time: Time the watch was running but user was stopped (from Time Core)
    idle_time_s = metrics['time_analysis']['check'].get('Idle', 0)
    
    # metrics['total_time_sec'] and 'timer_time_sec' removed as redundant (see elapsed_time/timer_time at end)
    metrics['stop_time_sec'] = round(idle_time_s, 2) # Coherent with Time Analysis
    metrics['logistical_pause_time_sec'] = round(pause_time_s, 2) # New Field: Watch Paused
    
    # Ratio: % of recorded time spent Idle
    metrics['stop_ratio_pct'] = round((idle_time_s / timer_time_s * 100), 2) if timer_time_s > 0 else 0

    # 3. Uphill Biomechanics
    grade_threshold = 10.0 if metrics['context'] == 'Trail' else 2.0
    uphill_mask = df['grade_pct'] > grade_threshold
    uphill_df = df[uphill_mask].copy()
    
    metrics['over_striding_flag'] = False
    metrics['mech_inefficiency_flag'] = False

    if not uphill_df.empty:
        if 'cadence' in uphill_df.columns:
            # Fix: Exclude 0 cadence (stops)
            valid_cadence = uphill_df[uphill_df['cadence'] > 0]['cadence']
            avg_uphill_cadence = valid_cadence.mean() if not valid_cadence.empty else 0
            avg_uphill_speed = uphill_df['enhanced_speed'].mean()
            
            # Stride calc (Assume RPM if < 120)
            if avg_uphill_cadence > 0:
                is_rpm = avg_uphill_cadence < 120
                spm = avg_uphill_cadence * 2 if is_rpm else avg_uphill_cadence
                stride_m = avg_uphill_speed * 60 / spm
            else:
                stride_m = 0
            
            # Use raw fit value for the flag check as originally requested ("Cadence < 60")
            raw_cadence = avg_uphill_cadence
            
            metrics['uphill_avg_cadence'] = round(raw_cadence, 1)
            metrics['uphill_avg_stride_m'] = round(stride_m, 2)

            if stride_m > 0.65 and raw_cadence < 60:
                metrics['over_striding_flag'] = True
        
        uphill_time_s = uphill_df['time_diff'].sum()
        uphill_ascent_m = uphill_df['alt_diff'].sum()
        vam = (uphill_ascent_m / uphill_time_s) * 3600 if uphill_time_s > 0 else 0
        metrics['uphill_vam_avg'] = round(vam, 1)
        
        # Max VAM over 20-minute rolling window
        # Calculate instantaneous VAM per record (m/h)
        uphill_df['instant_vam'] = np.where(
            uphill_df['time_diff'] > 0,
            (uphill_df['alt_diff'] / uphill_df['time_diff']) * 3600,
            0
        )
        
        # 20-minute rolling (1200 seconds ~ 1200 records at 1Hz)
        window_20m = min(1200, len(uphill_df))
        if window_20m > 60:  # At least 1 minute of data
            rolling_vam = uphill_df['instant_vam'].rolling(window=window_20m, min_periods=60).mean()
            max_vam_20m = rolling_vam.max()
            metrics['uphill_vam_max_20m'] = round(max_vam_20m, 1) if pd.notna(max_vam_20m) else None
        else:
            metrics['uphill_vam_max_20m'] = None
        
        if 'heart_rate' in uphill_df.columns:
            avg_uphill_hr = uphill_df['heart_rate'].mean()
            if vam < 600 and avg_uphill_hr > 165:
                 metrics['mech_inefficiency_flag'] = True
                 
    # 4. Cardiac Cost (Revised: Beats during movement / Total Moving Distance)
    if 'heart_rate' in df.columns:
        # Beats = sum(HR(bpm) * time(min))
        # Use moving mask roughly defined earlier? Recalculating for precision
        is_moving = df['enhanced_speed'] > 0.5 # Moving
        moving_df = df[is_moving]
        
        if not moving_df.empty:
            total_beats_moving = (moving_df['heart_rate'] * (moving_df['time_diff'] / 60)).sum()
            moving_dist_km = moving_df['dist_diff'].sum() / 1000.0
            
            metrics['cardiac_cost'] = round(total_beats_moving / moving_dist_km, 1) if moving_dist_km > 0 else 0
        else:
             metrics['cardiac_cost'] = 0

    # 5. Durability Index (Legacy Declinaton)
    if 'heart_rate' in df.columns and not active_df.empty:
        # Use active_df to determine segments
        # Recalculate elapsed min for the active subset or keep original time? 
        # Typically Durability is about fatigue over TIME. So we should use the timestamp from active_df.
        active_df['elapsed_min'] = (active_df['timestamp'] - active_df['timestamp'].iloc[0]).dt.total_seconds() / 60
        active_df['segment_10m'] = (active_df['elapsed_min'] // 10).astype(int)
        
        segments = active_df.groupby('segment_10m').agg({'enhanced_speed': 'mean', 'heart_rate': 'mean'})
        # Filter valid
        segments = segments[segments['enhanced_speed'] > 0]
        segments['EF_raw'] = segments['enhanced_speed'] / segments['heart_rate']
        
        if len(segments) >= 2:
            first_ef = segments.iloc[0]['EF_raw']
            last_ef = segments.iloc[-1]['EF_raw']
            decline_pct = ((first_ef - last_ef) / first_ef) * 100 if first_ef > 0 else 0
            metrics['durability_index_decline_pct'] = round(decline_pct, 1)

    # 6. Negative Split
    if not active_df.empty:
        half_dist = active_df['distance'].max() / 2
        first_half = active_df[active_df['distance'] <= half_dist]
        second_half = active_df[active_df['distance'] > half_dist]
        
        speed1 = first_half['enhanced_speed'].mean()
        speed2 = second_half['enhanced_speed'].mean()
        metrics['negative_split_index'] = round(speed2 / speed1, 3) if speed1 > 0 else 0
    else:
        metrics['negative_split_index'] = 0

    # 7. Downhill Agility
    if 'grade_pct' in df.columns:
        downhill_mask = df['grade_pct'] < -10
        downhill_df = df[downhill_mask]
        
        if not downhill_df.empty:
            avg_descent_speed = downhill_df['enhanced_speed'].mean()
            descended_m = downhill_df['alt_diff'].sum()
            time_s = downhill_df['time_diff'].sum()
            vert_descent_speed = (abs(descended_m) / time_s) * 3600 if time_s > 0 else 0
            
            metrics['avg_descent_speed_ms'] = round(avg_descent_speed, 2)
            metrics['vert_descent_speed_mh'] = round(vert_descent_speed, 1)

    # 8. Watts Per Beat
    if 'power' in df.columns and 'heart_rate' in df.columns:
        power_df = df[df['power'] > 0]
        if not power_df.empty:
            metrics['watts_per_beat'] = round(power_df['power'].mean() / power_df['heart_rate'].mean(), 2)

    # 9. Pace Variability Index (VI)
    # Use active_df to calculate NG and Average
    if 'grade' in active_df.columns and 'enhanced_speed' in active_df.columns and not active_df.empty:
        g = active_df['grade'].clip(-0.30, 0.30)
        cost = 155.4 * g**5 - 30.4 * g**4 - 43.3 * g**3 + 46.3 * g**2 + 19.5 * g + 3.6
        gap_speed = active_df['enhanced_speed'] * (cost / 3.6)
        
        rolling_gap = gap_speed.rolling(window=30, min_periods=1).mean()
        ngp_speed = np.mean(rolling_gap ** 4) ** 0.25
        
        avg_gap_speed_raw = gap_speed.mean()
        
        if avg_gap_speed_raw > 0:
            metrics['pace_variability_index'] = round(ngp_speed / avg_gap_speed_raw, 3)

    # --- 4. Zones & Decoupling ---
    max_hr = 190
    if user_intent and user_intent.get('max_hr'):
         max_hr = int(user_intent['max_hr'])
    elif 'heart_rate' in df.columns:
         max_val = df['heart_rate'].max()
         max_hr = max_val if pd.notna(max_val) else 190
    
    # Zones (5 zones)
    if 'heart_rate' in df.columns:
        df['hr_zone'] = pd.cut(df['heart_rate'], bins=[0, 0.6*max_hr, 0.7*max_hr, 0.8*max_hr, 0.9*max_hr, 1000], labels=['Z1', 'Z2', 'Z3', 'Z4', 'Z5'])
        hr_dist = df.groupby('hr_zone', observed=False)['time_diff'].sum().to_dict()
        metrics['hr_zones'] = {k: round(v, 1) for k, v in hr_dist.items()}
        
        if 'power' in df.columns:
            # Filter for active segments only (eliminate idle/coasting)
            power_df = df[df['power'] > 0]
            
            if not power_df.empty:
                mid = len(power_df) // 2
                first = power_df.iloc[:mid]
                second = power_df.iloc[mid:]
                
                ef1 = (first['power'] / first['heart_rate']).mean()
                ef2 = (second['power'] / second['heart_rate']).mean()
            else:
                ef1 = None
                ef2 = None
            
            if pd.notna(ef1) and ef1 > 0 and pd.notna(ef2):
                decoupling = ((ef1 - ef2) / ef1) * 100
                metrics['decoupling_pct'] = round(decoupling, 2)
            else:
                 metrics['decoupling_pct'] = None
        
        # Metabolic Health Object
        metrics['metabolic_health'] = {
            "aerobic_decoupling_pct": metrics.get('decoupling_pct'),
            "efficiency_factor_first_half": round(ef1, 2) if ef1 else None,
            "efficiency_factor_second_half": round(ef2, 2) if ef2 else None
        }

    # --- 5. Laps & Events (Payload Structure) ---
    
    # Arrays for simple 1km splits
    # Add cumulative distance check
    df['km_split'] = (df['distance'] // 1000).astype(int)
    
    laps = []
    for km, group in df.groupby('km_split'):
        avg_hr_val = group['heart_rate'].mean() if 'heart_rate' in group else None
        
        # Determine dominant state with context
        state_dom = group['state'].mode()[0] if not group.empty else 'N/A'
        avg_grade = group['grade_pct'].mean()
        
        # Refine Walk -> Power Hiking if Uphill (> 2% grade)
        if state_dom == 'Walk' and avg_grade > 2.0:
            state_dom = 'Power Hiking'
            
        # Filter group for averages (exclude stops within the lap)
        active_group = group[(group['state'] != 'Idle') & (group['enhanced_speed'] > 0.1)]
        
        if active_group.empty:
            avg_gap = 0
            avg_hr = 0
        else:
             avg_gap = round(active_group['gap_speed'].mean(), 2) if pd.notna(active_group['gap_speed'].mean()) else 0
             avg_hr = int(active_group['heart_rate'].mean()) if 'heart_rate' in active_group and pd.notna(active_group['heart_rate'].mean()) else 0

        # Environment Trend
        env_data = {}
        if 'temperature' in group.columns:
             temps = group['temperature'].dropna()
             if not temps.empty:
                 t_start = temps.iloc[0]
                 t_end = temps.iloc[-1]
                 t_avg = temps.mean()
                 diff = t_end - t_start
                 if diff < -2:
                     trend = "Cooling"
                 elif diff > 2:
                     trend = "Heating"
                 else:
                     trend = "Stable"
                 
                 env_data = {
                     "temp_avg": round(t_avg, 1),
                     "temp_start": t_start,
                     "temp_end": t_end,
                     "trend": trend
                 }

        lap_data = {
            'km': int(km + 1),
            'avg_gap': avg_gap,
            'avg_hr': avg_hr,
            'elev_gain': round(group[group['alt_diff'] > 0]['alt_diff'].sum(), 1),
            'elev_loss': round(abs(group[group['alt_diff'] < 0]['alt_diff'].sum()), 1),
            'state_dom': state_dom,
            'environment': env_data
        }
        laps.append(lap_data)
        
    metrics['laps'] = laps
    
    # Events (Anomalies)
    events = []
    # 1. Long Pause > 5 min
    # We used time_diff > 5s for pause state, but let's check magnitude
    long_pauses = df[df['time_diff'] > 300]
    for idx, row in long_pauses.iterrows():
         events.append(f"Long Pause ({int(row['time_diff']/60)}m) at {int(row['distance'])}m")
         
    # 2. HR Spikes
    if 'heart_rate' in df.columns:
        if df['heart_rate'].max() > 200:
             events.append(f"High Heart Rate Detected ({df['heart_rate'].max()} bpm)")

    metrics['events'] = events

    # Summary Metrics (Requested)
    if 'heart_rate' in df.columns:
        metrics['max_heart_rate'] = int(df['heart_rate'].max())
        metrics['avg_heart_rate'] = int(df['heart_rate'].mean())
    else:
        metrics['max_heart_rate'] = None
        metrics['avg_heart_rate'] = None

    metrics['total_dist_km'] = round(df['distance'].max() / 1000, 2)
    metrics['elevation_gain_m'] = round(df[df['alt_diff'] > 0]['alt_diff'].sum(), 1)
    metrics['elevation_loss_m'] = round(abs(df[df['alt_diff'] < 0]['alt_diff'].sum()), 1)
    
    # Times
    # Approx elapsed from timestamps if available
    try:
        metrics['elapsed_time'] = round((df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds(), 1)
    except:
        metrics['elapsed_time'] = None
        
    metrics['timer_time'] = round(df['time_diff'].sum(), 1)

    # --- Environment Summary ---
    env_summary = {}
    if 'temperature' in df.columns:
        temps = df['temperature'].dropna()
        if not temps.empty:
            max_temp = temps.max()
            min_temp = temps.min()
            
            # Determine condition
            is_heat = max_temp > 24
            is_cold = min_temp < 16
            
            if is_heat and is_cold:
                condition = "Mixed Heat/Cold"
            elif is_heat:
                condition = "Heat"
            elif is_cold:
                condition = "Cold"
            else:
                condition = "Moderate"
            
            # Heat exposure: time with temp > 24°C
            heat_mask = df['temperature'] > 24
            heat_exposure_sec = df.loc[heat_mask, 'time_diff'].sum()
            heat_exposure_min = round(heat_exposure_sec / 60, 1)
            
            env_summary = {
                "condition": condition,
                "max_temp": round(max_temp, 1),
                "min_temp": round(min_temp, 1),
                "heat_exposure_duration_min": heat_exposure_min
            }
    
    metrics['environment_summary'] = env_summary

    # Ensure user_intent is blank object if None
    if metrics.get('user_intent') is None:
        metrics['user_intent'] = {}

    return metrics

def generate_ai_report(metrics: Dict, system_prompt: str, user_intent: Dict, csv_data: str = None, metadata: Dict = None, consolidate: bool = False, api_key: str = None, skip_ai: bool = False) -> tuple[str, str]:
    """
    Generates an AI narrative report based on the computed metrics.
    Returns: (ai_report_text, json_payload_string)
    """
    from google import genai
    from google.genai import types
    import os
    import time
    
    # Construct Payload
    payload = {
        "metrics": metrics,
        "user_intent": user_intent,
        "metadata": metadata or {},
        "metadata": metadata or {}
    }
    
    json_payload = json.dumps(payload, indent=2, default=str)
    
    if skip_ai:
        return None, json_payload
        
    # Configure AI
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        return f"Error: No API Key found. API Key is required for AI analysis.\n\nPayload Preview:\n{json_payload[:1000]}...", json_payload
        
    client = genai.Client(api_key=key)
    
    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model='gemini-3-pro-preview',
                contents=json_payload,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt
                )
            )
            # Safeguard: response.text can be None if content is blocked or empty
            if response.text is None:
                return f"AI Generation Warning: Response text is None. (Likely Blocked or Empty). \nResponse Candidates: {response.candidates}", json_payload
            return response.text, json_payload
            
        except Exception as e:
            err_str = str(e)
            # Retry on rate limit (429) or service unavailable (503)
            if ("429" in err_str or "503" in err_str) and attempt == 0:
                 time.sleep(30)
                 continue
                 
            return f"AI Generation Failed: {err_str}\n\nPayload Preview:\n{json_payload[:1000]}...", json_payload
            
    return "AI Generation Failed after retry.", json_payload

def save_session_artifacts(output_dir: Path, base_name: str, df: pd.DataFrame, metrics: Dict, ai_report_text: str, input_payload_json: str):
    """
    Saves session artifacts (CSV, Payload, Report) to a dedicated directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    csv_path = output_dir / f"{base_name}.csv"
    df.to_csv(csv_path, index=False)
    
    # Save Input Payload (JSON)
    payload_path = output_dir / f"{base_name}_payload.json"
    payload_path.write_text(input_payload_json, encoding='utf-8')
    
    # Save AI Report (Markdown)
    report_path = output_dir / f"{base_name}_report.md"
    report_path.write_text(ai_report_text, encoding='utf-8')



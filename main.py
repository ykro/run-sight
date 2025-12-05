# /// script
# dependencies = [
#   "pandas",
#   "numpy",
#   "fitdecode",
#   "rich",
#   "google-genai",
#   "python-dotenv",
# ]
# ///

import os
import sys
import argparse
import time
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

import pandas as pd
import numpy as np
import fitdecode
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.markdown import Markdown
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

# Constants
GEMINI_MODEL = 'gemini-3-pro-preview'
SYSTEM_PROMPT_PATH = Path('analysis.md')
CONSOLE = Console()

class AnalysisError(Exception):
    pass

def setup_client():
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        CONSOLE.print("[bold red]Error:[/bold red] GEMINI_API_KEY not found in environment variables.")
        sys.exit(1)
    return genai.Client(api_key=api_key)

def load_system_prompt():
    if not SYSTEM_PROMPT_PATH.exists():
        CONSOLE.print(f"[bold red]Error:[/bold red] {SYSTEM_PROMPT_PATH} not found.")
        sys.exit(1)
    return SYSTEM_PROMPT_PATH.read_text(encoding='utf-8')

def parse_fit_file(file_path: Path) -> tuple[pd.DataFrame, Dict]:
    """Parses a .fit file and returns a DataFrame of records and a metadata dict."""
    CONSOLE.print(f"[dim]Parsing {file_path.name}...[/dim]")
    
    data_points = []
    metadata = {
        'filename': file_path.name,
        'processed_at': datetime.now().isoformat(),
        'session_data': {}
    }

    try:
        with fitdecode.FitReader(file_path) as fit_file:
            for frame in fit_file:
                if frame.frame_type == fitdecode.FIT_FRAME_DATA:
                    if frame.name == 'record':
                        # Extract record data
                        point = {}
                        for field in frame.fields:
                            if field.value is not None:
                                # Handle cases where value might be a quantity with units
                                # fitdecode usually returns raw values or unit-aware objects depending on config
                                # We'll assume simple extraction for now or check types if needed
                                point[field.name] = field.value
                        data_points.append(point)
                    
                    elif frame.name == 'session':
                        # Extract session summary
                        for field in frame.fields:
                             if field.value is not None:
                                metadata['session_data'][field.name] = field.value

    except Exception as e:
        raise AnalysisError(f"Failed to parse .fit file: {e}")

    if not data_points:
        raise AnalysisError("No 'record' frames found in .fit file.")

    df = pd.DataFrame(data_points)
    
    # Standardize column names if needed (Fit files usually have consistant names like 'timestamp', 'heart_rate', etc.)
    # Ensure datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df, metadata

def calculate_metrics(df: pd.DataFrame, metadata: Dict) -> Dict:
    """Calculates forensic metrics."""
    metrics = {}
    
    # 0. Preprocessing
    # Calculate deltas
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds().fillna(0)
    df['dist_diff'] = df['distance'].diff().fillna(0)
    
    # Handle altitude if exists
    if 'enhanced_altitude' in df.columns:
        df['alt_diff'] = df['enhanced_altitude'].diff().fillna(0)
        # Calculate grade (rise/run)
        # Avoid division by zero
        df['grade'] = np.where(df['dist_diff'] > 0, df['alt_diff'] / df['dist_diff'], 0)
        df['grade_pct'] = df['grade'] * 100
    else:
        df['alt_diff'] = 0
        df['grade'] = 0
        df['grade_pct'] = 0

    # Ensure speed exists (enhanced_speed is m/s)
    if 'enhanced_speed' not in df.columns:
        # Estimate from distance/time
        df['enhanced_speed'] = np.where(df['time_diff'] > 0, df['dist_diff'] / df['time_diff'], 0)

    # 1. Context Detection: Ascent Ratio (m/km)
    # Total Ascent
    total_ascent = df[df['alt_diff'] > 0]['alt_diff'].sum()
    total_dist_km = df['distance'].max() / 1000.0 if df['distance'].max() > 0 else 0.001
    
    ascent_ratio = total_ascent / total_dist_km
    metrics['ascent_ratio'] = round(ascent_ratio, 2)
    metrics['context'] = 'Trail' if ascent_ratio > 25 else 'Road'

    # 2. Continuity Index
    # True Moving Time (Speed > 0.6 m/s)
    moving_mask = df['enhanced_speed'] > 0.6
    total_time_s = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds()
    moving_time_s = df.loc[moving_mask, 'time_diff'].sum()
    stop_time_s = total_time_s - moving_time_s
    stop_ratio = (stop_time_s / total_time_s) * 100 if total_time_s > 0 else 0

    metrics['total_time_sec'] = round(total_time_s, 2)
    metrics['moving_time_sec'] = round(moving_time_s, 2)
    metrics['stop_time_sec'] = round(stop_time_s, 2)
    metrics['stop_ratio_pct'] = round(stop_ratio, 2)

    # 3. Uphill Biomechanics
    # Filter: Gradient > 10% (Trail) or > 2% (Road)
    grade_threshold = 10.0 if metrics['context'] == 'Trail' else 2.0
    uphill_mask = df['grade_pct'] > grade_threshold
    
    uphill_df = df[uphill_mask].copy()
    
    metrics['over_striding_flag'] = False
    metrics['mech_inefficiency_flag'] = False

    if not uphill_df.empty:
        # Check Cadence and Stride Length
        if 'cadence' in uphill_df.columns:
            # Check units. If max is < 120, likely RPM. If > 120, likely SPM. 
            # We'll assume standard fit 'cadence' is RPM (one foot). 
            # User criteria: Cadence < 60spm. This is extremely low.
            # We will use the raw value check as requested, but if the data is typical RPM (e.g. 70-80), 
            # then 60 is a reasonable low cutoff.
            
            # Estimate stride length: Speed (m/s) * 60 / Cadence (SPM?)
            # If cadence is RPM, SPM = 2 * RPM.
            # Stride length = Speed / (RPM/60) -> meters per revolution (2 steps).
            # Stride length usually refers to one step length.
            # Let's assume 'cadence' column is RPM (standard).
            # Then SPM = cadence * 2.
            # Step Length = Speed (m/s) * 60 / (cadence * 2).
            # BUT user criteria: "Stride Length > 0.65m AND Cadence < 60spm".
            # If they mean SPM, 60 is walking. If they mean RPM, 60 RPM = 120 SPM.
            # I will assume "Cadence" in user prompt refers to the FIT FILE VALUE.
            
            # Let's implement literally.
            
            # Uphill averages
            avg_uphill_cadence = uphill_df['cadence'].mean()
            avg_uphill_speed = uphill_df['enhanced_speed'].mean()
            
            # Stride length calc (assuming single step)
            # If we don't know unit, we can guess or just stick to user strict logic.
            # Common fit: cadence (rpm).
            # We'll compute stride_m = speed * 60 / (cadence * 2) IF we assume rpm.
            # OR just use speed/cadence relationship if user provided inputs are raw.
            # Let's stick to calculating stride length properly assuming RPM.
            if avg_uphill_cadence > 0:
                 # Assume RPM if mean < 120, SPM if > 120
                is_rpm = avg_uphill_cadence < 120
                spm = avg_uphill_cadence * 2 if is_rpm else avg_uphill_cadence
                stride_m = avg_uphill_speed * 60 / spm
            else:
                stride_m = 0
            
            # Flag Logic
            # "Stride Length > 0.65m AND Cadence < 60spm"
            # 60spm is super low. I suspect they mean 60 RPM.
            # If I treat user inputs as "Raw Fit Values", then:
            raw_cadence = avg_uphill_cadence
            
            metrics['uphill_avg_cadence'] = round(raw_cadence, 1)
            metrics['uphill_avg_stride_m'] = round(stride_m, 2)

            if stride_m > 0.65 and raw_cadence < 60:
                metrics['over_striding_flag'] = True
        
        # Mech Inefficiency: VAM < 600m/h AND HR > Zone 4 (approx 165)
        # VAM m/h = (Vertical Ascent / Time) 
        # Average VAM on uphills
        uphill_time_s = uphill_df['time_diff'].sum()
        uphill_ascent_m = uphill_df['alt_diff'].sum()
        vam = (uphill_ascent_m / uphill_time_s) * 3600 if uphill_time_s > 0 else 0
        
        metrics['uphill_vam'] = round(vam, 1)
        
        if 'heart_rate' in uphill_df.columns:
            avg_uphill_hr = uphill_df['heart_rate'].mean()
            metrics['uphill_avg_hr'] = round(avg_uphill_hr, 1)
            if vam < 600 and avg_uphill_hr > 165:
                 metrics['mech_inefficiency_flag'] = True

    # 4. Physiological Cost
    # Cardiac Cost: Total Heart Beats / Total Distance (km)
    if 'heart_rate' in df.columns:
        # Beats = sum(HR(bpm) * time(min))
        total_beats = (df['heart_rate'] * (df['time_diff'] / 60)).sum()
        cardiac_cost = total_beats / total_dist_km if total_dist_km > 0 else 0
        metrics['cardiac_cost'] = round(cardiac_cost, 1)
    else:
        metrics['cardiac_cost'] = None

    # 5. Durability Index
    # Efficiency Factor (EF) = Speed (HP? No, Speed) / HR. User said Speed/HR.
    # 10 minute segments.
    if 'heart_rate' in df.columns:
        # Create 10min bin
        df['elapsed_min'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds() / 60
        df['segment_10m'] = (df['elapsed_min'] // 10).astype(int)
        
        segments = df.groupby('segment_10m').agg({
            'enhanced_speed': 'mean',
            'heart_rate': 'mean'
        })
        
        # Filter valid segments (Speed > 0 mainly)
        segments = segments[segments['enhanced_speed'] > 0]
        segments['EF'] = segments['enhanced_speed'] * 60 / segments['heart_rate'] # Speed is m/s. EF often m/min / bpm? Or just raw ratio. User just said Speed/HR.
        # I'll use m/min / BPM for readable numbers like 1-2. 
        # m/s / bpm is like 3 / 150 = 0.02.
        # Let's just do Speed(m/s) / HR for raw ratio unless specified.
        # Actually standard EF in TrainingPeaks is Yards/Min / HR or similar. 
        # I'll calculate simply Speed (m/s) / HR.
        segments['EF_raw'] = segments['enhanced_speed'] / segments['heart_rate']
        
        if len(segments) >= 2:
            first_ef = segments.iloc[0]['EF_raw']
            last_ef = segments.iloc[-1]['EF_raw']
            decline_pct = ((first_ef - last_ef) / first_ef) * 100 if first_ef > 0 else 0
            metrics['durability_index_decline_pct'] = round(decline_pct, 1)
            metrics['first_ef'] = first_ef
            metrics['last_ef'] = last_ef
        else:
            metrics['durability_index_decline_pct'] = 0.0

    # 6. Negative Split Index
    # Avg Speed 2nd Half / Avg Speed 1st Half
    half_dist = df['distance'].max() / 2
    first_half = df[df['distance'] <= half_dist]
    second_half = df[df['distance'] > half_dist]
    
    speed1 = first_half['enhanced_speed'].mean()
    speed2 = second_half['enhanced_speed'].mean()
    
    ns_index = speed2 / speed1 if speed1 > 0 else 0
    metrics['negative_split_index'] = round(ns_index, 3)

    # 7. GAP (Grade Adjusted Pace)
    # Estimate GAP for hardest uphill segment. 
    # Hardest uphill = max gradient segment?
    # Simple approx: Minetti. Cost = J/kg/m.
    # Logic: Finding worst uphill segment and calculating Gap there.
    # Or maybe user wants GAP stats overall? "Estimate GAP for the hardest uphill segment".
    # I'll find the minute with highest avg grade.
    if 'grade' in df.columns:
        # Rolling 1 min average of grade
        # Assume roughly 1 record per second
        df['rolling_grade'] = df['grade'].rolling(60).mean()
        max_grade_idx = df['rolling_grade'].idxmax()
        
        if pd.notna(max_grade_idx):
            # Get data for that segment 
            segment_grade = df.loc[max_grade_idx, 'rolling_grade']
            segment_speed = df.loc[max_grade_idx, 'enhanced_speed'] # Should be rolling speed too
            # Smoothed speed
            segment_rolling_speed = df['enhanced_speed'].rolling(60).mean().loc[max_grade_idx]
            
            # Strava/Minetti Approx Rule of Thumb:
            # GAP = Pace / (1 + 9 * grade) for running?
            # Or Energy Cost Ratio.
            # Minetti Energy Cost (J/kg/m) C(i) = 155.4i^5 - 30.4i^4 - 43.3i^3 + 46.3i^2 + 19.5i + 3.6
            # C(0) = 3.6
            # Ratio = C(grade) / 3.6
            # GAP Speed = Real Speed * Ratio
            # Wait, GAP Speed (flat equivalent) is HIGHER on uphills.
            # So Real Speed * Ratio.
            
            i = segment_grade # gradient (rise/run)
            cost = 155.4*(i**5) - 30.4*(i**4) - 43.3*(i**3) + 46.3*(i**2) + 19.5*i + 3.6
            ratio = cost / 3.6
            gap_speed = segment_rolling_speed * ratio
            
            metrics['hardest_uphill_grade_pct'] = round(segment_grade * 100, 1)
            metrics['hardest_uphill_real_speed'] = round(segment_rolling_speed, 2)
            metrics['hardest_uphill_gap_speed_est'] = round(gap_speed, 2)
        else:
             metrics['hardest_uphill_gap_speed_est'] = None

    # 8. Downhill Agility
    # Gradient < -10%
    if 'grade_pct' in df.columns:
        downhill_mask = df['grade_pct'] < -10
        downhill_df = df[downhill_mask]
        
        if not downhill_df.empty:
            avg_descent_speed = downhill_df['enhanced_speed'].mean()
            # Vertical Descent Speed (m/h) = (Descent meters / time)
            descended_m = downhill_df['alt_diff'].sum() # Should be negative
            time_s = downhill_df['time_diff'].sum()
            vert_descent_speed = (abs(descended_m) / time_s) * 3600 if time_s > 0 else 0
            
            metrics['avg_descent_speed_ms'] = round(avg_descent_speed, 2)
            metrics['vert_descent_speed_mh'] = round(vert_descent_speed, 1)
        else:
            metrics['avg_descent_speed_ms'] = None
            metrics['vert_descent_speed_mh'] = None

    # 9. Power Efficiency
    if 'power' in df.columns and 'heart_rate' in df.columns:
        # Filter power > 0
        power_df = df[df['power'] > 0]
        if not power_df.empty:
            avg_power = power_df['power'].mean()
            avg_hr_pwr = power_df['heart_rate'].mean()
            metrics['watts_per_beat'] = round(avg_power / avg_hr_pwr, 2) if avg_hr_pwr > 0 else 0
        else:
             metrics['watts_per_beat'] = None

    return metrics

def generate_ai_report(metrics_list: List[Dict], system_prompt: str, feedback: str, consolidate: bool = False):
    """Generates the AI report using Google GenAI."""
    
    client = setup_client()
    
    if consolidate:
        mode_text = "MODE 2: CONSOLIDATED TREND ANALYSIS"
        data_payload = json.dumps(metrics_list, indent=2)
    else:
        mode_text = "MODE 1: INDIVIDUAL REPORT"
        data_payload = json.dumps(metrics_list[0], indent=2) # Single item

    prompt = f"""
    {system_prompt}
    
    === REQUEST ===
    {mode_text}
    
    USER FEEDBACK: {feedback}
    
    DATA PAYLOAD:
    {data_payload}
    """

    CONSOLE.print(f"[bold blue]Consulting {GEMINI_MODEL}...[/bold blue]")
    
    retries = 3
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )
            return response.text
        except types.ClientError as e:
            # Check for 429 using string matching if status code isn't directly available in this SDK version exception easily
            # But usually it's in e.code or similar. 
            # We'll just catch general errors for now and look for rate limit indications
            error_str = str(e)
            if "429" in error_str or "503" in error_str:
                if attempt < retries - 1:
                    CONSOLE.print(f"[bold red]Rate Limit/Service Error. Retrying in 60s... ({attempt+1}/{retries})[/bold red]")
                    time.sleep(60)
                else:
                    raise AnalysisError("Refused by AI after 3 retries.")
            else:
                raise e

def save_artifacts(df: pd.DataFrame, metadata: Dict, metrics: Dict, file_path: Path):
    """Saves intermediate CSV/JSON artifacts."""
    base_name = file_path.stem
    parent = file_path.parent
    
    # Save CSV
    csv_path = parent / f"{base_name}.csv"
    df.to_csv(csv_path, index=False)
    
    # Save Metadata JSON
    meta_path = parent / f"{base_name}.json"
    meta_path.write_text(json.dumps(metadata, indent=2, default=str), encoding='utf-8')
    
    # Save Metrics JSON
    metrics_path = parent / f"{base_name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str), encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description="Runsight: Forensic Athletic Analysis")
    parser.add_argument("path", type=Path, help="Path to .fit file or directory")
    parser.add_argument("-r", "--recursive", action="store_true", help="Process directory recursively")
    parser.add_argument("-c", "--consolidate", action="store_true", help="Generate consolidated report")
    parser.add_argument("--feedback", type=str, default="", help="User feedback context")
    
    args = parser.parse_args()
    
    system_prompt = load_system_prompt()
    
    files_to_process = []
    if args.path.is_file():
        files_to_process = [args.path]
    elif args.path.is_dir():
        pattern = "**/*.fit" if args.recursive else "*.fit"
        files_to_process = list(args.path.glob(pattern))
    
    if not files_to_process:
        CONSOLE.print("[yellow]No .fit files found.[/yellow]")
        return

    all_metrics = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        
        task = progress.add_task(f"Processing {len(files_to_process)} files...", total=len(files_to_process))
        
        for file_path in files_to_process:
            try:
                # Phase A: Parse
                df, metadata = parse_fit_file(file_path)
                
                # Phase B: Metrics
                metrics = calculate_metrics(df, metadata)
                metrics['filename'] = file_path.name
                
                # Save Artifacts
                save_artifacts(df, metadata, metrics, file_path)
                
                all_metrics.append(metrics)
                progress.advance(task)
                
            except Exception as e:
                CONSOLE.print(f"[red]Error processing {file_path.name}: {e}[/red]")

    if not all_metrics:
        CONSOLE.print("[red]No metrics generated.[/red]")
        return

    # Phase C: AI Report
    try:
        report = generate_ai_report(
            all_metrics, 
            system_prompt, 
            args.feedback, 
            consolidate=args.consolidate
        )
        
        CONSOLE.print(Panel(Markdown(report), title="Forensic Report", border_style="green"))
    except Exception as e:
        CONSOLE.print(f"[bold red]AI Reporting Failed:[/bold red] {e}")

if __name__ == "__main__":
    main()

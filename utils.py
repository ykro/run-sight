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

def load_system_prompt(path: str = 'analysis.md') -> str:
    p = Path(path)
    if not p.exists():
        # Fallback if running from a different dir
        return "You are an Elite Sports Data Scientist..." 
    return p.read_text(encoding='utf-8')

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

def save_to_firestore(collection: str, document_id: str, data: Dict):
    """Saves data to a Firestore document."""
    try:
        # Client will automatically look for GOOGLE_APPLICATION_CREDENTIALS
        db = firestore.Client(project=GCP_PROJECT_ID, database=FIRESTORE_DB_NAME)
        doc_ref = db.collection(collection).document(document_id)
        doc_ref.set(data)
        return True
    except Exception as e:
        print(f"Failed to save to Firestore: {e}")
        return False

# --- CORE LOGIC ---

def parse_fit_file(file_path: Path) -> tuple[pd.DataFrame, Dict]:
    # ... (Keep existing implementation) ...
    """Parses a .fit file and returns a DataFrame of records and a metadata dict."""
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
                        point = {}
                        for field in frame.fields:
                            if field.value is not None:
                                point[field.name] = field.value
                        data_points.append(point)
                    elif frame.name == 'session':
                        for field in frame.fields:
                             if field.value is not None:
                                metadata['session_data'][field.name] = field.value

    except Exception as e:
        raise AnalysisError(f"Failed to parse .fit file: {e}")

    if not data_points:
        raise AnalysisError("No 'record' frames found in .fit file.")

    df = pd.DataFrame(data_points)
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df, metadata

def calculate_metrics(df: pd.DataFrame, metadata: Dict) -> Dict:
    # ... (Keep existing implementation) ...
    """Calculates forensic metrics."""
    metrics = {}
    
    # Preprocessing
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds().fillna(0)
    df['dist_diff'] = df['distance'].diff().fillna(0)
    
    if 'enhanced_altitude' in df.columns:
        df['alt_diff'] = df['enhanced_altitude'].diff().fillna(0)
        df['grade'] = np.where(df['dist_diff'] > 0, df['alt_diff'] / df['dist_diff'], 0)
        df['grade_pct'] = df['grade'] * 100
    else:
        df['alt_diff'] = 0
        df['grade'] = 0
        df['grade_pct'] = 0

    if 'enhanced_speed' not in df.columns:
        df['enhanced_speed'] = np.where(df['time_diff'] > 0, df['dist_diff'] / df['time_diff'], 0)

    # 1. Context Detection
    total_ascent = df[df['alt_diff'] > 0]['alt_diff'].sum()
    total_dist_km = df['distance'].max() / 1000.0 if df['distance'].max() > 0 else 0.001
    ascent_ratio = total_ascent / total_dist_km
    metrics['ascent_ratio'] = round(ascent_ratio, 2)
    metrics['context'] = 'Trail' if ascent_ratio > 25 else 'Road'

    # 2. Continuity Index
    moving_mask = df['enhanced_speed'] > 0.6
    total_time_s = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds()
    moving_time_s = df.loc[moving_mask, 'time_diff'].sum()
    stop_time_s = total_time_s - moving_time_s
    stop_ratio = (stop_time_s / total_time_s) * 100 if total_time_s > 0 else 0
    
    metrics['total_time_sec'] = round(total_time_s, 2)
    metrics['stop_time_sec'] = round(stop_time_s, 2)
    metrics['stop_ratio_pct'] = round(stop_ratio, 2)

    # 3. Uphill Biomechanics
    grade_threshold = 10.0 if metrics['context'] == 'Trail' else 2.0
    uphill_mask = df['grade_pct'] > grade_threshold
    uphill_df = df[uphill_mask].copy()
    
    metrics['over_striding_flag'] = False
    metrics['mech_inefficiency_flag'] = False

    if not uphill_df.empty:
        if 'cadence' in uphill_df.columns:
            avg_uphill_cadence = uphill_df['cadence'].mean()
            avg_uphill_speed = uphill_df['enhanced_speed'].mean()
            # Stride calc
            if avg_uphill_cadence > 0:
                is_rpm = avg_uphill_cadence < 120
                spm = avg_uphill_cadence * 2 if is_rpm else avg_uphill_cadence
                stride_m = avg_uphill_speed * 60 / spm
            else:
                stride_m = 0
                
            metrics['uphill_avg_cadence'] = round(avg_uphill_cadence, 1)
            metrics['uphill_avg_stride_m'] = round(stride_m, 2)

            if stride_m > 0.65 and avg_uphill_cadence < 60:
                metrics['over_striding_flag'] = True
        
        uphill_time_s = uphill_df['time_diff'].sum()
        uphill_ascent_m = uphill_df['alt_diff'].sum()
        vam = (uphill_ascent_m / uphill_time_s) * 3600 if uphill_time_s > 0 else 0
        metrics['uphill_vam'] = round(vam, 1)
        
        if 'heart_rate' in uphill_df.columns:
            avg_uphill_hr = uphill_df['heart_rate'].mean()
            if vam < 600 and avg_uphill_hr > 165:
                 metrics['mech_inefficiency_flag'] = True

    # 4. Physiological Cost
    if 'heart_rate' in df.columns:
        total_beats = (df['heart_rate'] * (df['time_diff'] / 60)).sum()
        metrics['cardiac_cost'] = round(total_beats / total_dist_km, 1) if total_dist_km > 0 else 0

    # 5. Durability Index
    if 'heart_rate' in df.columns:
        df['elapsed_min'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds() / 60
        df['segment_10m'] = (df['elapsed_min'] // 10).astype(int)
        
        segments = df.groupby('segment_10m').agg({'enhanced_speed': 'mean', 'heart_rate': 'mean'})
        segments = segments[segments['enhanced_speed'] > 0]
        segments['EF_raw'] = segments['enhanced_speed'] / segments['heart_rate']
        
        if len(segments) >= 2:
            first_ef = segments.iloc[0]['EF_raw']
            last_ef = segments.iloc[-1]['EF_raw']
            decline_pct = ((first_ef - last_ef) / first_ef) * 100 if first_ef > 0 else 0
            metrics['durability_index_decline_pct'] = round(decline_pct, 1)

    # 6. Negative Split
    half_dist = df['distance'].max() / 2
    first_half = df[df['distance'] <= half_dist]
    second_half = df[df['distance'] > half_dist]
    
    speed1 = first_half['enhanced_speed'].mean()
    speed2 = second_half['enhanced_speed'].mean()
    metrics['negative_split_index'] = round(speed2 / speed1, 3) if speed1 > 0 else 0

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

    # 8. Power Efficiency
    if 'power' in df.columns and 'heart_rate' in df.columns:
        power_df = df[df['power'] > 0]
        if not power_df.empty:
            metrics['watts_per_beat'] = round(power_df['power'].mean() / power_df['heart_rate'].mean(), 2)

    # 9. Coeficiente de Eficiencia (Speed to Power Ratio)
    if 'power' in df.columns and 'enhanced_speed' in df.columns:
        # Filtramos zonas de carrera real (> 1.5 m/s) y potencia positiva
        re_df = df[(df['power'] > 0) & (df['enhanced_speed'] > 1.5)]
        
        if not re_df.empty:
            avg_speed = re_df['enhanced_speed'].mean()
            avg_power = re_df['power'].mean()
            metrics['running_efficiency_raw'] = round(avg_speed / avg_power, 5)

    # 10. Pace Variability Index (VI)
    if 'grade' in df.columns and 'enhanced_speed' in df.columns:
        # Limitar pendiente a rangos correibles (-30% a +30%) para estabilidad
        g = df['grade'].clip(-0.30, 0.30)
        
        # Costo Energético de Minetti (Simplificado para cálculo rápido)
        cost = 155.4 * g**5 - 30.4 * g**4 - 43.3 * g**3 + 46.3 * g**2 + 19.5 * g + 3.6
        
        # GAP Speed
        gap_speed = df['enhanced_speed'] * (cost / 3.6)
        
        # Velocidad Normalizada (NGP)
        rolling_gap = gap_speed.rolling(window=30, min_periods=1).mean()
        ngp_speed = np.mean(rolling_gap ** 4) ** 0.25
        
        avg_gap_speed = gap_speed.mean()
        
        if avg_gap_speed > 0:
            metrics['pace_variability_index'] = round(ngp_speed / avg_gap_speed, 3)

    return metrics

def generate_ai_report(metrics_list: List[Dict], system_prompt: str, feedback: str, consolidate: bool = False, api_key: str = None) -> str:
    """Generates the AI report using Google GenAI."""
    
    client = setup_client(api_key)
    
    if consolidate:
        mode_text = "MODE 2: CONSOLIDATED TREND ANALYSIS"
        data_payload = json.dumps(metrics_list, indent=2)
    else:
        mode_text = "MODE 1: INDIVIDUAL REPORT"
        data_payload = json.dumps(metrics_list[0], indent=2)

    prompt = f"""
    {system_prompt}
    
    === REQUEST ===
    {mode_text}
    
    USER FEEDBACK: {feedback}
    
    DATA PAYLOAD:
    {data_payload}
    """
    
    retries = 3
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )
            return response.text
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "503" in error_str:
                if attempt < retries - 1:
                    time.sleep(60)
                else:
                    raise AnalysisError("Refused by AI after 3 retries.")
            else:
                raise e

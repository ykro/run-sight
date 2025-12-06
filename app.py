import streamlit as st
import pandas as pd
import json
import os
import tempfile
import io
import re
from pathlib import Path
from dotenv import load_dotenv
import plotly.express as px
import matplotlib.pyplot as plt
from fpdf import FPDF
from fpdf.enums import XPos, YPos

import utils as core

# Page Config
st.set_page_config(
    page_title="Runsight AI | Performance Analysis",
    layout="wide"
)

# Load Env (Locally)
load_dotenv()

# --- PDF Generation Class ---
class PDFReport(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 14)
        self.cell(0, 10, 'Runsight AI', border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
        self.set_font('Helvetica', 'I', 10)
        self.cell(0, 5, 'Performance Report', border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
        self.ln(5)

    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 12)
        self.set_fill_color(220, 220, 220)
        self.cell(0, 8, title, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
        self.ln(3)

    def chapter_body(self, text):
        self.set_font('Helvetica', '', 10)
        self.multi_cell(0, 6, text)
        self.ln()

    def add_plot(self, img_buf):
        # Image from bytes buffer
        self.image(img_buf, w=170)
        self.ln(5)

def strip_markdown(text):
    """
    Strips basic markdown syntax (**bold**, ## Header, etc) for clean PDF output.
    """
    text = re.sub(r'#{1,6}\s?', '', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    return text

def sanitize_input(text):
    """Basic sanitization for user input notes."""
    if not text:
        return ""
    # Remove potentially dangerous characters/scripts
    # Allow alphanumeric, basic punctuation, newlines
    safe_text = re.sub(r'[<>{}]', '', text) 
    return safe_text.strip()[:500] # Limit length

def validate_fit_file(file_obj):
    """
    Validates if the uploaded file is a valid FIT file containing record data.
    """
    try:
        # We need to read the file, so we wrap it or reset pointer
        from fitdecode import FitReader, FitHeaderError
        
        # Streamlit UploadedFile behavior: getvalue() returns bytes
        # FitReader expects a path or file-like object
        with io.BytesIO(file_obj.getvalue()) as f:
            with FitReader(f) as fit:
                # Iterate briefly to check for data
                has_records = False
                for frame in fit:
                    if frame.frame_type == core.fitdecode.FIT_FRAME_DATA and frame.name == 'record':
                        has_records = True
                        break
                
                if not has_records:
                    return False, "File contains no activity records."
                    
        return True, "Valid"
    except Exception:
        return False, "Corrupted or invalid .fit file format."

def create_enhanced_pdf(text, metrics, df):
    pdf = PDFReport()
    pdf.add_page()
    
    clean_text = strip_markdown(text)
    
    # 1. AI Analysis Text
    pdf.chapter_title("Deep Performance Analysis")
    safe_text = clean_text.encode('latin-1', 'replace').decode('latin-1')
    pdf.chapter_body(safe_text)
    pdf.ln(5)

    # 2. Metrics Summary Table
    pdf.chapter_title("Key Performance Metrics")
    pdf.set_font("Helvetica", size=9)
    col_width = pdf.epw / 3
    data = [
        ("Context", metrics.get('context', 'N/A')),
        ("Cardiac Cost", f"{metrics.get('cardiac_cost', 0)} beats/km"),
        ("Stop Ratio", f"{metrics.get('stop_ratio_pct', 0)} %"),
        ("Durability (Decline)", f"{metrics.get('durability_index_decline_pct', 0)} %"),
        ("Uphill VAM", f"{metrics.get('uphill_vam', 0)} m/h"),
        ("Neg. Split Index", str(metrics.get('negative_split_index', 0))),
    ]
    
    row_height = 6
    for key, value in data:
        pdf.cell(col_width, row_height, key, border=1)
        pdf.cell(col_width, row_height, str(value), border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.ln(5)

    # 3. Charts
    pdf.chapter_title("Telemetry Visualization")
    if 'heart_rate' in df.columns:
        plt.figure(figsize=(10, 3))
        plt.plot(df['timestamp'], df['heart_rate'], color='red', linewidth=1)
        plt.title("Heart Rate Profile")
        plt.ylabel("BPM")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        pdf.add_plot(buf)
        plt.close()

    if 'enhanced_speed' in df.columns:
        plt.figure(figsize=(10, 3))
        plt.plot(df['timestamp'], df['enhanced_speed'], color='blue', linewidth=1)
        plt.title("Speed Profile")
        plt.ylabel("m/s")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        pdf.add_plot(buf)
        plt.close()
    
    return bytes(pdf.output())

# --- UI State Management ---
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'report_content' not in st.session_state:
    st.session_state.report_content = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'filename' not in st.session_state:
    st.session_state.filename = ""
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False

def start_processing():
    st.session_state.is_processing = True

# Title
st.title("Runsight AI: Performance Analysis")

# --- CONTROL PANEL ---
api_key_env = os.getenv("GEMINI_API_KEY")

# Use a Form to prevent interaction while processing and bundle inputs
with st.container(border=True):
    with st.form("analysis_form"):
        col1, col2 = st.columns([3, 1]) 
        
        with col1:
            st.markdown("#### Input")
            # Keys are required for persistent access across reruns
            uploaded_file = st.file_uploader("Upload .fit file", type=['fit'], 
                                            label_visibility="collapsed", key='fit_file')
            
        with col2:
            st.markdown("#### Notes (Optional)")
            feedback = st.text_area("Context", height=100, placeholder="e.g. Race Day...",
                                   label_visibility="collapsed", key='feedback_text') 

        # Submit Button
        # Locked whenever processing is active
        if api_key_env:
            st.form_submit_button("Start Analysis", 
                                  type="primary", 
                                  disabled=st.session_state.is_processing,
                                  on_click=start_processing)
        else:
            st.error("API Key missing in environment.")

# --- PROCESSING ---
# Triggered by session state, not just the button press
if st.session_state.is_processing:
    # Need to retrieve values from session state because we might be in a rerun
    uploaded_file = st.session_state.get('fit_file')
    feedback = st.session_state.get('feedback_text', '')

    if uploaded_file:
        # Clear previous results immediately
        st.session_state.analysis_done = False
        st.session_state.report_content = None
        st.session_state.metrics = None
        st.session_state.df = None
        
        # > Validation Checks
        is_valid, msg = validate_fit_file(uploaded_file)
        if not is_valid:
            st.error(f"Invalid File: {msg}")
            st.session_state.is_processing = False
            st.rerun()
            
        # > Sanitization
        safe_feedback = sanitize_input(feedback)
        
        with st.spinner("Analysis in Progress... Consulting AI Model..."):
            try:
                # File handling
                with tempfile.NamedTemporaryFile(delete=False, suffix=".fit") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = Path(tmp_file.name)

                try:
                    # 1. Parse & Metrics
                    df, metadata = core.parse_fit_file(tmp_path)
                    metrics = core.calculate_metrics(df, metadata)
                    metrics['filename'] = uploaded_file.name
                    
                    # 2. AI Report
                    system_prompt = core.load_system_prompt("analysis.md")
                    # Smart Downsampling for AI Context
                    if len(df) > 2000:
                        step = len(df) // 2000
                        csv_data_str = df.iloc[::step, :].to_csv(index=False)
                    else:
                        csv_data_str = df.to_csv(index=False)

                    report_content = core.generate_ai_report(
                        [metrics], 
                        system_prompt, 
                        safe_feedback, 
                        csv_data=csv_data_str,
                        metadata=metadata,
                        api_key=api_key_env
                    )

                    # 3. Cloud Storage Integration
                    # Upload original FIT file to GCS
                    gcs_uri = core.upload_to_gcs(tmp_path, destination_blob_name=uploaded_file.name)
                    if gcs_uri:
                        metrics['gcs_uri'] = gcs_uri
                        # Silent upload (User requested removal of toast)

                    # Save result to Firestore
                    if core.GCP_PROJECT_ID:
                        import time
                        doc_id = f"{uploaded_file.name}_{int(time.time())}"
                        firestore_data = {
                            'filename': uploaded_file.name,
                            'timestamp': time.time(),
                            'metrics': metrics,
                            'report': report_content,
                            'feedback': safe_feedback
                        }
                        core.save_to_firestore('reports', doc_id, firestore_data)
                        # Silent save (User requested removal of toast)
                    
                    # Update State
                    st.session_state.df = df
                    st.session_state.metrics = metrics
                    st.session_state.report_content = report_content
                    st.session_state.filename = uploaded_file.name
                    st.session_state.analysis_done = True
                    
                    st.toast("Analysis Completed Successfully!", icon="✅")
                    
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink()
                        
            except Exception as e:
                st.error(f"Analysis Failed: {e}")
            finally:
                # Unlock and reset
                st.session_state.is_processing = False
                st.rerun()
    else:
        st.warning("Please upload a file to begin.")
        st.session_state.is_processing = False
        st.rerun()

# --- RESULTS DISPLAY ---
if st.session_state.analysis_done:
    st.divider()
    
    col_dl, col_pad = st.columns([1, 4])
    with col_dl:
        if st.session_state.report_content:
            pdf_bytes = create_enhanced_pdf(
                st.session_state.report_content, 
                st.session_state.metrics,
                st.session_state.df
            )
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_bytes,
                file_name=f"runsight_report_{st.session_state.filename}.pdf",
                mime="application/pdf",
                type="primary"
            )

    tab1, tab2, tab3 = st.tabs(["Performance Report", "Telemetry", "Metrics"])
    
    with tab1:
        st.subheader("Analysis")
        st.markdown(st.session_state.report_content)
        
    with tab2:
        st.subheader("Telemetry Analysis")
        df = st.session_state.df
        if 'heart_rate' in df.columns:
            st.plotly_chart(px.line(df, x='timestamp', y='heart_rate', title='Heart Rate Analysis').update_layout(xaxis_title='Time', yaxis_title='BPM'), width="stretch")
        if 'enhanced_speed' in df.columns:
            st.plotly_chart(px.line(df, x='timestamp', y='enhanced_speed', title='Speed Analysis').update_layout(xaxis_title='Time', yaxis_title='Speed (m/s)'), width="stretch")

    with tab3:
        st.subheader("Calculated Metrics")
        st.json(st.session_state.metrics)

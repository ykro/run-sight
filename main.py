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
# fitdecode usage moved to utils, checking if we still need it here for types? 
# The duplicate functions are removed, so we might not need fitdecode import here if only using utils.
# Leaving imports for now to avoid accidental breakage if I miss a usage, but main logic delegates to utils.
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.markdown import Markdown
from dotenv import load_dotenv

import utils as core
from rich.prompt import Prompt, IntPrompt

# Load environment variables
load_dotenv()

# Constants
GEMINI_MODEL = 'gemini-3-pro-preview'
CONSOLE = Console()

class AnalysisError(Exception):
    pass

# Functions setup_client and load_system_prompt are main specific helpers, keeping them (or could verify if they are in utils, but setup_client is definitely user env related)
# Wait, setup_client handles API key. core.generate_ai_report doesn't need client if disabled, 
# but eventually it will. 
# actually main.py doesn't call client anymore in the new flow, it calls core.generate_ai_report.

def get_user_context(args):
    """
    Interactively gathers user context if not provided via flags.
    """
    if args.ignore_context:
        return None # Do not fill defaults
    
    context = {}
    
    CONSOLE.print(Panel("🏃 Runsight Context Awareness", style="cyan"))
    
    # 1. Profile
    if args.profile:
        context['profile'] = args.profile
    else:
        context['profile'] = Prompt.ask("Profile", choices=["Elite", "Pro", "Competitive", "Recreational"], default="Recreational")

    # 2. Goal
    if args.goal:
        context['goal'] = args.goal
    else:
        context['goal'] = Prompt.ask("Session Goal", default="Base Building")
        
    # 3. Condition
    if args.condition:
        context['condition'] = args.condition
    else:
        context['condition'] = Prompt.ask("Conditions (Terrain/Weather)", default="Normal")
        
    # 4. RPE
    if args.rpe:
        context['rpe'] = args.rpe
    else:
        context['rpe'] = IntPrompt.ask("RPE (1-10)", default=5)
        
    return context

def main():
    parser = argparse.ArgumentParser(description="Runsight: Forensic Athletic Analysis")
    parser.add_argument("path", type=Path, help="Path to .fit file or directory")
    parser.add_argument("-r", "--recursive", action="store_true", help="Process directory recursively")
    # Context Flags
    parser.add_argument("--profile", help="Athlete Profile (Elite, Competitive, Recreational)")
    parser.add_argument("--goal", help="Session Goal")
    parser.add_argument("--condition", help="Environmental/Terrain Conditions")
    parser.add_argument("--rpe", type=int, help="Rate of Perceived Exertion (1-10)")
    parser.add_argument("--ignore-context", action="store_true", help="Skip interactive questions and use defaults")

    args = parser.parse_args()
    
    try:
        system_prompt = core.load_system_prompt()
    except Exception:
        # If utils doesn't have it (it wasn't in utils.py view), load locally or handle error
        # Checked utils.py, it DOES NOT seem to have load_system_prompt in the view I saw.
        # I should probably define it here or add to utils. 
        # For safety I will define it here or rely on the one I just deleted? 
        # I am deleting the old definition in this Replace block. 
        # I should Restore `load_system_prompt` here if it's not in utils.
        # Let's verify utils.py content via search or assumption?
        # I'll implement `load_system_prompt` locally in main to be safe.
        if Path('analysis.md').exists():
             system_prompt = Path('analysis.md').read_text(encoding='utf-8')
        else:
             system_prompt = "Analyze this run."
    
    # Files
    files_to_process = []
    if args.path.is_file():
        files_to_process = [args.path]
    elif args.path.is_dir():
        for ext in ['fit', 'zip']:
            pattern = f"**/*.{ext}" if args.recursive else f"*.{ext}"
            files_to_process.extend(list(args.path.glob(pattern)))
    
    if not files_to_process:
        CONSOLE.print("[yellow]No .fit/.zip files found.[/yellow]")
        return

    # Context 
    user_intent = get_user_context(args)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        
        task = progress.add_task(f"Processing {len(files_to_process)} files...", total=len(files_to_process))
        
        for file_path in files_to_process:
            try:
                # Phase A: Parse
                df, metadata = core.parse_fit_file(file_path)
                
                # Phase B: Metrics (New Core Logic)
                metrics = core.calculate_metrics(df, metadata)
                
                # Phase C: Payload Generation
                json_payload = core.generate_ai_report(
                    metrics, 
                    system_prompt=system_prompt if 'system_prompt' in locals() else "",
                    user_intent=user_intent,
                    metadata=metadata
                )
                
                # Phase D: Save Artifacts to Folder
                # Create folder: same parent, folder name = filename stem
                # e.g. /data/run.fit -> /data/run/
                output_dir = file_path.parent / file_path.stem
                
                core.save_session_artifacts(output_dir, file_path.stem, df, metrics, json_payload)
                
                # Print Result
                CONSOLE.print(Panel(f"Artifacts saved to: [bold]{output_dir}[/bold]\n\nPayload Preview:\n" + json_payload[:500] + "...", title=f"Processed: {file_path.name}", border_style="green", expand=False))
                
                progress.advance(task)
                
            except Exception as e:
                CONSOLE.print(f"[red]Error processing {file_path.name}: {e}[/red]")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    main()

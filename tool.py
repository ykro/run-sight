

import os
import sys
import argparse
import time
import json
import traceback
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

import pandas as pd
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt, IntPrompt
from dotenv import load_dotenv

import utils as core

# Load environment variables
load_dotenv()

# Constants
GEMINI_MODEL = 'gemini-3-pro-preview'
CONSOLE = Console()

def get_user_context(args):
    """
    Interactively gathers user context if not provided via flags.
    """
    context = {}
    
    # Check flags first
    if args.profile:
        context['profile'] = args.profile
    if args.goal:
        context['goal'] = args.goal
    if args.condition:
        context['condition'] = args.condition
    if args.rpe:
        context['rpe'] = args.rpe
    if args.max_hr:
        context['max_hr'] = args.max_hr
        
    # If ignore-context is set, return what we have (or None if empty)
    if args.ignore_context:
        return context if context else None

    # Interactive Prompts for missing values
    CONSOLE.print(Panel("🏃 Runsight Context Awareness", style="cyan"))
    
    if 'profile' not in context:
        context['profile'] = Prompt.ask("Profile", choices=["Elite", "Pro", "Competitive", "Recreational"], default="Recreational")

    if 'goal' not in context:
        context['goal'] = Prompt.ask("Session Goal", default="Base Building")
        
    if 'condition' not in context:
        context['condition'] = Prompt.ask("Conditions (Terrain/Weather)", default="Normal")
        
    if 'rpe' not in context:
        context['rpe'] = IntPrompt.ask("RPE (1-10)", default=5)
        
    return context

def main():
    parser = argparse.ArgumentParser(description="Runsight: Forensic Athletic Analysis")
    parser.add_argument("path", type=Path, help="Path to .fit file or directory")
    
    # Context Flags
    parser.add_argument("--profile", help="Athlete Profile (Elite, Competitive, Recreational)")
    parser.add_argument("--goal", help="Session Goal")
    parser.add_argument("--condition", help="Environmental/Terrain Conditions")
    parser.add_argument("--rpe", type=int, help="Rate of Perceived Exertion (1-10)")
    parser.add_argument("--max-hr", type=int, help="Override Max HR for Zone calculations")
    parser.add_argument("--ignore-context", action="store_true", help="Skip interactive questions and use defaults")
    parser.add_argument("--payload-only", action="store_true", help="Generate only the JSON payload, skipping AI analysis")

    args = parser.parse_args()
    
    # Load System Prompt
    system_prompt = core.load_system_prompt()

    # Find files
    files_to_process = []
    if args.path.is_file():
        files_to_process = [args.path]
    elif args.path.is_dir():
        # Implicitly recursive for directories
        for ext in ['fit', 'zip']:
            pattern = f"**/*.{ext}"
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
                CONSOLE.print(f"[dim]Parsing {file_path.name}...[/dim]")
                df, metadata = core.parse_fit_file(file_path)
                
                # Phase B: Metrics
                CONSOLE.print("[dim]Calculating metrics...[/dim]")
                metrics = core.calculate_metrics(df, metadata, user_intent=user_intent)
                
                # Phase C: Report/Payload
                # Logic from main.py: Generate payload (and possibly AI report if enabled later)
                if not args.payload_only:
                    CONSOLE.print("[dim]Generating AI Report (Sending Request)...[/dim]")
                else:
                    CONSOLE.print("[dim]Generating Payload (Skipping AI)...[/dim]")

                ai_report, json_payload = core.generate_ai_report(
                    metrics, 
                    system_prompt=system_prompt, 
                    user_intent=user_intent, 
                    metadata=metadata,
                    skip_ai=args.payload_only
                )
                if not args.payload_only:
                    CONSOLE.print("[dim]Request Completed.[/dim]")
                
                # Phase D: Save to Dedicated Folder
                output_dir = file_path.parent / file_path.stem
                # Note: save_session_artifacts expects ai_report_text as string. If skipped, ai_report is None.
                # We should handle this or pass a placeholder. 
                # Let's pass empty string or handled in util? 
                # The util writes text. None.write_text will fail.
                # Updating logic to safeguard against None report.
                core.save_session_artifacts(output_dir, file_path.stem, df, metrics, ai_report or "", json_payload)

                # Print Result
                if args.payload_only:
                     CONSOLE.print(Panel(f"Payload saved to: [bold]{output_dir}[/bold]\n\nPayload Preview:\n" + json_payload[:500] + "...", title=f"Payload Generated: {file_path.name}", border_style="green", expand=False))
                elif ai_report and ai_report.strip().startswith("{"):
                    # Fallback (AI Failed/Disabled)
                    CONSOLE.print(Panel(f"Artifacts saved to: [bold]{output_dir}[/bold]\n\nPayload Preview:\n" + ai_report[:500] + "...", title=f"Processed: {file_path.name}", border_style="green", expand=False))
                else:
                     # It's an AI Report
                    CONSOLE.print(Panel(Markdown(ai_report), title=f"Runsight Analysis: {file_path.name}", border_style="blue", expand=False))
                    CONSOLE.print(f"[dim]Artifacts saved to: {output_dir}[/dim]")
                
                progress.advance(task)
                
            except Exception as e:
                CONSOLE.print(f"[red]Error processing {file_path.name}: {e}[/red]")
                # CONSOLE.print(traceback.format_exc())

if __name__ == "__main__":
    main()

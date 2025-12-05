

import os
import sys
import argparse
import time
import json
from pathlib import Path
from typing import Dict, Any, List

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.markdown import Markdown
from dotenv import load_dotenv

import utils as core

# Load environment variables
load_dotenv()

CONSOLE = Console()

def save_artifacts(df: Any, metadata: Dict, metrics: Dict, file_path: Path):
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
    
    # Load System Prompt
    try:
        system_prompt = core.load_system_prompt('analysis.md')
    except Exception as e:
        CONSOLE.print(f"[bold red]Error loading prompt:[/bold red] {e}")
        return
    
    # Find files
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
                df, metadata = core.parse_fit_file(file_path)
                
                # Phase B: Metrics
                metrics = core.calculate_metrics(df, metadata)
                metrics['filename'] = file_path.name
                
                # Save Artifacts
                save_artifacts(df, metadata, metrics, file_path)
                
                # Cloud Upload (GCS) - REMOVED per user request
                # gcs_uri = core.upload_to_gcs(file_path)
                # if gcs_uri:
                #    metrics['gcs_uri'] = gcs_uri
                
                all_metrics.append(metrics)
                progress.advance(task)
                
            except Exception as e:
                CONSOLE.print(f"[red]Error processing {file_path.name}: {e}[/red]")

    if not all_metrics:
        CONSOLE.print("[red]No metrics generated.[/red]")
        return

    # Phase C: AI Report
    try:
        report = core.generate_ai_report(
            all_metrics, 
            system_prompt, 
            args.feedback, 
            consolidate=args.consolidate
        )
        
        CONSOLE.print(Panel(Markdown(report), title="Forensic Report", border_style="green"))
        
        # Save Report to MD
        if args.consolidate:
            report_path = Path("consolidated_report.md")
            report_path.write_text(report, encoding='utf-8')
            CONSOLE.print(f"[green]Report saved to {report_path}[/green]")
        elif len(files_to_process) == 1:
            report_path = files_to_process[0].with_name(f"{files_to_process[0].stem}_report.md")
            report_path.write_text(report, encoding='utf-8')
            CONSOLE.print(f"[green]Report saved to {report_path}[/green]")
            
        # Cloud Save (Firestore) - REMOVED per user request


    except Exception as e:
        CONSOLE.print(f"[bold red]AI Reporting Failed:[/bold red] {e}")

if __name__ == "__main__":
    main()

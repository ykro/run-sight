import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from google.cloud import storage, firestore, exceptions
from google.oauth2 import service_account
from rich.console import Console
from rich.panel import Panel

# Load environment variables
load_dotenv()

CONSOLE = Console()

def get_env_var(name, required=True):
    val = os.getenv(name)
    if not val and required:
        CONSOLE.print(f"[bold red]Error:[/bold red] Environment variable {name} not set.")
        sys.exit(1)
    return val

def test_gcs(project_id, bucket_name, credentials_path):
    CONSOLE.print(f"[bold blue]--- Cloud Storage (GCS) Setup ---[/bold blue]")
    
    try:
        storage_client = storage.Client(project=project_id)
        
        # Check if bucket exists
        try:
            bucket = storage_client.get_bucket(bucket_name)
            CONSOLE.print(f"[green]✓ Bucket '{bucket_name}' exists.[/green]")
        except exceptions.NotFound:
            CONSOLE.print(f"[yellow]! Bucket '{bucket_name}' not found. Attempting to create...[/yellow]")
            try:
                bucket = storage_client.create_bucket(bucket_name, location="US")
                CONSOLE.print(f"[green]✓ Bucket '{bucket_name}' created successfully.[/green]")
            except Exception as e:
                CONSOLE.print(f"[bold red]x Failed to create bucket:[/bold red] {e}")
                return False

        # Smoke Test: Upload, Read, Delete
        blob_name = "setup_smoke_test.txt"
        blob = bucket.blob(blob_name)
        
        CONSOLE.print("[dim]Performing smoke test (Write/Read/Delete)...[/dim]")
        
        # Write
        blob.upload_from_string("This is a smoke test file from setup_gcs.py")
        
        # Read
        content = blob.download_as_text()
        if "smoke test" not in content:
            CONSOLE.print(f"[bold red]x Content mismatch in GCS file.[/bold red]")
            return False
            
        # Delete
        blob.delete()
        CONSOLE.print(f"[green]✓ GCS Smoke Test Passed.[/green]")
        return True

    except Exception as e:
        CONSOLE.print(f"[bold red]x GCS Error:[/bold red] {e}")
        return False

def test_firestore(project_id, db_name, credentials_path):
    CONSOLE.print(f"\n[bold blue]--- Firestore Setup (DB: {db_name}) ---[/bold blue]")
    
    try:
        db = firestore.Client(project=project_id, database=db_name)
        
        # Smoke Test: Write, Read, Delete
        collection_name = "setup_smoke_tests"
        doc_id = "test_doc"
        doc_ref = db.collection(collection_name).document(doc_id)
        
        CONSOLE.print("[dim]Performing smoke test (Write/Read/Delete)...[/dim]")
        
        # Write
        test_data = {"timestamp": time.time(), "status": "testing"}
        doc_ref.set(test_data)
        
        # Read
        doc = doc_ref.get()
        if not doc.exists:
            CONSOLE.print(f"[bold red]x Failed to read back document from Firestore.[/bold red]")
            return False
            
        # Delete
        doc_ref.delete()
        CONSOLE.print(f"[green]✓ Firestore Smoke Test Passed.[/green]")
        return True

    except Exception as e:
        CONSOLE.print(f"[bold red]x Firestore Error:[/bold red] {e}")
        CONSOLE.print("[yellow]Note: If the database does not exist, you must create it via the Google Cloud Console or gcloud CLI: 'gcloud firestore databases create'[/yellow]")
        return False

def main():
    CONSOLE.print(Panel("Runsight GCP Setup & Verification", style="bold green"))
    
    project_id = get_env_var("GCP_PROJECT_ID")
    bucket_name = get_env_var("GCS_BUCKET_NAME")
    firestore_db = os.getenv("FIRESTORE_DB_NAME", "(default)")
    creds_path = get_env_var("GOOGLE_APPLICATION_CREDENTIALS")
    
    # Verify credentials file exists
    if not os.path.exists(creds_path):
        CONSOLE.print(f"[bold red]Error:[/bold red] Credentials file not found at {creds_path}")
        sys.exit(1)
        
    # Check services
    gcs_ok = test_gcs(project_id, bucket_name, creds_path)
    firestore_ok = test_firestore(project_id, firestore_db, creds_path)
    
    CONSOLE.print("\n")
    if gcs_ok and firestore_ok:
        CONSOLE.print(Panel("✅ All Cloud Checks Passed. System Ready.", style="green"))
    else:
        CONSOLE.print(Panel("❌ One or more checks failed. See logs above.", style="red"))

if __name__ == "__main__":
    main()

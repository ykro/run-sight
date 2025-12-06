# Runsight (Forensic Athletic Analysis CLI)

Runsight is a high-performance Python CLI tool designed to perform deep forensic analysis on endurance activity telemetry data (Garmin `.fit` files) and generate elite coaching reports using Generative AI (Google Gemini V2).

## Features

- **Advanced Fit Parsing:** Extracts high-fidelity raw data from fit files using `fitdecode`.

## Web App Demo
You can try the hosted version here:
[Runsight Web App](https://runsight-web-504954692234.us-central1.run.app/)

- **Forensic Metrics Engine:** Calculates specialized metrics not found in standard apps:
  - **Continuity Index:** True moving time vs stop time analysis.
  - **Uphill Biomechanics:** Detects over-striding and mechanical inefficiencies.
  - **Physiological Cost:** Cardiac cost (Beats/km) and Power Efficiency (Watts/Beat).
  - **Durability Index:** Tracks efficiency factor decay over time.
  - **Context Detection:** Automatically detects Trail vs Road.
- **AI-Powered Reporting:** Generates detailed, professional coaching reports in Spanish using Google's `gemini-3-pro-preview`.
- **Consolidated Analysis:** Can process directories recursively and generate aggregate trend reports.

## Prerequisites

- Python 3.12+ recommended
- `uv` package manager (optional but recommended for zero-setup execution)
- A Google Gemini API Key

## Setup

1. **Install `uv` (if not already installed):**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd runsight
   ```

3. **Configure Environment:**
   Copy `.env.example` to `.env` and add your Gemini API Key.
   ```bash
   cp .env.example .env
   # Edit .env and paste your key
   ```

## Usage

You can run the script directly with `uv`:

```bash
# Process a single file
uv run main.py path/to/activity.fit

# Process a directory recursively
uv run main.py path/to/activities/ --recursive

# Process a directory and generate a consolidated trend report
uv run main.py path/to/activities/ --recursive --consolidate

# Add user feedback to the context
uv run main.py path/to/activity.fit --feedback "Feeling tired from yesterday's intervals"
```

## Flags

- `path`: (Required) Path to .fit file or directory.
- `-r`, `--recursive`: Search provided directory recursively for .fit files.
- `-c`, `--consolidate`: Generate a single collected report for all processed files.
- `--feedback`: String containing subjective user feedback to influence the AI analysis.

## Output

For each processed file `activity.fit`, the tool generates:
- `activity.csv`: Raw time-series data.
- `activity.json`: Metadata.
- `activity_metrics.json`: Calculated forensic metrics.

The AI report is printed to stdout (terminal).

# Contributing to MINT

MINT is an experimental detector project, so contributions that improve observability, calibration, artifact rejection, camera compatibility, and reproducibility are especially welcome.

## Development

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install pytest
python -m pytest
```

Please avoid committing local capture output, event crops, calibration summaries, or private experiment notes.

## Scientific Claims

Use careful language in issues, pull requests, and docs. A single dark-covered webcam can identify candidate transient sensor events, but it does not confirm a cosmic ray by itself. Verification detection with multiple sensors is the likely path toward stronger confidence.

# PD Voice Analysis

Local demo application for the Parkinson's voice-screening prototype. It has a React/Vite frontend and a FastAPI backend. This is a screening demo, not a diagnostic tool.

## Run on a new Windows laptop (VS Code)

### 1. Install prerequisites

- [Git](https://git-scm.com/download/win)
- [Python 3.11](https://www.python.org/downloads/release/python-3110/) — do not use Python 3.14; the audio dependencies do not support it yet.
- [Node.js LTS](https://nodejs.org/)
- VS Code, plus the Python extension (recommended)

### 2. Clone and open the project

In PowerShell:

```powershell
git clone https://github.com/KinshukP/pd-voice-analysis.git
cd pd-voice-analysis
code .
```

### 3. Copy the model files separately

The `models/` folder is intentionally excluded from GitHub. Copy it from the demo laptop to the new laptop using a private USB drive or other secure transfer, then place it in the project root:

```text
pd-voice-analysis/
  models/
    confidence_v2.joblib
    detector_v2.joblib
    detector_v2_concepts.joblib
    detector_v2_data.joblib
    detector_v2_shap.joblib
```

Do not commit or upload these model files to the public repository.

### 4. Install the Python packages

From the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents activation, run this once for the current terminal and then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

In VS Code, select `.venv\Scripts\python.exe` as the Python interpreter when prompted.

### 5. Install the frontend packages

```powershell
cd frontend
npm ci
cd ..
```

`npm ci` installs exactly the dependency versions recorded in `frontend/package-lock.json`.

## Start the demo

Open **two** PowerShell terminals in the project root.

In terminal 1, start the API:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn pdvoice.api:app --reload
```

For a presentation with explicitly labelled scripted demo results, set demo mode before starting the API:

```powershell
.\.venv\Scripts\Activate.ps1
$env:PDVOICE_DEMO="1"
python -m uvicorn pdvoice.api:app --reload
```

In terminal 2, start the website:

```powershell
cd frontend
npm run dev
```

Open the URL printed by Vite, normally [http://localhost:5173](http://localhost:5173). Confirm the backend is ready at [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health). A `"model_loaded": true` response means the model files were found.

Stop either server with `Ctrl+C`.

## Sync the new laptop later

When code changes are pushed to GitHub, use:

```powershell
git pull
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cd frontend
npm ci
```

Do not use `npm update` for demos: `npm ci` keeps frontend package versions in sync with the repository. If model files change, copy the updated `models/` folder separately as in step 3.

## Wi-Fi demo (optional)

The normal `localhost` URL works only on the laptop running the app. The app has no login/authentication, so do not expose it on public Wi-Fi.

```powershell
# API terminal (replace 10.0.0.244 with this laptop's Wi-Fi IPv4 address)
$env:PDVOICE_CORS_ORIGINS="http://10.0.0.244:5173"
python -m uvicorn pdvoice.api:app --reload --host 0.0.0.0 --port 8000

# Website terminal
cd frontend
npm run dev -- --host 0.0.0.0
```

Other devices can then open `http://10.0.0.244:5173/?api=http://10.0.0.244:8000`. Allow the Windows Firewall prompt only on private networks. Mobile browser microphone access generally requires HTTPS, so on Wi-Fi use file upload or deploy with HTTPS for recording.

<div align="center">
  <img src="https://github.com/chandravadhannadella/TrafficDensityPrediction/blob/main/Screenshots/trafficIQ-logo.png?raw=true" alt="TrafficIQ logo" width="112" />

  <h1>TrafficIQ</h1>
  <h3>AI-Powered Traffic Density Prediction</h3>

  <p>
    <strong>Turn road images and videos into traffic insights.</strong><br />
    A computer-vision prototype for vehicle detection, traffic-density estimation, and time-based video analysis.
  </p>

  <p>
    <a href="https://github.com/chandravadhannadella/TrafficDensityPrediction">
      <img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github" alt="GitHub repository" />
    </a>
    <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+" />
    <img src="https://img.shields.io/badge/Flask-Web%20App-000000?style=for-the-badge&logo=flask" alt="Flask web app" />
    <img src="https://img.shields.io/badge/Computer%20Vision-YOLOv8-7B61FF?style=for-the-badge" alt="YOLOv8 computer vision" />
    <img src="https://img.shields.io/badge/License-MIT-2EA44F?style=for-the-badge" alt="MIT License" />
  </p>

  <p>
    <a href="#-project-preview">Preview</a> •
    <a href="#-features">Features</a> •
    <a href="#-getting-started">Getting Started</a> •
    <a href="#-api-reference">API</a> •
    <a href="#-project-structure">Project Structure</a>
  </p>
</div>

---

> [!IMPORTANT]
> **Research prototype:** Density labels are proxy labels derived from vehicle counts, not independently verified ground-truth traffic annotations. Predictions are experimental and must not be used for safety-critical or operational traffic-management decisions.

## 🚦 Project Overview

**TrafficIQ** is a Flask-based computer-vision application that estimates traffic density from road images and videos. It combines **YOLOv8n vehicle detection**, saved machine-learning classifiers, OpenCV-based media processing, and an interactive dashboard to present vehicle counts and traffic-density predictions.

The application supports two separate inference workflows: a **V4-Image model** for image analysis and a **V3 pipeline** for video inference. Video analysis samples frames and aggregates features over time windows to show how estimated density changes throughout a clip.

## 🖥️ Project Preview

<div align="center">
  <img src="Screenshots/Screenshot%202026-10-05%20201949.png" alt="TrafficIQ dashboard with image and video analysis options" width="100%" />
  <em>TrafficIQ dashboard — choose image analysis or video analysis from one workspace.</em>
</div>

## ✨ Features

<table>
  <tr>
    <td width="50%">
      <h3>🖼️ Image Analysis</h3>
      Upload a road image and view detected vehicles, vehicle counts, predicted density, confidence, and annotated output.
    </td>
    <td width="50%">
      <h3>🎞️ Video Analysis</h3>
      Analyze traffic across time windows and inspect density changes, per-window predictions, and video output.
    </td>
  </tr>
  <tr>
    <td>
      <h3>🚗 Vehicle Detection</h3>
      Use YOLOv8n to detect supported classes: cars, buses, trucks, motorcycles, and bicycles.
    </td>
    <td>
      <h3>📈 Traffic Analytics</h3>
      Review vehicle distributions, confidence scores, density timelines, and time-window summaries.
    </td>
  </tr>
  <tr>
    <td>
      <h3>🧭 Tracking & Features</h3>
      Track detections across sampled video frames and aggregate temporal features for video inference.
    </td>
    <td>
      <h3>🔌 REST API</h3>
      Upload media, start processing, check job status, and retrieve results programmatically.
    </td>
  </tr>
</table>

## 🖼️ Application Gallery

<div align="center">
  <h3>Dashboard</h3>
  <img src="Screenshots/Screenshot%202026-10-05%20201949.png" alt="TrafficIQ dashboard" width="100%" />

  <h3>Analysis Workspace</h3>
  <img src="Screenshots/Screenshot%202026-10-05%20202353.png" alt="TrafficIQ analysis workspace" width="100%" />

  <h3>Image Analysis Results</h3>
  <img src="Screenshots/Screenshot%202026-10-05%20201806.png" alt="TrafficIQ image analysis with vehicle detections" width="100%" />
  <img src="Screenshots/Screenshot%202026-10-05%20201824.png" alt="TrafficIQ image vehicle breakdown and summary" width="100%" />

  <h3>Video Analysis Results</h3>
  <img src="Screenshots/Screenshot%202026-10-05%20201911.png" alt="TrafficIQ video analysis player" width="100%" />
  <img src="Screenshots/Screenshot%202026-10-05%20201935.png" alt="TrafficIQ video analytics and timeline" width="100%" />
</div>

> The gallery uses the screenshot filenames currently documented in this repository. If a screenshot is renamed, update its path here too.

## ⚙️ How It Works

```text
Road Image / Video
        │
        ▼
  Upload via Dashboard or REST API
        │
        ▼
  YOLOv8n Vehicle Detection
        │
        ├── Image pipeline ──► V4-Image classifier
        │
        └── Video pipeline ──► Frame sampling + tracking
                               + time-window features
                                      │
                                      ▼
                           Saved density classifier
                                      │
                                      ▼
                     LOW / MEDIUM / HIGH prediction
                                      │
                                      ▼
                   Counts • Confidence • Visual results
```

### Density Categories

| Category | Meaning |
|---|---|
| `LOW` | Lower observed vehicle count |
| `MEDIUM` | Moderate observed vehicle count |
| `HIGH` | Higher observed vehicle count |

The image-model metadata documents proxy thresholds of **8 or fewer vehicles for LOW**, **9–15 for MEDIUM**, and **16 or more for HIGH**. These thresholds are experimental proxies, not validated real-world traffic-density standards.

## 🧰 Technology Stack

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white" alt="Flask" />
  <img src="https://img.shields.io/badge/YOLOv8-7B61FF?style=flat-square" alt="YOLOv8" />
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white" alt="OpenCV" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white" alt="pandas" />
  <img src="https://img.shields.io/badge/HTML5-E34F26?style=flat-square&logo=html5&logoColor=white" alt="HTML5" />
  <img src="https://img.shields.io/badge/CSS3-1572B6?style=flat-square&logo=css3&logoColor=white" alt="CSS3" />
  <img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black" alt="JavaScript" />
  <img src="https://img.shields.io/badge/pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white" alt="pytest" />
</p>

| Technology | Role |
|---|---|
| Python + Flask | Web application, routes, and REST API |
| Ultralytics YOLOv8n | Vehicle detection |
| OpenCV | Image and video processing |
| scikit-learn | Traffic-density classification |
| NumPy + pandas | Feature preparation and numerical processing |
| HTML, CSS, JavaScript | Dashboard and browser interactions |
| pytest | Automated tests |

## 🚀 Getting Started

### Prerequisites

- Python **3.10 or newer**
- Git and pip
- Internet access to install dependencies
- **4 GB RAM or more recommended** for inference
- Optional CUDA-compatible GPU with a matching PyTorch build

### 1. Clone the repository

```bash
git clone https://github.com/chandravadhannadella/TrafficDensityPrediction.git
cd TrafficDensityPrediction
```

### 2. Create a virtual environment

A virtual environment keeps TrafficIQ's Python packages separate from other projects.

**Windows — PowerShell**

```powershell
python -m venv .venv
```

If `python` is not recognized, try:

```powershell
py -m venv .venv
```

**macOS / Linux**

```bash
python3 -m venv .venv
```

### 3. Activate the environment

**Windows — PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation for this session, run the following and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

**Windows — Command Prompt**

```cmd
.venv\Scripts\activate.bat
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

When activated, the terminal normally shows `(.venv)`.

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If installation fails due to a PyTorch or CUDA compatibility issue, follow the official [PyTorch installation guide](https://pytorch.org/get-started/locally/) and select a build that matches your operating system and hardware.

### 5. Verify model artifacts

The repository includes `yolov8n.pt`. The saved classifier artifacts and any required metadata must also exist at the paths expected by the image and video pipelines. If an artifact is missing, its inference workflow may fail to load.

### 6. Start TrafficIQ

From the project root, with `.venv` activated:

```bash
python app.py
```

Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.

> Flask's built-in server is intended for local development. Use an appropriately configured production WSGI server before deploying publicly.

## 🧹 Virtual Environment & `__pycache__`

You do **not** need to manually create or upload `__pycache__` folders.

| Item | What it is | Upload to GitHub? |
|---|---|---|
| `.venv/` | Local Python environment and installed packages | No |
| `__pycache__/` | Automatically generated Python bytecode cache | No |
| `*.pyc` | Compiled Python bytecode files | No |
| `yolov8n.pt` | YOLOv8n detector weights used by the app | Included in this repository |
| Training videos | Local development/training media | Excluded when ignored by `.gitignore` |

Python creates `__pycache__` folders automatically when modules are imported. They are safe to leave on your computer and are not required in version control. If `.venv` is damaged, recreate it using the setup steps above; do not delete your source code or required model artifacts.

## 🎬 Supported Media

- **Images:** JPG, JPEG, PNG, BMP, WebP
- **Videos:** MP4, AVI, MOV, MKV
- **Configured upload limit:** 200 MB

Video processing can require significant memory and time, depending on the clip's resolution, length, and hardware. Large training videos are not included when excluded by `.gitignore`; use your own compatible media or obtain the required data separately.

## 🔌 API Reference

The Flask app documents the following endpoints on port `5000`:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | Check API availability and view the model disclaimer |
| `POST` | `/api/upload-image` | Upload an image using multipart field `image` |
| `POST` | `/api/upload` | Upload a video using multipart field `video` |
| `POST` | `/api/process-image/<job_id>` | Start image inference |
| `POST` | `/api/process/<job_id>` | Start video inference |
| `GET` | `/api/status/<job_id>` | Read job status and progress |
| `GET` | `/api/results/<job_id>` | Retrieve completed results |

### Example: upload an image

```bash
curl http://127.0.0.1:5000/api/health

curl -X POST \
  -F "image=@road.jpg" \
  http://127.0.0.1:5000/api/upload-image
```

Use the returned `job_id` to start processing and query the result:

```bash
curl -X POST http://127.0.0.1:5000/api/process-image/<job_id>
curl http://127.0.0.1:5000/api/status/<job_id>
curl http://127.0.0.1:5000/api/results/<job_id>
```

Replace `<job_id>` with the actual identifier returned by the upload response. For video, upload with `-F "video=@traffic.mp4"` to `/api/upload`, then call `/api/process/<job_id>`.

**Implementation note:** Video jobs run in a background thread within the Flask process, and job state is held in memory. Restarting the server clears active and completed job records.

## 🧪 Run Tests

With the virtual environment activated and dependencies installed:

```bash
python -m pytest tests/ -v
```

Run one test module:

```bash
python -m pytest tests/test_backend_api.py -v
```

Some tests may require saved model artifacts, sample media, or local configuration. Investigate individual test errors rather than assuming every failure means the entire application is broken.

## 📁 Project Structure

```text
TrafficDensityPrediction/
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── yolov8n.pt
├── backend/
│   ├── routes/       # Upload, processing, status, and result endpoints
│   ├── services/     # Image, video, detection, and prediction services
│   └── utils/        # Shared backend helpers
├── data/             # Datasets and local media (some files excluded)
├── model/
│   ├── detection/
│   ├── evaluation/
│   ├── inference/
│   ├── training/
│   └── utils/
├── outputs/           # Generated results (local)
├── Screenshots/       # README screenshots and TrafficIQ logo
├── scripts/           # Dataset and utility scripts
├── static/
│   ├── assets/
│   ├── CSS/
│   └── js/
├── templates/          # Flask HTML templates
└── tests/              # Automated tests
```

## ⚠️ Limitations

- Density classes are based on vehicle-count proxy labels, not independently verified traffic annotations.
- Detection quality depends on camera angle, lighting, occlusion, resolution, and model weights.
- Vehicle counts and tracking may be imperfect when vehicles overlap or leave the frame.
- Model confidence is not a guarantee of real-world accuracy.
- Video job state is stored in process memory and does not persist after server restarts.
- The Flask development server is not intended for production deployment.

## 🙌 Acknowledgments

TrafficIQ uses open-source technologies including [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), [OpenCV](https://opencv.org/), [scikit-learn](https://scikit-learn.org/), [Flask](https://flask.palletsprojects.com/), [NumPy](https://numpy.org/), and [pandas](https://pandas.pydata.org/).

Please review the license terms for third-party libraries, pretrained weights, datasets, and external assets before redistributing them.

## 📄 License

This project is licensed under the **MIT License**. See [`LICENSE`](LICENSE) for the full license text. Third-party packages, pretrained model weights, datasets, and other external assets remain subject to their own licenses and terms.

---

<div align="center">
  <img src="Screenshots/trafficIQ-logo.png" alt="TrafficIQ logo" width="56" />
  <p><strong>TrafficIQ</strong> · Turning visual traffic data into useful insights.</p>
  <sub>Built with Python, computer vision, and machine learning.</sub>
</div>

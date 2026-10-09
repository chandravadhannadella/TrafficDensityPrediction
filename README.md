# 🚦 TrafficIQ — AI-Powered Traffic Density Prediction

**TrafficIQ** is a computer-vision web application that estimates traffic density from road images and videos. It combines YOLOv8 vehicle detection, machine-learning classification, image and video processing, and an interactive Flask dashboard to analyze traffic conditions.

The application detects vehicles, extracts traffic-related features, predicts traffic density, and presents the results through annotated images, video analysis, vehicle counts, confidence scores, and time-based analytics.

> **Research prototype:** Density labels are based on proxy labels derived from vehicle counts, not independently verified ground-truth traffic annotations. Predictions are experimental and should not be used for safety-critical or operational traffic-management decisions.

## ✨ Features

- **Image-based analysis:** Upload a road image and view vehicle detections, vehicle counts, predicted density, and confidence.
- **Video-based analysis:** Process traffic videos and examine density predictions across time windows.
- **Vehicle detection:** Detect supported vehicle classes, including cars, buses, trucks, motorcycles, and bicycles, using YOLOv8n.
- **Vehicle tracking:** Track detections across sampled video frames to support temporal feature extraction.
- **Separate prediction pipelines:** Use a dedicated V4-Image model for image inference and a V3 model for video inference.
- **Traffic analytics:** Review per-window predictions, confidence scores, vehicle distributions, and traffic timelines.
- **Annotated outputs:** Generate visual outputs that help users inspect detected vehicles and analysis results.
- **Interactive dashboard:** Access image and video workflows through a Flask-based web interface.
- **REST API:** Upload media, initiate inference, check processing status, and retrieve results programmatically.
- **Automated tests:** Run the included tests to check API routes, detection, prediction, and video inference components.

## 🖥️ Application Screenshots

Place your screenshots in the existing `Screenshots/` folder. Use the filenames below or change the image paths to match your actual files.

### Dashboard

![TrafficIQ dashboard](https://github.com/chandravadhannadella/TrafficDensityPrediction/blob/main/Screenshots/Screenshot%202026-10-05%20201949.png?raw=true)

### Analysis Workspace

![TrafficIQ analysis workspace](https://github.com/chandravadhannadella/TrafficDensityPrediction/blob/main/Screenshots/Screenshot%202026-10-05%20202353.png?raw=true)

### Image Analysis Results

![TrafficIQ image analysis results](https://github.com/chandravadhannadella/TrafficDensityPrediction/blob/main/Screenshots/Screenshot%202026-10-05%20201806.png?raw=true)

![TrafficIQ vehicle breakdown and summary](https://github.com/chandravadhannadella/TrafficDensityPrediction/blob/main/Screenshots/Screenshot%202026-10-05%20201824.png?raw=true)

### Video Analysis Results

![TrafficIQ video analysis player](https://github.com/chandravadhannadella/TrafficDensityPrediction/blob/main/Screenshots/Screenshot%202026-10-05%20201911.png?raw=true)

![TrafficIQ video analytics and timeline](https://github.com/chandravadhannadella/TrafficDensityPrediction/blob/main/Screenshots/Screenshot%202026-10-05%20201935.png?raw=true)

> **Note:** GitHub displays an image only when the referenced file exists at the specified path. Replace these placeholder filenames with your actual screenshot filenames if they differ.

## ⚙️ How It Works

1. **Upload media:** The user uploads a traffic image or video through the dashboard or REST API.
2. **Detect vehicles:** YOLOv8n identifies supported vehicle classes in the input.
3. **Extract features:** The image pipeline prepares image-level features, while the video pipeline samples frames and aggregates features across time windows.
4. **Predict density:** The relevant saved classifier predicts a traffic-density category and returns associated confidence and vehicle-count information.
5. **Display results:** The dashboard presents predictions, vehicle counts, annotated outputs, and video analytics.

Image and video inference use separate model pipelines. The image pipeline uses the V4-Image model, while the video pipeline uses the V3 model and its associated artifacts.

### Traffic Density Categories

| Category | Description |
|---|---|
| `LOW` | Lower observed vehicle count |
| `MEDIUM` | Moderate observed vehicle count |
| `HIGH` | Higher observed vehicle count |

The image-model metadata documents proxy thresholds of 8 or fewer vehicles for `LOW`, 9–15 for `MEDIUM`, and 16 or more for `HIGH`. These are experimental proxy-label thresholds, not validated real-world traffic-density standards.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application logic and machine-learning workflows |
| Flask | Web application and REST API |
| Ultralytics YOLOv8n | Vehicle detection |
| OpenCV | Image processing and video analysis |
| scikit-learn | Traffic-density classification |
| NumPy and pandas | Numerical operations and feature preparation |
| HTML5 | Web page structure |
| CSS3 | Dashboard styling and responsive layout |
| JavaScript | Frontend interactions and API communication |
| pytest | Automated testing |

## 📋 Requirements

Before installing the project, make sure you have:

- Python 3.10 or newer.
- pip, the Python package installer.
- Git, for cloning the repository.
- An internet connection for downloading dependencies.
- At least 4 GB of RAM recommended for inference; additional memory may improve performance.
- Optional CUDA-compatible GPU for acceleration, with a compatible PyTorch installation.

**Important:** Dependency compatibility can vary by Python version, operating system, and CUDA configuration. Use a compatible PyTorch build if the requirements file does not match your system.

## 🚀 Installation and Setup

Follow these steps to download, configure, and run TrafficIQ on your computer.

### 1. Clone the Repository

Open a terminal and run:

```bash
git clone https://github.com/chandravadhannadella/TrafficDensityPrediction.git
cd TrafficDensityPrediction
```

### 2. Verify Your Python Installation

Check that Python and pip are available:

```bash
python --version
python -m pip --version
```

If `python` is not recognized on Windows, try:

```powershell
py --version
```

Use a compatible Python version before proceeding.

### 3. Create a Virtual Environment

A virtual environment isolates this project's Python dependencies from other projects on your computer.

**Windows — PowerShell:**

```powershell
python -m venv .venv
```

If the `python` command is unavailable but the Python launcher works, use:

```powershell
py -m venv .venv
```

**macOS or Linux:**

```bash
python3 -m venv .venv
```

This creates a local `.venv/` directory in the project root. It should not be uploaded to GitHub.

### 4. Activate the Virtual Environment

**Windows — PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation because of its execution policy, run the following command in the current terminal session and retry activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

**Windows — Command Prompt:**

```cmd
.venv\Scripts\activate.bat
```

**macOS or Linux:**

```bash
source .venv/bin/activate
```

After activation, your terminal will usually show `(.venv)` at the beginning of the command prompt.

### 5. Upgrade pip and Install Dependencies

With the virtual environment activated, run:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Wait until installation completes. Some computer-vision and machine-learning dependencies may take several minutes to install.

If installation fails because of a PyTorch or CUDA compatibility issue, consult the official [PyTorch installation guide](https://pytorch.org/get-started/locally/) and install a compatible build for your system.

### 6. Verify Required Model Files

TrafficIQ depends on its detector weights and saved classifier artifacts.

The repository includes `yolov8n.pt`. Ensure that the classifier files and any metadata required by the image and video pipelines are also present in the locations expected by the application.

If a required model artifact is missing, the corresponding inference pipeline may fail to initialize or make predictions. Check the model-loading paths in the project's source code before running the application.

### 7. Run the Application

From the repository root, with the virtual environment activated, execute:

```bash
python app.py
```

Open your browser and visit:

**http://127.0.0.1:5000**

The application uses Flask's development server by default. Do not expose this development server directly to the public internet.

## 🧹 Virtual Environments and `__pycache__`

Python automatically creates `__pycache__/` directories containing compiled bytecode files when modules are imported.

These files are normal and do not need to be created manually.

- **`.venv/`:** Contains the project's isolated Python environment and installed packages.
- **`__pycache__/`:** Contains automatically generated Python bytecode files.
- **`*.pyc`:** Individual compiled Python bytecode files.

These files should generally remain on your computer and be excluded from GitHub using `.gitignore`.

You do **not** need to delete `__pycache__` folders before running the application. Python recreates them when needed.

If the virtual environment becomes corrupted, deactivate it, remove the local `.venv/` directory, and recreate it using the installation steps above. Do not delete your source code, datasets, model artifacts, or generated results when doing this.

## ▶️ Using the Application

1. Start the Flask application.
2. Open the dashboard in your browser.
3. Choose the image or video analysis workflow.
4. Upload a supported road image or traffic video.
5. Start processing and wait for the inference to complete.
6. Review the predicted density, vehicle counts, confidence, and visual outputs.
7. For videos, inspect the time-window predictions and traffic timeline.

### Supported Media Formats

**Images:** JPG, JPEG, PNG, BMP, and WebP.

**Videos:** MP4, AVI, MOV, and MKV.

The application is configured with an upload limit of 200 MB. Large video files may require substantial memory and processing time.

The training videos used during development are not included in the GitHub repository when excluded by `.gitignore`. Users should provide their own compatible sample media or obtain the necessary data separately.

## 🔌 REST API

The Flask application exposes endpoints for health checks, media uploads, processing, status checks, and result retrieval.

The following endpoints are documented by the project:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | Check API availability and view the model disclaimer |
| `POST` | `/api/upload-image` | Upload an image using the `image` multipart field |
| `POST` | `/api/upload` | Upload a video using the `video` multipart field |
| `POST` | `/api/process-image/<job_id>` | Start image inference |
| `POST` | `/api/process/<job_id>` | Start video inference |
| `GET` | `/api/status/<job_id>` | Check job status and progress |
| `GET` | `/api/results/<job_id>` | Retrieve completed results |

### Example: Image Analysis

Check API availability:

```bash
curl http://127.0.0.1:5000/api/health
```

Upload an image:

```bash
curl -X POST \
  -F "image=@road.jpg" \
  http://127.0.0.1:5000/api/upload-image
```

Use the returned `job_id` in the processing request:

```bash
curl -X POST \
  http://127.0.0.1:5000/api/process-image/<job_id>
```

Check processing status:

```bash
curl http://127.0.0.1:5000/api/status/<job_id>
```

Retrieve completed results:

```bash
curl http://127.0.0.1:5000/api/results/<job_id>
```

Replace `<job_id>` with the actual identifier returned by the application.

### Example: Video Analysis

Upload a video:

```bash
curl -X POST \
  -F "video=@traffic.mp4" \
  http://127.0.0.1:5000/api/upload
```

Use the returned job identifier to start video processing:

```bash
curl -X POST \
  http://127.0.0.1:5000/api/process/<job_id>
```

Check the job status and retrieve its results using the same status and results endpoints shown above.

> **API behavior:** Video processing runs in a background thread within the Flask process. Job state is held in memory, so restarting the server clears active and completed job records.

## 📊 Prediction Outputs

### Image Analysis

Image results can include:

- Predicted traffic-density category.
- Classifier confidence or score.
- Vehicle counts by supported class.
- Class probabilities.
- Model version information.
- Annotated image information.

### Video Analysis

Video results can include:

- Overall predicted density and confidence.
- Number of analyzed time windows.
- Per-window density predictions.
- Traffic timeline and associated analytics.
- Original and annotated video filenames.

Confidence values represent the classifier's output, not a guarantee of real-world accuracy.

## 📁 Project Structure

The following is a high-level guide to the repository. Individual model artifacts and supporting files may vary with the selected model pipeline.

```text
TrafficDensityPrediction/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── yolov8n.pt
├── backend/
│   ├── routes/
│   │   ├── prediction_routes.py
│   │   └── upload_routes.py
│   ├── services/
│   │   ├── detection_service.py
│   │   ├── image_service.py
│   │   ├── prediction_service.py
│   │   └── video_service.py
│   └── utils/
├── data/
│   └── datasets and local media
├── model/
│   ├── detection/
│   ├── evaluation/
│   ├── inference/
│   ├── training/
│   └── utils/
├── outputs/
├── Screenshots/
├── scripts/
├── static/
│   ├── assets/
│   ├── CSS/
│   └── js/
├── templates/
│   ├── analytics.html
│   ├── dashboard.html
│   └── index.html
└── tests/
    ├── test_backend_api.py
    ├── test_detector.py
    ├── test_prediction.py
    └── additional test modules
```

### Key Directories

- **`backend/`** — Flask route handlers, processing services, and shared backend utilities.
- **`data/`** — Datasets and local media used for development and evaluation. Large training videos may be excluded from version control.
- **`model/`** — Detection, inference, tracking, training, evaluation, and supporting machine-learning components.
- **`outputs/`** — Generated analysis outputs and other local results.
- **`Screenshots/`** — Screenshots used to document the application.
- **`scripts/`** — Dataset-generation and utility scripts.
- **`static/`** — Frontend CSS, JavaScript, and static assets.
- **`templates/`** — Flask HTML templates.
- **`tests/`** — Automated tests for backend APIs and model pipelines.

## 🧪 Running Tests

Activate the virtual environment and ensure dependencies are installed. From the project root, run:

```bash
python -m pytest tests/ -v
```

To run a specific test module:

```bash
python -m pytest tests/test_backend_api.py -v
```

Some tests may require model artifacts, sample media, or additional local configuration. A failed test should be investigated using its error output rather than assumed to indicate an application-wide failure.

## ⚙️ Configuration and Development

Video sampling and feature-extraction settings are defined in the project's pipeline configuration. The documented defaults include:

- Detection confidence threshold: `0.25`
- Frame sampling interval: every `30` frames
- Video analysis window: `10` seconds

Verify the configuration file and current source code before changing these values, because inference behavior can depend on the selected model pipeline.

### Recommended Development Workflow

1. Create and activate `.venv`.
2. Install dependencies from `requirements.txt`.
3. Keep local datasets and generated media separate from source code.
4. Run the Flask application and validate image and video workflows.
5. Run the relevant tests after code changes.
6. Commit source-code changes to Git without including virtual environments, cache files, secrets, or unnecessary large media.

## ⚠️ Limitations

- Density labels are derived from proxy vehicle-count thresholds rather than independently verified ground truth.
- Detection performance depends on camera angle, lighting, occlusion, resolution, and model weights.
- Vehicle counts and tracking results may be imperfect when vehicles overlap or leave the frame.
- Classifier confidence is not equivalent to calibrated real-world certainty.
- Video job state is stored in process memory and does not persist across server restarts.
- The included Flask development server is not intended for production deployment.
- Large datasets and training videos may be excluded from GitHub to keep the repository manageable.

## 🙌 Acknowledgments

TrafficIQ uses the following open-source technologies:

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — object detection.
- [OpenCV](https://opencv.org/) — computer vision and video processing.
- [scikit-learn](https://scikit-learn.org/) — machine-learning utilities and classification.
- [Flask](https://flask.palletsprojects.com/) — web application framework.
- [NumPy](https://numpy.org/) and [pandas](https://pandas.pydata.org/) — data processing.

Please review the relevant licenses and usage terms for third-party packages, pretrained weights, and datasets before redistributing them.

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for the complete license text.

The license applies to the project materials that the repository owner has the right to license. Third-party libraries, pretrained model weights, datasets, and other external assets remain subject to their respective licenses and terms.

**TrafficIQ — Exploring traffic density estimation through computer vision and machine learning.**

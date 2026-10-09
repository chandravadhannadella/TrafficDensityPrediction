# TrafficIQ: Traffic Density Prediction

TrafficIQ is a computer-vision application for estimating traffic density from road images and videos. It combines YOLOv8 vehicle detection with scikit-learn classifiers and presents the results in a Flask dashboard.

> **Research prototype:** Density labels are based on proxy labels derived from vehicle counts. They are not verified ground truth and should not be used for safety-critical or operational traffic-management decisions.

## Features

- Analyze a road image and view an annotated image, vehicle counts, density class, and confidence.
- Analyze a video in time windows and review per-window predictions, a traffic timeline, confidence, and video playback.
- Detect cars, buses, trucks, motorcycles, and bicycles with YOLOv8n.
- Track detected vehicles across sampled video frames and aggregate features over time windows.
- Use a dedicated V4-Image model for images and a V3 model for video inference.
- Access the same upload and prediction workflow through the REST API.

## Application Screenshots

Add your screenshot files under `docs/images/`, or change the paths below to match where you store them. The filenames are placeholders for the screenshots supplied with this project.

### Dashboard

![TrafficIQ dashboard](docs/images/dashboard.png)

### Analysis Workspace and Processing

![TrafficIQ analysis workspace while processing an upload](docs/images/analysis-processing.png)

### Image Results

![TrafficIQ image analysis result with vehicle detections](docs/images/image-results.png)

![TrafficIQ image result with vehicle breakdown and summary](docs/images/image-breakdown.png)

### Video Results

![TrafficIQ video analysis player](docs/images/video-player.png)

![TrafficIQ video metrics, timeline, and window analysis](docs/images/video-analytics.png)

## How It Works

1. Upload an image or video through the dashboard or API.
2. YOLOv8n detects supported vehicle classes. Video inference samples frames and uses centroid tracking to support temporal features.
3. The pipeline prepares image-level features or aggregates video features into time windows.
4. A saved classifier predicts `LOW`, `MEDIUM`, or `HIGH` density and returns confidence and related counts.
5. The dashboard displays the prediction and generated visual outputs.

Image and video predictions use separate model pipelines. The image pipeline defaults to `v4_image`; the video pipeline loads V3 artifacts.

## Technology

- Python and Flask for the web application and REST API
- Ultralytics YOLOv8n and OpenCV for vehicle detection and video processing
- scikit-learn, NumPy, and pandas for feature preparation and classification
- HTML, CSS, and JavaScript for the dashboard

## Requirements

- Python 3.10 or newer
- pip
- At least 4 GB RAM recommended for inference
- Optional CUDA-compatible GPU and matching PyTorch installation for GPU acceleration

## Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
python -m venv .venv
```

Activate the environment and install dependencies.

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The repository includes `yolov8n.pt` and the saved classifier artifacts under `model/saved_models/`. Confirm those files are included in your GitHub repository; inference will fail if required weights or model files are absent. PyTorch installation requirements can vary by operating system and CUDA version; use the appropriate PyTorch build if the pinned requirements do not match your machine.

## Run Locally

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000). The application starts with Flask's development server; use a production WSGI server and appropriate deployment configuration before exposing it publicly.

The upload limit is 200 MB. Supported image formats are JPG/JPEG, PNG, BMP, and WebP. Supported video formats are MP4, AVI, MOV, and MKV.

## REST API

All endpoints are served by the Flask application on port `5000` by default.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Check API availability and view the model disclaimer |
| `POST` | `/api/upload-image` | Upload an image using multipart field `image` |
| `POST` | `/api/upload` | Upload a video using multipart field `video` |
| `POST` | `/api/process-image/<job_id>` | Start image inference |
| `POST` | `/api/process/<job_id>` | Start video inference |
| `GET` | `/api/status/<job_id>` | Read the job status and progress |
| `GET` | `/api/results/<job_id>` | Retrieve completed results |

Example image workflow:

```bash
curl http://localhost:5000/api/health

curl -X POST -F "image=@road.jpg" \
  http://localhost:5000/api/upload-image
```

Use the returned `job_id` to start processing, check status, and retrieve results:

```bash
curl -X POST http://localhost:5000/api/process-image/<job_id>
curl http://localhost:5000/api/status/<job_id>
curl http://localhost:5000/api/results/<job_id>
```

For video, use `-F "video=@traffic.mp4"` with `/api/upload`, then call `/api/process/<job_id>`. Processing runs in a background thread in the Flask process. Job state is held in memory, so restarting the server clears active and completed job records.

## Prediction Outputs

Image results include the predicted density, confidence, vehicle counts by class, class probabilities, model version, and annotated image information. Video results include the overall density and confidence, the number of analyzed windows, per-window predictions, and the original and annotated video filenames.

Density classes are `LOW`, `MEDIUM`, and `HIGH`. The image model metadata documents proxy thresholds of 8 or fewer vehicles for LOW, 9-15 for MEDIUM, and 16 or more for HIGH. Treat model confidence as the classifier's score, not as a guarantee of real-world accuracy.

## Repository Layout

```text
.
├── app.py                       # Flask application entry point
├── backend/
│   ├── routes/                  # Upload, processing, status, and result endpoints
│   ├── services/                # Image and video job processing
│   └── utils/                   # Shared backend helpers
├── model/
│   ├── inference/               # Image and video predictors and model paths
│   ├── detection/               # Detection and tracking components
│   ├── prediction/              # Feature extraction and prediction utilities
│   ├── training/                # Model training code
│   └── saved_models/            # Classifiers, scalers, and feature metadata
├── data/                        # Datasets, processed media, and results
├── outputs/                     # Generated outputs
├── scripts/                     # Dataset and utility scripts
├── static/                      # CSS, JavaScript, and frontend assets
├── templates/                   # Flask templates
├── tests/                       # Automated tests
├── uploads/                     # Uploaded user media
├── requirements.txt
└── yolov8n.pt                    # YOLOv8n detector weights
```

## Configuration and Development

Video sampling and feature settings are defined in `model/pipeline_config.py`, including the detection confidence threshold, frame sampling interval, time-window length, and YOLO input size. The current defaults include a confidence threshold of `0.25`, sampling every 30 frames, and 10-second video windows.

Run the test suite from the repository root:

```bash
python -m pytest tests/ -v
```

## Limitations

- Density labels are proxy labels based on counts, not independently verified traffic-density annotations.
- Detection quality depends on camera angle, lighting, occlusion, image resolution, and the detector weights.
- Video jobs and their results are stored in process memory and do not persist across application restarts.
- The included Flask server is for local development, not a production deployment.

## Acknowledgments

This project uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), [OpenCV](https://opencv.org/), [scikit-learn](https://scikit-learn.org/), and [Flask](https://flask.palletsprojects.com/).

## License

No `LICENSE` file is currently present in the repository. Add a license file and update this section before publishing if you intend to grant others permission to use, modify, or distribute the project.

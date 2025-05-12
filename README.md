# GANSDOCTOR API

![GANSDOCTOR Logo](https://storage.googleapis.com/gansdoctor_skripsi/Logo%20GansDoctor.png) 

A Deep Learning-based API for detecting AI-generated faces using TensorFlow and FastAPI.

## 🔍 Overview

GANSDOCTOR API is a powerful API that analyzes facial images to determine whether they are AI-generated or real. This solution leverages deep learning models to provide accurate detection of synthetic faces produced by GANs (Generative Adversarial Networks).

## 🚀 Features

- Real-time detection of AI-generated faces
- High accuracy deep learning model
- Scalable FastAPI backend
- Containerized with Docker
- Easy deployment to Google Cloud Run
- RESTful API endpoints

## 🛠 Tech Stack

- **Backend**: FastAPI (0.95.2)
- **Server**: Uvicorn (0.22.0), Gunicorn (20.0.4)
- **Deep Learning**: TensorFlow (2.19.0)
- **Image Processing**: OpenCV, scikit-image, Pillow
- **Cloud**: Google Cloud Run, Google Cloud Storage
- **Containerization**: Docker
- **CI/CD**: GitHub Actions

## 📦 API Endpoints

### POST `/detect`
Analyze an image to determine if it contains AI-generated faces.

**Request:**
```json
{
  "image_url": "string"  // or file upload
}
```
-----------------------
**Response:**
```json
{
    "statusCode": integer,
    "message": "string",
    "data": {
        "label": "string",
        "confidence": float,
        "image_url": "string",
        "probabilities": {
            "FAKE": float,
            "REAL": float
        }
    }
}
```


## 🐳 Running with Docker

1. Build the Docker image:
```bash
docker build -t gansdoctor .
```

2. Run the container:
```bash
docker run -p 8080:8080 gansdoctor
```

The API will be available at `http://localhost:8080`

## ☁️ Deployment

GANSDOCTOR is configured for automatic deployment to Google Cloud Run via GitHub Actions. Pushes to the `main` branch trigger the deployment workflow.

### Manual Deployment to Cloud Run:
```bash
gcloud run deploy gansdoctor \
  --image asia-southeast2-docker.pkg.dev/YOUR_PROJECT/YOUR_REPO/YOUR_IMAGE \
  --platform managed \
  --region asia-southeast2 \
  --allow-unauthenticated \
  --memory 1Gi \
  --timeout 900s
```

## 📝 Requirements

See [requirements.txt](requirements.txt) for complete Python dependencies.


# Face Emotion Detection

A web application that detects emotions from facial expressions in real-time using a webcam and displays interactive visualizations.  
The model is trained on the FER-2013 dataset.

## Stack
- Backend: Python (Flask)
- Frontend: React + Vite


## Prerequisites
- Python >= 3.10
- Node.js >= 18
- npm

## Run the app

### With Docker
```bash
docker compose up --build
```

### Without Docker
Run the frontend
```bash
cd frontend
npm install
npm run dev 
```
Run the backend
```bash
cd backend
pip install -r requirements.txt
flask --app ./main.py run
```

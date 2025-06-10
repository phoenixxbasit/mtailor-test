---

# 🧠 Image Classification on Cerebrium (ONNX + Docker + FastAPI)

This project deploys an image classification model (ResNet-18 trained on ImageNet) to the **Cerebrium serverless GPU platform** using **ONNX**, **FastAPI**, and **Docker**. It takes an input image, runs preprocessing, and returns the predicted class ID.

---

## 🚀 How to Run the Project

### ✅ EASIEST: Use `test_server.py` (RECOMMENDED)

This script already includes all necessary credentials (API key and endpoint).

1. **Create virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies**

   ```bash
   pip install argparse requests
   ```

3. **Run the test script**

   ```bash
   python test_server.py --image_path images/n01440764_tench.JPEG
   ```

---

## 🐳 Alternative: Run Locally with Docker

1. **Build Docker image**

   ```bash
   docker build -t image-classifier .
   ```

2. **Run Docker container**

   ```bash
   docker run -p 8000:8000 image-classifier
   ```

3. **Send a prediction request**

   ```bash
   curl -X POST http://localhost:8000/predict -F "image=@images/n01440764_tench.JPEG"
   ```

---

## 📂 Summary

* Model: ResNet-18 (ONNX)
* Platform: Cerebrium (Docker deployment)
* Main API: FastAPI
* Run using: `test_server.py` (easy) or Docker (optional)
* Output: Predicted class ID

---

That’s it — simple and ready to go!

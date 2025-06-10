from fastapi import FastAPI, File, UploadFile
from model import ModelService
import uvicorn

app = FastAPI()
service = ModelService(model_path="models/model.onnx")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    result = service.predict_image(contents, return_probabilities=False, top_k=1)
    return {
        "class_id": result["predicted_class_id"],
        "class_name": result["predicted_class_name"],
        "confidence": result["confidence"],
        "times": result["processing_time"]
    }

@app.get("/health")
async def health():
    info = service.get_service_info()
    return {"status": "healthy", "model_info": info["model_info"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)

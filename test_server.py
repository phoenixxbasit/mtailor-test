import argparse, requests, os

CEREBRIUM_URL = "https://api.cortex.cerebrium.ai/v4/p-0e00b842/mtailor-imagenet-classifier"
BEARER_TOKEN = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJwcm9qZWN0SWQiOiJwLTBlMDBiODQyIiwiaWF0IjoxNzQ5NTQyMzM0LCJleHAiOjIwNjUxMTgzMzR9.hll2NCcb7VY7AJJIhIog-sI2uE_6izRJ0RL5CFu-Un1XO7EXQxlkZlJwHNMws6pwqOdxxQLnygZGgKCVvFElM3nwm4XabOZc3_H5xkK5HF6uyJdf6y4puEL3-8cWQGSsVptS4xAU6KQAQme5UzdHy7H_qAWqeiRN8zgPgo8_HTvrodYIdgSrB8bpfn_CcbXCwYJm_bOZkXjUZnowUwcWgy2MuMT6E40-qy74Tn59WsG0uEdE_srgM-yCh_VHVcey69StNr5YN4g0w8pKENADmpDsbP_h2oNyi7R7eU2ixitbHfd226UPRDRO7-2xUdUeAuOZqpap8z9tH4zPl_S3ow"

HEADERS = {
    "Authorization": f"Bearer {BEARER_TOKEN}"
}

def predict_image(image_path):
    with open(image_path, "rb") as img_file:
        files = {"file": img_file}
        response = requests.post(f"{CEREBRIUM_URL}/predict", headers=HEADERS, files=files)

    if response.status_code == 200:
        result = response.json()
        print(f"✅ Prediction: Class ID = {result.get('class_id')}")
    else:
        print(f"❌ Failed: {response.status_code} - {response.text}")

def run_health_check():
    response = requests.get(f"{CEREBRIUM_URL}/health", headers=HEADERS)
    if response.status_code == 200:
        print(f"✅ Health Check Passed: {response.json()}")
    else:
        print(f"❌ Health Check Failed: {response.status_code} - {response.text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to image for prediction")
    parser.add_argument("--test", action="store_true", help="Run custom platform tests")

    args = parser.parse_args()

    if args.test:
        print("Running platform tests...")
        run_health_check()
    elif args.image:
        if not os.path.exists(args.image):
            print(f"❌ Image not found: {args.image}")
        else:
            predict_image(args.image)
    else:
        print("❌ Please provide --image <path> or --test")
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Request,
    Header,
    HTTPException,
)
from pathlib import Path
from datetime import datetime
import shutil
import csv
import os

app = FastAPI()

SAVE_DIR = Path("./received")
SAVE_DIR.mkdir(exist_ok=True)

LOG_FILE = Path("uploads.csv")

API_KEY = os.getenv(
    "MODEL_UPLOAD_API_KEY", 
    "jg9438ghjhj9438jJs8ufh0NFYa7s8fnYA08F7Y"
    )

ALLOWED_IPS = {
    "127.0.0.1",
    "192.168.1.100",
}


def append_log(timestamp, ip, modelname, status):
    file_exists = LOG_FILE.exists()

    with LOG_FILE.open("a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["timestamp", "ip", "model", "status"])

        writer.writerow([timestamp, ip, modelname, status])


@app.post("/upload-model")
async def upload_model(
    request: Request,
    file: UploadFile = File(...),
    x_api_key: str = Header(...),
):
    try:
        ip = request.client.host
        timestamp = datetime.now().isoformat()

        if ip not in ALLOWED_IPS:
            append_log(timestamp, ip, file.filename, "DENIED_IP")
            raise HTTPException(status_code=403, detail="IP address not allowed")

        if x_api_key != API_KEY:
            append_log(timestamp, ip, file.filename, "DENIED_API_KEY")
            raise HTTPException(status_code=401, detail="Invalid API key")

        destination = SAVE_DIR / file.filename

        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        append_log(timestamp, ip, file.filename, "SUCCESS")

        return {
            "status": "success",
            "saved_to": str(destination),
        }
    except Exception as e:
        print(f"Couldnt upload model {e}")
        return {
            "status": "error",
            "message": e
        }

@app.get("/ping")
async def ping(request: Request):
    try:
        print(f"> PING FROM {request.client.host}")
        return {
            "status": "success",
            "message": "pong!",
        }
    except Exception as e:
        print(f"Couldn't ping {e}")
        return {
            "status": "error",
            "message": e
        }

"""

# Run command

uvicorn model-receiver:app --host 0.0.0.0 --port 8000 --reload

"""
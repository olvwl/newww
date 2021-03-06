import os
from google.cloud import storage
from fastapi import FastAPI, File
from segmentation import get_yolov5, get_image_from_bytes
from starlette.responses import Response
from starlette.responses import JSONResponse
import io
from PIL import Image
import json
import numpy as np
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel


model = get_yolov5()

app = FastAPI(
    title="Custom YOLOV5 Machine Learning API",
    description="""Obtain object value out of image
                    and return image and json result""",
    version="0.0.1",
)

origins = [
    "http://localhost",
    "http://localhost:8000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

path = "/home/syafridamelania/github/newww"
    

@app.post("/object-to-json")
async def detect_return_json_result(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
    detect_res = json.loads(detect_res)
    print(detect_res)
    return {"result": detect_res}

@app.post("/object-to-img")
async def detect_return_base64_img(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    results.render()  # updates results.imgs with boxes and labels
    ##print("kucing" ,results)
    client = storage.Client()
    bucket = client.get_bucket('olvwl-server.appspot.com')
    for img in results.imgs:
        bytes_io = io.BytesIO()
        img_base64 = Image.fromarray(img)
        img_base64.save(bytes_io, format="jpeg")
        ##print("kucing", bytes_io)
        imageResult = Response(bytes_io.getvalue(), media_type= "image/jpeg")
    
        image_data = bytes_io.getvalue()

        with open('image.jpeg','wb') as image:
            image.write(image_data)
            image.close()
            
        blob = bucket.blob('image.jpeg')
        blob.upload_from_filename('image.jpeg')
        blob.make_public()
        url = blob.public_url
        CACHE_CONTROL="public, max-age=0"
        blob.cache_control = CACHE_CONTROL
        blob.patch()

        jsonConvert = '{"imageUrl" : "https://storage.googleapis.com/olvwl-server.appspot.com/image.jpeg"}'
        x = json.loads(jsonConvert)

    return x

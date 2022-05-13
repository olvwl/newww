import os
from fastapi import FastAPI, File
from segmentation import get_yolov5, get_image_from_bytes
from starlette.responses import Response
import io
from PIL import Image
import json
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI()

model = get_yolov5()

path = "home/syafridamelania/github/newww"

@app.post("/object-to-json")
async def detect_food_return_json_result(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
    detect_res = json.loads(detect_res)
    return {"result": detect_res}


@app.post("/object-to-img")
async def detect_return_base64_img(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    results.render()  # updates results.imgs with boxes and labels
    ##print("kucing" ,results)
    for img in results.imgs:
        bytes_io = io.BytesIO()
        img_base64 = Image.fromarray(img)
        img_base64.save(bytes_io, format="jpeg")
        imageResult = Response(bytes_io.getvalue(), media_type= "image/jpeg")
    
        image_data = bytes_io.getvalue()

        with open('image.jpeg','wb') as image:
            image.write(image_data)
            image.close()

        filePath = os.path.join(path, "image.jpeg")

    return FileResponse(FilePath)

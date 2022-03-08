import copy
import os
import uvicorn

import uvicorn
import cv2

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from imutils.perspective import four_point_transform

app = FastAPI()


def save_file(file:UploadFile, path:str):
    try:
        os.mkdir("images")
        print("[INFO] Created Directory : {}".format(os.getcwd()))
    except Exception as e:
        print("[INFO] 'images' Directory already exists") 
    file_name = os.getcwd()+path+file.filename.replace(" ", "-")
    with open(file_name,'wb+') as f:
        f.write(file.file.read())
        f.close()
    print("[INFO] saved {}".format(file_name))
    return file_name



def detect_bounds(image,savename=""):
    orig_img = image.copy()
    # preprocess the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 75, 200)
    # find and sort the contours
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # go through each contour
    for contour in contours:
        # approximate each contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.05 * peri, True)
        # check if we have found our document
        if len(approx) == 4:
            doc_cnts = approx
            break
    # apply warp perspective to get the top-down view
    warped = four_point_transform(orig_img, doc_cnts.reshape(4, 2))
    # warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    try:
        os.mkdir("corrected")
        print("[INFO] Created Directory : {}".format(os.getcwd()))
    except Exception as e:
        print("[INFO] 'corrected' Directory already exists")   
    # write the image in the ouput directory
    file_name = os.getcwd() + "/corrected/" + os.path.basename(savename)
    cv2.imwrite(file_name, warped)
    print("[INFO] saved {}".format(file_name))
    return True



@app.post("/uploadfile/")
async def create_upload_files(imagefile: UploadFile = File(...)):
    file_name = save_file(imagefile, "/images/")
    load_image = cv2.imread(file_name)
    if_detected = detect_bounds(load_image, imagefile.filename)
    print("[INFO] Control back to caller")
    return {"file_name":imagefile.filename, "document_detected": if_detected}




@app.get("/")
async def main():
    return {"connection_status": "OK"}


if __name__ == "__main__":
    uvicorn.run("fileupload:app", host="127.0.0.1", port=8000, reload=True)
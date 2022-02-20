import model

import os
from urllib import response
from flask import Flask, flash, request, redirect, url_for, Response, render_template, send_from_directory
import flask
from werkzeug.utils import secure_filename
ROOT_DIR = os.path.abspath(os.curdir)
print(ROOT_DIR)
# from tensorflow.keras import models

# seg_model = models.load_model("/content/drive/MyDrive/unet_resnet34_body_parse.h5")


UPLOAD_FOLDER = ROOT_DIR + "/static/upload"
DWNLD_FOLDER = ROOT_DIR + "/static/gen"
ALLOWED_EXTENSIONS = set(["txt", "pdf", "png", "jpg", "jpeg", "gif"])

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["DWNLD_FOLDER"] = DWNLD_FOLDER

@app.route("/upload", methods = ["GET", "POST"])
def upload_file():
    if request.method == "POST":
        # check if the post request has the file part
        if "file" not in request.files:
            flash("No file part")
            return "no file1"
        file = request.files["file"]
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == "":
            flash("No selected file")
            return "no file2"
        if file:
            filename = secure_filename(file.filename)
            if request.args.get("cat") == "person":
                file.save(os.path.join(app.config["UPLOAD_FOLDER"], "1.jpg"))
            elif request.args.get("cat") == "cloth":
                file.save(os.path.join(app.config["UPLOAD_FOLDER"], "2.jpg"))
    return Response({"path": "1.jpg", "status": "success"}, status=200) 

@app.route("/upload_p", methods = ["GET", "POST"])
def upload_p():
    if request.method == "POST":
        # check if the post request has the file part
        if "file" not in request.files:
            flash("No file part")
            return "no file1"
        file = request.files["file"]
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == "":
            flash("No selected file")
            return "no file2"
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], "1.jpg"))
    return Response({"path": "1.jpg", "status": "success"}, status=200) 

@app.route("/upload_c", methods = ["GET", "POST"])
def upload_c():
    if request.method == "POST":
        # check if the post request has the file part
        if "file" not in request.files:
            flash("No file part")
            return "no file1"
        file = request.files["file"]
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == "":
            flash("No selected file")
            return "no file2"
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], "2.jpg"))
    return Response({"path": "2.jpg", "status": "success"}, status=200) 

@app.route("/generate", methods = ["GET", "POST"])
def get_result():
    if request.method == "GET":

        pth = model.load_output(ROOT_DIR)

    try:
        return flask.send_file(ROOT_DIR + "/static/gen/1.jpg", as_attachment=True)
    except FileNotFoundError:
        flask.abort(404)



@app.route("/")
def main():
    return render_template("index.html")

if __name__ == "__main__":
    app.secret_key = "super secret key"
    app.config["SESSION_TYPE"] = "filesystem"
    app.run()
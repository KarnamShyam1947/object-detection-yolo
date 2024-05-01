from flask import *
from predict import predict_image_using_bytes

main = Blueprint("main", __name__)

@main.route("/")
def index():
    return render_template("index.html")

@main.route("/predict", methods = ['POST'])
def predict():
    if request.method == 'POST':
        image = request.files.get('image')

        result = predict_image_using_bytes(image.read())

        return result

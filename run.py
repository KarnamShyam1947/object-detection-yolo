from flask import *
from api import api
from predict import predict_image_using_bytes
from app import main

app = Flask(__name__)

app.register_blueprint(main)
api.init_app(app)

if __name__ == '__main__' : 
    app.run(debug=True)

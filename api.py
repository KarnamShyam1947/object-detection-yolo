from flask_restx import Api, Namespace, Resource, reqparse
from werkzeug.datastructures import FileStorage
from predict import predict_api

api = Api(
    version="1.0",
    title="Workers safty equpiment detection",
    description="use this api for Workers safty equpiment detection.",
    validate=True,
    doc="/api"
)

fileReqParser = reqparse.RequestParser()
fileReqParser.add_argument(name="image", type=FileStorage, location="files")

predictController = Namespace(
    "predict controller",
    "upload image and get prediction",
    "/predict"
)

@predictController.route("/workers-safty")
class PredictResource(Resource):
    @predictController.expect(fileReqParser)
    def post(self):
        args = fileReqParser.parse_args()
        file = args['image']
        result = predict_api(file.read())
        return result

api.add_namespace(predictController)

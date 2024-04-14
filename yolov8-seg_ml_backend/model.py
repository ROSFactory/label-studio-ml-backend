import os
import math
import json
from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from ultralytics import YOLO


class NewModel(LabelStudioMLBase):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(NewModel, cls).__new__(cls)
        return cls._instance

    def __init__(self, *args, **kwargs):
        #     print(**kwargs)
        # super(NewModel, self).__init__(**kwargs)
        if not hasattr(
            self, "initialized"
        ):  # This ensures __init__ code is executed only once
            super(NewModel, self).__init__(**kwargs)
            model_path = os.getenv("MODEL_PATH", "/app/models/yolov8s-seg.pt")
            self.model = YOLO(model_path)
            self.initialized = True
            print("Init DONE")

    def predict(
        self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs
    ) -> List[Dict]:
        """Write your inference logic here
        :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
        :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml.html#Passing-data-to-ML-backend)
        :return predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Raw-JSON-format-of-completed-tasks)
        """
        # basename = "/users/nicolas/Library/Application Support/label-studio" # from mac
        # basename = "/home/nicolas/.local/share/label-studio" # for label-studio ml backend in dev mode
        basename = "/app/data"  # for label-studio ml backend in prod mode
        # self.model = YOLO("/Users/nicolas/git/computer-vision-airbus/curiosity/models/yolov8n-seg.pt")
        predictions = []

        for task in tasks:
            image_relative_path = task.get("data").get("image")
            image_absolute_path = os.path.join(
                basename, image_relative_path.replace("/data/", "media/")
            )
            print(f"Absolute path: ", image_absolute_path)
            results = self.model(
                image_absolute_path
            )  # 1 result per image. 1 image => len(Results) = 1
            task_predictions = []

            for result in results:
                # image_height, image_width = result.orig_shape
                detections = json.loads(result.tojson(normalize=True))
                score = 1

                for detection in detections:
                    # print(f"Detection: {detection}\n\n")
                    points = []

                    if "segments" in detection:
                        confidence = detection.get("confidence")
                        if confidence < score:
                            score = confidence  # useful to return a score for images having multiple detection
                        for x, y in zip(
                            detection.get("segments").get("x"),
                            detection.get("segments").get("y"),
                        ):
                            points.append(
                                [x * 100, y * 100]
                            )  # x, y, to be provided in percentages of overall image dimension

                        prediction = {
                            "from_name": "label",
                            "to_name": "image",
                            "type": "polygonlabels",
                            "value": {
                                "points": points,
                                "polygonlabels": [detection.get("name")],
                            },
                            "score": confidence,
                        }
                        task_predictions.append(prediction)

        # print(f'''\
        # Run prediction on {tasks}
        # Received context: {context}
        # Project ID: {self.project_id}
        # Label config: {self.label_config}
        # Parsed JSON Label config: {self.parsed_label_config}''')

        predictions.append(
            {
                "result": task_predictions,
                "score": score,  # Customize this score as needed
            }
        )

        return predictions

    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get("my_data")
        old_model_version = self.get("model_version")
        print(f"Old data: {old_data}")
        print(f"Old model version: {old_model_version}")

        # store new data to the cache
        self.set("my_data", "my_new_data_value")
        self.set("model_version", "my_new_model_version")
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print("fit() completed successfully.")
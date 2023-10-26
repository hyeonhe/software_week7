import os
from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def analyze_image(filename):
    # Load YOLOv4 model and weights
    net = cv2.dnn.readNet('yolov4/yolov4.weights', 'yolov4/yolov4.cfg')

    # Load class names
    with open('yolov4/coco.names', 'r') as f:
        classes = f.read().strip().split('\n')

    image = cv2.imread(filename)

    if image is not None:
        # 이미지가 성공적으로 읽어졌을 때 분석 로직을 수행
        height, width, _ = image.shape

        # YOLOv4 객체 감지
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(net.getUnconnectedOutLayersNames())

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        # Non-maximum suppression to remove duplicate detections
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Get detected objects and their coordinates
        detected_objects = []
        for i in indices:
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            box = boxes[i]
            detected_objects.append({
                "label": label,
                "confidence": confidence,
                "box": box
            })

        return {"width": width, "height": height, "detected_objects": detected_objects}
    else:
        return {"error": "이미지를 읽을 수 없습니다."}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 이미지 업로드 처리
        if 'file' not in request.files:
            return jsonify({"error": "이미지 파일이 업로드되지 않았습니다."})

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "이미지 파일을 선택하지 않았습니다."})

        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            # 이미지 분석
            result = analyze_image(filename)
            return jsonify(result)

    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "이미지 파일이 업로드되지 않았습니다."})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "이미지 파일을 선택하지 않았습니다."})

    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # 이미지 분석
        result = analyze_image(filename)
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

from ultralytics import YOLO
import torch, io, cv2, os
from PIL import Image
import numpy as np

class_names = ['Helmet', 'Goggles', 'Jacket', 'Gloves', 'Footwear']
#                0           1           2       3          4

def test_bytes():
    with open("test_image/5.jpeg", "rb") as f:
        image_bytes = io.BytesIO(f.read())
        pil_obj = Image.open(image_bytes)

        # print(pil_obj)
        pil_obj.show()

        return pil_obj

def predict(image_bytes):
    os.remove("static/result.jpg")
    
    model = YOLO('best_50.pt')

    image_np = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    image_bytes = io.BytesIO(image_bytes)
    pil_obj = Image.open(image_bytes)


    result = model.predict(source=pil_obj, conf=0.4)
    # print(result)

    clsIdx = torch.tensor(result[0].boxes.cls, dtype=torch.int32).tolist()
    bboxs = torch.tensor(result[0].boxes.xyxy, dtype=torch.int32).tolist()
    confs = torch.tensor(result[0].boxes.conf, dtype=torch.float16).tolist()

    result = dict()
    cnt = dict()
    conf_dic = dict()
    r = dict()

    for idx, box, conf in zip(clsIdx, bboxs, confs):
        classname = class_names[idx]

        print('class name : ', classname, conf)
        x_min, y_min, x_max, y_max = box

        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        text = f"{classname}: {conf:.2f}"
        cv2.putText(image, text, (int(x_min), int(y_min - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cnt[classname] = cnt.get(classname, 0) + 1
        conf_dic[classname] = conf_dic.get(classname, [])
        conf_dic[classname].append(conf)
        
        result[classname] = max(result.get(classname, 0), conf)

    for item, value in result.items():
        r[item] = {
            'count' : cnt.get(item),
            'conf' : conf_dic.get(item),
        }

    cv2.imwrite("static/result.jpg", image)

    return r

def predict_image_using_bytes(image_bytes):
    return predict(image_bytes)

def predict_api(image_bytes):
    r = predict(image_bytes)
    r['image_url'] = "http://localhost:5000/static/result.jpg"   

    return r

    

def main():

    with open("1.jpeg", 'rb') as f:
        print(predict_image_using_bytes(f.read()))
    
if __name__ == '__main__' : 
    main()

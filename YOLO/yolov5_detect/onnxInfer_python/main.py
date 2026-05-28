import cv2
import numpy as np
import onnxruntime as ort
import time

ONNX_MODEL_PATH = "E:\\code_git\\Algorithm\\YOLO\\yolov5_detect\\onnxInfer_python\\yolov5s.onnx"
IMAGE_PATH = "E:\\code_git\\Algorithm\\YOLO\\yolov5_detect\\onnxInfer_python\\bus.jpg"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
INPUT_SIZE = 640

session = ort.InferenceSession(ONNX_MODEL_PATH)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

input_type = session.get_inputs()[0].type
print("模型输入精度类型：", input_type)

if "float16" in input_type:
    print("✅ 模型精度：FP16")
elif "float32" in input_type:
    print("✅ 模型精度：FP32")
elif "uint8" in input_type or "int8" in input_type:
    print("✅ 模型精度：INT8 (量化模型)")


def preprocess(image_path, input_size):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    scale = input_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h))
    canvas = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    canvas[(input_size - new_h) // 2:(input_size - new_h) // 2 + new_h,
    (input_size - new_w) // 2:(input_size - new_w) // 2 + new_w, :] = img_resized
    canvas = canvas / 255.0
    canvas = canvas.transpose(2, 0, 1)  # HWC -> CHW
    canvas = canvas[np.newaxis, :, :, :].astype(np.float32)
    return canvas, h, w

total_start = time.time()
start = time.time()
input_tensor, ori_h, ori_w = preprocess(IMAGE_PATH, INPUT_SIZE)
pre_time = (time.time() - start) * 1000

start = time.time()
outputs = session.run([output_name], {input_name: input_tensor})
infer_time = (time.time() - start) * 1000


def postprocess(outputs, ori_h, ori_w, input_size, conf_thres, iou_thres):
    predictions = np.squeeze(outputs[0])

    conf_mask = predictions[:, 4] > conf_thres
    predictions = predictions[conf_mask]
    if len(predictions) == 0:
        return [], [], []

    class_scores = np.max(predictions[:, 5:], axis=1)
    class_ids = np.argmax(predictions[:, 5:], axis=1)
    scores = predictions[:, 4] * class_scores

    boxes = predictions[:, :4].copy()
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    scale = input_size / max(ori_h, ori_w)
    pad_w = (input_size - ori_w * scale) / 2
    pad_h = (input_size - ori_h * scale) / 2

    boxes[:, 0] = (boxes[:, 0] - pad_w) / scale
    boxes[:, 1] = (boxes[:, 1] - pad_h) / scale
    boxes[:, 2] = (boxes[:, 2] - pad_w) / scale
    boxes[:, 3] = (boxes[:, 3] - pad_h) / scale

    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thres, iou_thres)
    if len(indices) == 0:
        return [], [], []

    indices = indices.flatten()
    final_boxes = boxes[indices].astype(int)
    final_scores = scores[indices]
    final_classes = class_ids[indices]
    return final_boxes, final_scores, final_classes

start = time.time()
boxes, scores, class_ids = postprocess(outputs, ori_h, ori_w, INPUT_SIZE, CONF_THRESHOLD, IOU_THRESHOLD)
post_time = (time.time() - start) * 1000

total_time = (time.time() - total_start) * 1000

CLASS_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']

img = cv2.imread(IMAGE_PATH)
for box, score, cls_id in zip(boxes, scores, class_ids):
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{CLASS_NAMES[cls_id]} {score:.2f}"
    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imwrite("result.jpg", img)
cv2.imshow("Detection Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(f"检测完成！共检测到 {len(boxes)} 个目标")

# ======================== 输出时间 ========================
print("="*50)
print(f"预处理时间：\t{pre_time:.2f} ms")
print(f"推理时间：\t{infer_time:.2f} ms")
print(f"后处理时间：\t{post_time:.2f} ms")
print(f"总耗时：\t{total_time:.2f} ms")
print(f"检测目标数：\t{len(boxes)} 个")
print("="*50)
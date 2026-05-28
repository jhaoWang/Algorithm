import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time

# ======================== 配置参数 ========================
ENGINE_PATH = "E:\\code_git\\Algorithm\\YOLO\\yolov5_detect\\trtInfer_python\\yolov5s.engine"
IMAGE_PATH = "E:\\code_git\\Algorithm\\YOLO\\yolov5_detect\\trtInfer_python\\bus.jpg"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
INPUT_SIZE = 640

# ======================== TensorRT 10.12 初始化 ========================
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

cuda_stream = cuda.Stream()

def load_engine(engine_path):
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    runtime = trt.Runtime(TRT_LOGGER)
    return runtime.deserialize_cuda_engine(engine_data)

engine = load_engine(ENGINE_PATH)

# 放在 engine = load_engine(...) 之后
print("="*50)
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    dtype = engine.get_tensor_dtype(name)
    mode = engine.get_tensor_mode(name)
    print(f"tensor: {name}, dtype: {dtype}, mode: {mode}")
print("="*50)
    

context = engine.create_execution_context()

# 获取输入输出张量名（TRT10 最新方式）
input_name = engine.get_tensor_name(0)   # 第0个是输入
output_name = engine.get_tensor_name(1)  # 第1个是输出

# 固定shape
input_shape = (1, 3, 640, 640)
output_shape = (1, 25200, 85)

# 分配内存
host_input = cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float32)
host_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
d_input = cuda.mem_alloc(host_input.nbytes)
d_output = cuda.mem_alloc(host_output.nbytes)

# ======================== 预处理（不变） ========================
def preprocess(image_path, input_size):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    scale = input_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h))
    canvas = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    canvas[(input_size - new_h)//2 : (input_size - new_h)//2 + new_h,
           (input_size - new_w)//2 : (input_size - new_w)//2 + new_w, :] = img_resized
    canvas = canvas / 255.0
    canvas = canvas.transpose(2, 0, 1)
    canvas = canvas[np.newaxis, ...].astype(np.float32)
    return canvas, h, w

# ======================== 后处理（不变） ========================
def postprocess(outputs, ori_h, ori_w, input_size, conf_thres, iou_thres):
    predictions = np.squeeze(outputs)
    conf_mask = predictions[:, 4] > conf_thres
    predictions = predictions[conf_mask]
    if len(predictions) == 0:
        return [], [], []

    class_scores = np.max(predictions[:,5:], axis=1)
    class_ids = np.argmax(predictions[:,5:], axis=1)
    scores = predictions[:,4] * class_scores

    cx, cy, w, h = predictions[:,0], predictions[:,1], predictions[:,2], predictions[:,3]
    boxes = np.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], axis=1)

    scale = input_size / max(ori_h, ori_w)
    pad_w = (input_size - ori_w * scale) / 2
    pad_h = (input_size - ori_h * scale) / 2
    boxes = (boxes - [pad_w, pad_h, pad_w, pad_h]) / scale

    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thres, iou_thres)
    if len(indices) == 0:
        return [], [], []

    indices = indices.flatten()
    return boxes[indices].astype(int), scores[indices], class_ids[indices]

# ======================== 推理流程 ========================
total_start = time.time()

# 预处理
start = time.time()
input_tensor, ori_h, ori_w = preprocess(IMAGE_PATH, INPUT_SIZE)
pre_time = (time.time() - start) * 1000

# TRT 10.12 推理（核心修复）
start = time.time()
np.copyto(host_input, input_tensor.ravel())
cuda.memcpy_htod(d_input, host_input)

# 正确 TRT10 API
context.set_tensor_address(input_name, int(d_input))
context.set_tensor_address(output_name, int(d_output))
context.execute_async_v3(cuda_stream.handle)

cuda.memcpy_dtoh(host_output, d_output)
outputs = host_output.reshape(output_shape)
infer_time = (time.time() - start) * 1000

# 后处理
start = time.time()
boxes, scores, class_ids = postprocess(outputs, ori_h, ori_w, INPUT_SIZE, CONF_THRESHOLD, IOU_THRESHOLD)
post_time = (time.time() - start) * 1000

total_time = (time.time() - total_start) * 1000

# ======================== 绘图 ========================
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
    cv2.putText(img, f"{CLASS_NAMES[cls_id]} {score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

cv2.imwrite("result_trt.jpg", img)
cv2.imshow("TRT10 Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ======================== 输出时间 ========================
print("="*50)
print(f"预处理时间：\t{pre_time:.2f} ms")
print(f"推理时间：\t{infer_time:.2f} ms")
print(f"后处理时间：\t{post_time:.2f} ms")
print(f"总耗时：\t\t{total_time:.2f} ms")
print(f"检测目标数：\t{len(boxes)} 个")
print("="*50)

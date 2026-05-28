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

# ======================== TensorRT 10.x 初始化 ========================
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
cuda_stream = cuda.Stream()

def load_engine(engine_path):
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    runtime = trt.Runtime(TRT_LOGGER)
    return runtime.deserialize_cuda_engine(engine_data)

engine = load_engine(ENGINE_PATH)
context = engine.create_execution_context()

input_name = engine.get_tensor_name(0)
output_name = engine.get_tensor_name(1)

input_shape = (1, 3, 640, 640)
output_shape = (1, 25200, 85)

# 分配内存
host_input = cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float32)
host_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
d_input = cuda.mem_alloc(host_input.nbytes)
d_output = cuda.mem_alloc(host_output.nbytes)

# ======================== 预处理 ========================
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

# ======================== 后处理 ========================
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

    try:
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thres, iou_thres).flatten()
    except:
        indices = []
    return boxes[indices].astype(int) if len(indices) else [], scores[indices], class_ids[indices]

# ======================== 预热（必须，否则第一次不准） ========================
print("Warming up TensorRT engine...")
dummy_input, _, _ = preprocess(IMAGE_PATH, INPUT_SIZE)
for _ in range(10):
    np.copyto(host_input, dummy_input.ravel())
    cuda.memcpy_htod_async(d_input, host_input, cuda_stream)
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))
    context.execute_async_v3(cuda_stream.handle)
    cuda.memcpy_dtoh_async(host_output, d_output, cuda_stream)
    cuda_stream.synchronize()
print("Warmup done!\n")

# ======================== 【完整链路测速：预处理 + 拷贝 + 推理 + 拷贝 + 后处理】 ========================
N = 50  # 测试50次取平均，更准确

total_times = []
pre_times = []
copy_in_times = []
infer_times = []
copy_out_times = []
post_times = []

print(f"Testing {N} times end-to-end...")

for _ in range(N):
    # 1. 预处理
    t0 = time.time()
    input_tensor, ori_h, ori_w = preprocess(IMAGE_PATH, INPUT_SIZE)
    t1 = time.time()

    # 2. CPU -> GPU 拷贝
    np.copyto(host_input, input_tensor.ravel())
    cuda.memcpy_htod_async(d_input, host_input, cuda_stream)
    t2 = time.time()

    # 3. 模型推理
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))
    cuda_stream.synchronize()
    t3 = time.time()
    context.execute_async_v3(cuda_stream.handle)
    cuda_stream.synchronize()
    t4 = time.time()

    # 4. GPU -> CPU 拷贝
    cuda.memcpy_dtoh_async(host_output, d_output, cuda_stream)
    cuda_stream.synchronize()
    t5 = time.time()

    # 5. 后处理 + NMS
    outputs = host_output.reshape(output_shape)
    boxes, scores, class_ids = postprocess(outputs, ori_h, ori_w, INPUT_SIZE, CONF_THRESHOLD, IOU_THRESHOLD)
    t6 = time.time()

    # 统计每一段耗时
    total_times.append((t6 - t0) * 1000)
    pre_times.append((t1 - t0) * 1000)
    copy_in_times.append((t2 - t1) * 1000)
    infer_times.append((t4 - t3) * 1000)
    copy_out_times.append((t5 - t4) * 1000)
    post_times.append((t6 - t5) * 1000)

# ======================== 输出完整耗时统计 ========================
print("=" * 70)
print(f"【YOLOv5 TensorRT 端到端完整耗时统计 (平均 {N} 次)】")
print("=" * 70)
print(f"预处理时间         : {np.mean(pre_times):>6.2f} ms")
print(f"CPU → GPU 数据拷贝  : {np.mean(copy_in_times):>6.2f} ms")
print(f"模型纯推理时间      : {np.mean(infer_times):>6.2f} ms")
print(f"GPU → CPU 数据拷贝  : {np.mean(copy_out_times):>6.2f} ms")
print(f"后处理(NMS)时间     : {np.mean(post_times):>6.2f} ms")
print(f"--------------------------------------------")
print(f"✅ 端到端总耗时     : {np.mean(total_times):>6.2f} ms")
print(f"检测目标数量        : {len(boxes)} 个")
print("=" * 70)

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
cv2.imshow("TRT Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
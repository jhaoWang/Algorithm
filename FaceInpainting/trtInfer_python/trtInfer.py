import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_ENGINE_PATH = r"E:\code_git\Algorithm\FaceInpainting\trtInfer_python\inpaint.engine"

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

engine = load_engine(TRT_ENGINE_PATH)
context = engine.create_execution_context()

# ====================== 1. 读取图片 ======================
img = cv2.imread(r"E:\pythonCode\lafin-master\lafin-master\examples\images\195579.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (256, 256))

landmark = cv2.imread(r"E:\pythonCode\lafin-master\lafin-master\checkpoints\results\landmark_inpaint\landmark\195579.png", cv2.IMREAD_GRAYSCALE)
landmark = cv2.resize(landmark, (256, 256))

mask = cv2.imread(r"E:\pythonCode\lafin-master\lafin-master\examples\masks\11257.png", cv2.IMREAD_GRAYSCALE)
mask = cv2.resize(mask, (256, 256))
mask = (mask > 0).astype(np.uint8) * 255

# ====================== 2. 预处理 ======================
# 人脸图
input_img = img.astype(np.float32) / 127.5 - 1.0
input_img = input_img.transpose(2, 0, 1).copy()  # <--- 修复
input_img = np.expand_dims(input_img, axis=0).copy()  # <--- 修复

# 关键点图
input_landmark = landmark.astype(np.float32) / 255.0
input_landmark = np.expand_dims(input_landmark, axis=0).copy()
input_landmark = np.expand_dims(input_landmark, axis=0).copy()

# mask图
input_mask = mask.astype(np.float32) / 255.0
input_mask = np.expand_dims(input_mask, axis=0).copy()
input_mask = np.expand_dims(input_mask, axis=0).copy()

# ====================== 3. TensorRT 推理 ======================
d_input_img = cuda.mem_alloc(input_img.nbytes)
d_input_landmark = cuda.mem_alloc(input_landmark.nbytes)
d_input_mask = cuda.mem_alloc(input_mask.nbytes)

output = np.empty((1, 3, 256, 256), dtype=np.float32)
d_output = cuda.mem_alloc(output.nbytes)

cuda.memcpy_htod(d_input_img, input_img)
cuda.memcpy_htod(d_input_landmark, input_landmark)
cuda.memcpy_htod(d_input_mask, input_mask)

# 推理
bindings = [int(d_input_img), int(d_input_landmark), int(d_input_mask), int(d_output)]
context.execute_v2(bindings)

cuda.memcpy_dtoh(output, d_output)

# ====================== 4. 后处理 ======================
output = output[0].transpose(1, 2, 0).copy()
output = np.clip(output * 255, 0, 255).astype(np.uint8)
output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

cv2.imshow("result", output_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
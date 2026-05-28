import cv2
import numpy as np
import onnxruntime as ort
import torch


# ====================== 加载 ONNX 模型 ======================
ort_sess = ort.InferenceSession("E:\\pythonCode\\lafin-master\\lafin-master\\inpaint.onnx")

input_type = ort_sess.get_inputs()[0].type
print("模型输入精度类型：", input_type)

if "float16" in input_type:
    print("✅ 模型精度：FP16")
elif "float32" in input_type:
    print("✅ 模型精度：FP32")
elif "uint8" in input_type or "int8" in input_type:
    print("✅ 模型精度：INT8 (量化模型)")


# ====================== 1. 读取 3 张真实图片 ======================
# 人脸图
img = cv2.imread("E:\\pythonCode\\lafin-master\\lafin-master\\examples\\images\\195579.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (256, 256))

# 关键点图 (landmark)
landmark = cv2.imread("E:\\pythonCode\\lafin-master\\lafin-master\\checkpoints\\results\\landmark_inpaint\\landmark\\195579.png", cv2.IMREAD_GRAYSCALE)
landmark = cv2.resize(landmark, (256, 256))

# mask图
mask = cv2.imread("E:\\pythonCode\\lafin-master\\lafin-master\\examples\\masks\\11257.png", cv2.IMREAD_GRAYSCALE)
mask = cv2.resize(mask, (256, 256))
mask = (mask > 0).astype(np.uint8) * 255

# ====================== 2. 图片预处理 ======================
# 人脸图：HWC → CHW → BATCH
# input_img = img.astype(np.float32) / 255.0
input_img = img.astype(np.float32) / 127.5 - 1.0
input_img = input_img.transpose(2, 0, 1)
input_img = np.expand_dims(input_img, axis=0)  # (1,3,256,256)

# 关键点图：HW → CHW → BATCH
input_landmark = landmark.astype(np.float32) / 255.0
input_landmark = np.expand_dims(input_landmark, axis=0)
input_landmark = np.expand_dims(input_landmark, axis=0)  # (1,1,256,256)

# mask图：HW → CHW → BATCH
input_mask = mask.astype(np.float32) / 255.0
input_mask = np.expand_dims(input_mask, axis=0)
input_mask = np.expand_dims(input_mask, axis=0)  # (1,1,256,256)

outputs = ort_sess.run(
    None,
    {
        "onnx::Mul_0": input_img,
        "onnx::Concat_1": input_landmark,
        "masks": input_mask
    }
)
output = outputs[0][0]
output = output.transpose(1, 2, 0)
output = np.clip(output * 255, 0, 255).astype(np.uint8)
output_bgr = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)


# 显示
cv2.imshow("res", output_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2
import numpy as np
import onnxruntime as ort

# ============ 自行修改配置 ============
ONNX_PATH    = "landmark.onnx"
IMG_PATH     = "E:\\pythonCode\\lafin-master\\lafin-master\\examples\\images\\195579.jpg"
MASK_PATH    = "E:\\pythonCode\\lafin-master\\lafin-master\\examples\\masks\\08012.png"
SAVE_RESULT  = "result_landmark.png"
INPUT_SIZE   = 256
LANDMARK_NUM = 68
# ======================================
ort_sess = ort.InferenceSession(ONNX_PATH)

src_img = cv2.imread(IMG_PATH)
h, w = src_img.shape[:2]

img = cv2.imread(IMG_PATH)
img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
img = img.astype(np.float32) / 255.0
img = np.transpose(img, (2, 0, 1))[np.newaxis, :]  # [1,3,256,256]

mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
mask = cv2.resize(mask, (INPUT_SIZE, INPUT_SIZE))
mask = mask.astype(np.float32) / 255.0
mask = mask[np.newaxis, np.newaxis, :]  # [1,1,256,256]

input_feed = img * (1.0 - mask) + mask

outputs = ort_sess.run(
    None,
    {
        "onnx::Mul_0": input_feed,
        "onnx::Sub_1": mask
    }
)
landmark_pred = outputs[0]

landmark_pred = landmark_pred.reshape(-1, LANDMARK_NUM, 2).astype(np.float32)
landmark_pred[landmark_pred >= INPUT_SIZE - 1] = INPUT_SIZE - 1
landmark_pred = landmark_pred[0]

scale_x = w / INPUT_SIZE
scale_y = h / INPUT_SIZE

black_background = np.zeros((h, w), dtype=np.uint8)

for (x, y) in landmark_pred:
    ori_x = int(x * scale_x)
    ori_y = int(y * scale_y)
    cv2.circle(black_background, (ori_x, ori_y), 2, 255, -1)  # 白色 = 255

cv2.imshow("Landmark Result", black_background)
cv2.imwrite(SAVE_RESULT, black_background)
cv2.waitKey(0)
cv2.destroyAllWindows()
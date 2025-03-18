import cv2
import numpy as np
from .model_service import EyeDiagnosisModel

def analyze_image(image_data):
    """处理图像二进制数据"""
    nparr = np.frombuffer(image_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR_BGR)

def process_images(left_data, right_data):
    model_service = EyeDiagnosisModel()
    """主处理流程"""
    try:
        # 解码图像
        left_img = analyze_image(left_data)
        right_img = analyze_image(right_data)

        # 验证图像有效性
        if left_img is None or right_img is None:
            raise ValueError("Invalid image data")

        if model_service.model is None:
            model_service.initialize_model()

        # 获取预测结果
        predictions = model_service.predict_probability(left_img, right_img)

        # 计算综合患病概率（取最大概率）
        print(predictions)
        return {
            "success": True,
            "detail": predictions
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
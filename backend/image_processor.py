import os

import cv2
import numpy as np
import torch

from A07_backend import settings
from .model_service import EyeDiagnosisModel, VesselSegmentor, OpticDiscSegmentor
from .oss_utils import upload_to_oss

model_service = EyeDiagnosisModel()


def analyze_image(image_data):
    """处理图像二进制数据"""
    nparr = np.frombuffer(image_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR_BGR)

def process_images(left_data, right_data, left_name, right_name):
    vessel_model = VesselProcessor()
    disk_model = OpticDiscProcessor()
    try:
        left_img = analyze_image(left_data)
        right_img = analyze_image(right_data)

        left_img = remove_black_borders(left_img)
        right_img = remove_black_borders(right_img)

        # 调整原图和掩膜的尺寸
        left_img = cv2.resize(left_img, (512, 512))
        right_img = cv2.resize(right_img, (512, 512))

        model_service.initialize_model()

        result = model_service.generate_heatmap(left_img, right_img)

        expanded_img = double_width(result['heatmap'])
        left_heatmap, right_heatmap = split_image(expanded_img)

        left_heatmap_name = left_name.replace('left', 'left_heatmap')
        right_heatmap_name = right_name.replace('right', 'right_heatmap')

        left_heatmap_url = upload_to_oss(left_heatmap_name, left_heatmap)
        right_heatmap_url = upload_to_oss(right_heatmap_name, right_heatmap)

        # 血管预测
        left_vessel = vessel_model.predict_vessels(left_data)
        right_vessel = vessel_model.predict_vessels(right_data)

        # 保存血管掩模
        left_vessel_name = left_name.replace('left', 'left_vessel')
        right_vessel_name = right_name.replace('right', 'right_vessel')

        left_vessel = vessel_enhancement(left_img, left_vessel)
        right_vessel = vessel_enhancement(right_img, right_vessel)


        left_vessel_url = upload_to_oss(left_vessel_name, left_vessel)
        right_vessel_url = upload_to_oss(right_vessel_name, right_vessel)

        # 新增视盘检测
        left_disk = disk_model.predict_disc(left_data)
        right_disk = disk_model.predict_disc(right_data)

        left_disk = cv2.bitwise_not(left_disk)
        right_disk = cv2.bitwise_not(right_disk)

        # 形态学腐蚀
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
        left_disk = cv2.dilate(left_disk, kernel)
        right_disk = cv2.dilate(right_disk, kernel)

        # 应用掩码
        left_disk = cv2.bitwise_and(left_img, left_img, mask=left_disk)
        right_disk = cv2.bitwise_and(right_img, right_img, mask=right_disk)

        left_disk = remove_black_borders(left_disk)
        right_disk = remove_black_borders(right_disk)

        left_disk = process_optic_disk(left_disk)
        right_disk = process_optic_disk(right_disk)

        left_disk = cv2.resize(left_disk, (512, 512))
        right_disk = cv2.resize(right_disk, (512, 512))

        # 保存视盘掩模（新增部分）
        left_disk_name = left_name.replace('left', 'left_disk')
        right_disk_name = right_name.replace('right', 'right_disk')

        left_disk_url = upload_to_oss(left_disk_name, left_disk)
        right_disk_url = upload_to_oss(right_disk_name, right_disk)

        return {
            "success": True,
            "predictions": result['predictions'],
            "left_heatmap_url": left_heatmap_url,
            "right_heatmap_url": right_heatmap_url,
            "left_vessel_url": left_vessel_url,
            "right_vessel_url": right_vessel_url,
            "left_disk_url": left_disk_url,
            "right_disk_url": right_disk_url,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def double_width(img):
    height, width = img.shape[:2]
    return cv2.resize(img, (width * 2, height))


def split_image(img):
    w = img.shape[1]
    return img[:, :w//2], img[:, w//2:]

def remove_black_borders(img):
    def smart_retina_preprocessing(cv_img):
        """处理视网膜图像预处理（使用OpenCV）"""
        # 转换为numpy数组后尺寸访问方式
        h, w = cv_img.shape[:2]

        # 计算填充尺寸
        if h > w:
            top = bottom = 0
            left = right = (h - w) // 2
        else:
            top = bottom = (w - h) // 2
            left = right = 0

        # 添加黑色边框
        padded = cv2.copyMakeBorder(cv_img,
                                    top, bottom,
                                    left, right,
                                    cv2.BORDER_CONSTANT,
                                    value=[0, 0, 0])
        return padded

    # 将PIL Image转换为OpenCV格式（BGR）
    cv_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 执行预处理
    cv_img = smart_retina_preprocessing(cv_img)

    # 转换为灰度图并进行阈值处理
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # 查找轮廓并找到最大轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img  # 如果没有找到轮廓返回原图
    cnt = max(contours, key=cv2.contourArea)

    # 获取边界矩形
    x, y, w, h = cv2.boundingRect(cnt)

    # 计算最大内接正方形
    square_size = max(w, h)
    if square_size < 10:
        return img

    center_x = x + w // 2
    center_y = y + h // 2

    # 计算裁剪坐标
    crop_x1 = max(0, center_x - square_size // 2)
    crop_y1 = max(0, center_y - square_size // 2)
    crop_x2 = min(cv_img.shape[1], crop_x1 + square_size)
    crop_y2 = min(cv_img.shape[0], crop_y1 + square_size)

    # 执行裁剪
    cropped = cv_img[crop_y1:crop_y2, crop_x1:crop_x2]

    # 转换回PIL Image（需要转换颜色空间）
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    return cropped_rgb


class VesselProcessor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialize_model()
        return cls._instance

    def initialize_model(self):
        """加载血管分割模型"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        self.model = VesselSegmentor().to(self.device)
        model_path = os.path.join(settings.BASE_DIR, 'backend', 'models', 'vessel_model.pth')
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def predict_vessels(self, image_data):
        """处理单张眼底图像"""
        try:
            img = analyze_image(image_data)
            img = remove_black_borders(img)
            input_tensor = self._preprocess(img)
            return self._postprocess(self._inference(input_tensor))
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None

    def _preprocess(self, img):
        """与训练一致的预处理"""
        img = cv2.resize(img, (512, 512))
        tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        return tensor.unsqueeze(0).to(self.device)

    def _inference(self, tensor):
        with torch.no_grad():
            return self.model(tensor)

    def _postprocess(self, output):
        mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def vessel_enhancement(original_img, vessel_mask):
    """
    基于血管掩膜的精准增强
    参数：
        original_img: 原始眼底图像 (BGR格式)
        vessel_mask: 血管二值掩膜 (单通道，255为血管区域)
        intensity_map: 可选强度映射图，用于自适应调整增强力度
    """
    # 1. 预处理验证
    assert original_img.shape[:2] == vessel_mask.shape[:2], "图像与掩膜尺寸不一致"

    # 2. 生成平滑权重掩膜
    kernel_size = max(original_img.shape) // 20 * 2 + 1  # 自适应核大小
    smoothed_mask = cv2.GaussianBlur(vessel_mask.astype(np.float32),
                                     (kernel_size, kernel_size), 0) / 255.0

    # 3. 多尺度血管增强
    enhanced = multi_scale_enhance(original_img)

    # 4. 自适应对比度增强（血管区域）
    enhanced_vessels = cv2.detailEnhance(enhanced, sigma_s=10, sigma_r=0.15)

    # 5. 动态融合
    final = dynamic_blend(original_img, enhanced_vessels, smoothed_mask)

    return final

def multi_scale_enhance(img):
    """修正后的LAB空间处理"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 仅增强亮度通道
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced_l = clahe.apply(l)

    # 保持a、b通道原样
    lab = cv2.merge((enhanced_l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # 注意转换方向

def dynamic_blend(base, enhanced, mask):
    """动态混合算法（修复版）"""
    # 将输入转换为float32避免溢出
    base = base.astype(np.float32)
    enhanced = enhanced.astype(np.float32)

    # 生成权重矩阵 (0.1~1.0)
    alpha = cv2.merge([mask * 0.9 + 0.1] * 3)  # 三维权重

    # 逐像素混合
    blended = base * (1 - alpha) + enhanced * alpha

    # 数值裁剪并转换回uint8
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return blended


class OpticDiscProcessor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialize_model()
        return cls._instance

    def initialize_model(self):
        """加载视盘分割模型"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        self.model = OpticDiscSegmentor().to(self.device)
        model_path = os.path.join(settings.BASE_DIR, 'backend', 'models', 'disk_model.pth')
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def predict_disc(self, image_data):
        """处理单张眼底图像的视盘检测"""
        try:
            img = analyze_image(image_data)
            img = remove_black_borders(img)
            original_shape = img.shape[:2]  # (高度, 宽度)
            input_tensor = self._preprocess(img)
            mask = self._postprocess(self._inference(input_tensor))
            return cv2.resize(mask, (original_shape[1], original_shape[0]))  # 调整回原始尺寸
        except Exception as e:
            print(f"视盘预测失败: {str(e)}")
            return None

    def _preprocess(self, img):
        """与训练一致的预处理"""
        img = cv2.resize(img, (512, 512))
        tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        return tensor.unsqueeze(0).to(self.device)

    def _inference(self, tensor):
        with torch.no_grad():
            return self.model(tensor)

    def _postprocess(self, output):
        mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

def process_optic_disk(disk_image):
    # 非锐化掩模增强细节
    blurred = cv2.GaussianBlur(disk_image, (5,5), 2)
    sharpened = cv2.addWeighted(disk_image, 1.5, blurred, -0.7, 0)

    # 自适应对比度增强（CLAHE）
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced_l = clahe.apply(l)
    enhanced_lab = cv2.merge((enhanced_l, a, b))

    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
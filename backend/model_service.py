import os
import cv2
import numpy as np
import tensorflow as tf
import keras
from django.conf import settings
from keras.src.applications.xception import Xception
from keras import layers, Model
import logging
import segmentation_models_pytorch as smp

from torch import nn

logger = logging.getLogger(__name__)

CLASS_NAMES = ['D', 'G', 'C', 'A', 'H', 'M', 'O']

IMAGE_SIZE = 299


def build_xception():
    base = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )

    # 添加与训练一致的SEBlock
    x = base.output
    x = SEBlock()(x)  # 必须包含这个层
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation='sigmoid')(x)

    return Model(inputs=base.input, outputs=outputs)


class EyeDiagnosisModel:
    _instance = None

    def __init__(self):
        self.class_names = CLASS_NAMES
        self.image_size = IMAGE_SIZE
        self.model = None
        self.heatmap_model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialize_model()
        return cls._instance

    def _build_heatmap_model(self):
        """构建热力图生成模型"""
        base = Xception(
            weights='imagenet',
            include_top=False,
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
        )

        # 获取SEBlock输出
        x = base.output
        se_output = SEBlock()(x)

        # 原始分类分支
        x_class = layers.GlobalAveragePooling2D()(se_output)
        x_class = layers.Dense(512, activation='relu')(x_class)
        x_class = layers.Dropout(0.5)(x_class)
        outputs_class = layers.Dense(len(CLASS_NAMES), activation='sigmoid')(x_class)

        # 热力图分支
        outputs_heatmap = layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(se_output)

        # 创建双输出模型
        self.heatmap_model = Model(
            inputs=base.input,
            outputs=[outputs_class, outputs_heatmap]
        )
        self.heatmap_model.set_weights(self.model.get_weights())

    def generate_heatmap(self, left_img, right_img):
        """生成热力图"""
        processed_img = self.preprocess_images(left_img, right_img)
        batch = np.expand_dims(processed_img, axis=0)

        # 获取预测结果和注意力权重
        preds, heatmap = self.heatmap_model.predict(batch)

        # 后处理热力图
        heatmap = np.squeeze(heatmap[0])
        heatmap = cv2.resize(heatmap, (self.image_size, self.image_size))
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())  # 归一化
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # 合并原始图像
        original_img = (processed_img * 255).astype(np.uint8)
        superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

        return {
            'predictions': {self.class_names[i]: float(pred) for i, pred in enumerate(preds[0])},
            'heatmap': superimposed_img,
            'heatmap_raw': heatmap.tolist()  # 原始热力图数据
        }


    def initialize_model(self):
        new_model = build_xception()
        model_path = os.path.join(settings.BASE_DIR, 'backend', 'models', 'final_model_20250319_154850.h5')
        try:
            # 尝试直接加载完整模型
            self.model = keras.models.load_model(
                model_path,
                custom_objects={
                    'weighted_bce': self.weighted_bce,
                    'MacroRecall': lambda: MacroRecall(len(CLASS_NAMES)),
                    'MacroF1': lambda: MacroF1(len(CLASS_NAMES)),
                    'SEBlock': SEBlock
                }
            )
            self._build_heatmap_model()
        except:
            print("Failed to load the full model. Falling back to loading weights.")
            # 失败时回退到权重加载
            new_model.load_weights(model_path)
            self.model = new_model


    @staticmethod
    def weighted_bce(y_true, y_pred):
        """自定义损失函数（保持与训练一致）"""
        class_weights = tf.constant([1.0] * 7)  # 实际权重应从训练数据计算
        loss = keras.losses.binary_crossentropy(y_true, y_pred)
        weights = tf.reduce_sum(class_weights * y_true, axis=-1) + 1.0
        return tf.reduce_mean(loss * weights)

    def preprocess_images(self, left_img, right_img):
        """与训练完全一致的预处理"""
        # 调整尺寸
        left = cv2.resize(left_img, (IMAGE_SIZE, IMAGE_SIZE))
        right = cv2.resize(right_img, (IMAGE_SIZE, IMAGE_SIZE))

        # 图片左右拼接并改为正方形
        img = np.concatenate([left, right], axis=1)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

        # 标准化（使用Xception专用预处理）
        return img / 255.0

    def predict_probability(self, left_img, right_img):
        """执行预测"""
        processed_img = self.preprocess_images(left_img, right_img)
        batch = np.expand_dims(processed_img, axis=0)
        predictions = self.model.predict(batch)[0]
        return {self.class_names[i]: float(pred) for i, pred in enumerate(predictions)}


class MacroRecall(keras.metrics.Metric):
    def __init__(self, num_classes, name='macro_recall', **kwargs):
        super(MacroRecall, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.recall_per_class = [keras.metrics.Recall() for _ in range(num_classes)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        for i in range(self.num_classes):
            self.recall_per_class[i].update_state(y_true[:, i], y_pred[:, i], sample_weight)

    def result(self):
        return tf.reduce_mean([recall.result() for recall in self.recall_per_class])

    def reset_state(self):
        for recall in self.recall_per_class:
            recall.reset_state()


class MacroF1(keras.metrics.Metric):
    def __init__(self, num_classes, name='macro_f1', **kwargs):
        super(MacroF1, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.precision_per_class = [keras.metrics.Precision() for _ in range(num_classes)]
        self.recall_per_class = [keras.metrics.Recall() for _ in range(num_classes)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        for i in range(self.num_classes):
            self.precision_per_class[i].update_state(y_true[:, i], y_pred[:, i], sample_weight)
            self.recall_per_class[i].update_state(y_true[:, i], y_pred[:, i], sample_weight)

    def result(self):
        f1_scores = []
        for i in range(self.num_classes):
            p = self.precision_per_class[i].result()
            r = self.recall_per_class[i].result()
            f1 = 2 * p * r / (p + r + keras.backend.epsilon())
            f1_scores.append(f1)
        return tf.reduce_mean(f1_scores)

    def reset_state(self):
        for p in self.precision_per_class:
            p.reset_state()
        for r in self.recall_per_class:
            r.reset_state()


class SEBlock(layers.Layer):
    def __init__(self, ratio=16, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.se = keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dense(self.channels // self.ratio, activation='relu'),
            layers.Dense(self.channels, activation='sigmoid'),
            layers.Reshape((1, 1, self.channels))
        ])
        super(SEBlock, self).build(input_shape)

    def call(self, inputs):
        return inputs * self.se(inputs)


class VesselSegmentor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation='sigmoid'
        )

    def forward(self, x):
        return self.model(x)


class OpticDiscSegmentor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation='sigmoid'
        )

    def forward(self, x):
        return self.model(x)

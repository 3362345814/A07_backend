import os
import cv2
import numpy as np
import tensorflow as tf
import keras
from django.conf import settings
from keras.src.applications.xception import Xception
from keras import layers, Model
import logging

logger = logging.getLogger(__name__)

CLASS_NAMES = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']

IMAGE_SIZE = 299


def build_xception():
    base = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )

    # 自定义顶层
    x = layers.GlobalAveragePooling2D()(base.output)
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

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialize_model()
        return cls._instance

    def initialize_model(self):
        new_model = build_xception()
        model_path = os.path.join(settings.BASE_DIR, 'backend', 'models', 'Xception.h5')
        print(model_path)
        try:
            # 尝试直接加载完整模型
            self.model = keras.models.load_model(
                model_path,
                custom_objects={
                    'weighted_bce': self.weighted_bce,
                    'MacroRecall': lambda: MacroRecall(len(CLASS_NAMES)),
                    'MacroF1': lambda: MacroF1(len(CLASS_NAMES))
                }
            )
        except:
            # 失败时回退到权重加载
            new_model.load_weights(model_path)
            self.model = new_model

    @staticmethod
    def weighted_bce(y_true, y_pred):
        """自定义损失函数（保持与训练一致）"""
        class_weights = tf.constant([1.0] * 8)  # 实际权重应从训练数据计算
        loss = keras.losses.binary_crossentropy(y_true, y_pred)
        weights = tf.reduce_sum(class_weights * y_true, axis=-1) + 1.0
        return tf.reduce_mean(loss * weights)

    def preprocess_images(self, left_img, right_img):
        """与训练完全一致的预处理"""
        # 调整尺寸
        left = cv2.resize(left_img, (IMAGE_SIZE, IMAGE_SIZE))
        right = cv2.resize(right_img, (IMAGE_SIZE, IMAGE_SIZE))

        # 拼接处理
        combined = np.concatenate([left, right], axis=1)
        combined = cv2.resize(combined, (IMAGE_SIZE, IMAGE_SIZE))

        # 标准化（使用Xception专用预处理）
        return keras.applications.xception.preprocess_input(combined)

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

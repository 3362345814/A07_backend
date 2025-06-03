import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 禁用所有 GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 可选：避免 onednn 优化冲突
import cv2
import keras
import numpy as np
import tensorflow as tf
from keras import Model
from keras.src.saving import load_model

from A07_backend import settings

# 配置参数
MODEL_PATH = os.path.join(settings.BASE_DIR, 'backend', 'models', 'final_model_20250319_154850.h5')  # 替换为你的模型路径
IMAGE_SIZE = 299
CLASS_NAMES = ['D', 'G', 'C', 'A', 'H', 'M', 'O']


# 以下是从原代码中复制的必要组件，确保可以独立运行
class SEBlock(keras.layers.Layer):
    def __init__(self, ratio=16, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.se = keras.Sequential([
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(self.channels // self.ratio, activation='relu'),
            keras.layers.Dense(self.channels, activation='sigmoid'),
            keras.layers.Reshape((1, 1, self.channels))
        ])
        super(SEBlock, self).build(input_shape)

    def call(self, inputs):
        return inputs * self.se(inputs)


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


def weighted_bce(y_true, y_pred):
    """加权损失函数"""
    class_counts = np.array([1000, 800, 1200, 900, 1100, 950, 850])  # 示例值，替换为实际值
    median = np.median(class_counts)
    class_weights = tf.cast(median / (class_counts + 1e-6), tf.float32)
    loss = keras.losses.binary_crossentropy(y_true, y_pred)
    weights = tf.reduce_sum(class_weights * y_true, axis=-1) + tf.reduce_sum(1.0 * (1 - y_true), axis=-1)
    return tf.reduce_mean(loss * weights)


class LayerCAM:
    def __init__(self):
        """
        初始化LayerCAM可视化器

        参数:
            model: 加载的Keras模型
            target_layer_names: 要可视化的目标层名称列表
        """
        # 加载模型
        model = load_model(MODEL_PATH, custom_objects={
            'SEBlock': SEBlock,
            'weighted_bce': weighted_bce,
            'MacroRecall': lambda: MacroRecall(len(CLASS_NAMES)),
            'MacroF1': lambda: MacroF1(len(CLASS_NAMES)),
        })
        # 定义要可视化的目标层
        target_layer_names = [
            'block1_conv1',  # 浅层特征
            'block3_sepconv2_act',
            'block4_sepconv2_act',  # 中层特征
            'block8_sepconv2_act',  # 中深层特征
            'block12_sepconv3_act',  # 深层特征
        ]
        self.model = model
        self.target_layers = [model.get_layer(name) for name in target_layer_names]

        # 创建模型输出和指定层的梯度计算图
        self.grad_model = Model(
            inputs=model.inputs,
            outputs=[model.output] + [layer.output for layer in self.target_layers]
        )

    def compute_heatmaps(self, left_image, right_image, threshold=0.5, eps=1e-8):
        """
        计算所有概率大于threshold的类别的LayerCAM热力图

        参数:
            image: 输入图像(预处理后的)
            threshold: 概率阈值
            eps: 防止除零的小常数

        返回:
            字典: {
                'predictions': 预测概率数组,
                'heatmaps': {
                    class_name: {
                        layer_name: heatmap
                    }
                }
            }
        """
        left_image = cv2.resize(left_image, (IMAGE_SIZE // 2 + 1, IMAGE_SIZE))
        right_image = cv2.resize(right_image, (IMAGE_SIZE // 2, IMAGE_SIZE))
        image = np.concatenate([left_image, right_image], axis=1)
        original_image = image.copy()
        image = image / 255.0
        # 转换为batch形式
        img_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            outputs = self.grad_model(img_tensor)
            preds = outputs[0]
            layer_outputs = outputs[1:]

            # 获取所有概率大于阈值的类别
            active_classes = tf.where(preds[0] > threshold)

            # 存储结果
            results = {
                'predictions': {CLASS_NAMES[i]: float(pred) for i, pred in enumerate(preds[0].numpy())},
                'heatmaps': {}
            }

            # 为每个激活的类别计算热力图
            for class_idx in active_classes:
                class_idx = class_idx.numpy()[0]
                class_name = CLASS_NAMES[class_idx]
                results['heatmaps'][class_name] = {}

                # 计算该类别的梯度
                class_score = preds[:, class_idx]
                grads = tape.gradient(class_score, layer_outputs)

                # 为每个目标层计算热力图
                for i, (layer_name, layer_output, grad) in enumerate(
                        zip([layer.name for layer in self.target_layers], layer_outputs, grads)):
                    # LayerCAM核心计算
                    weights = tf.nn.relu(grad)  # 使用ReLU过滤负梯度
                    weighted_output = weights * layer_output

                    # 沿通道维度求和
                    heatmap = tf.reduce_sum(weighted_output, axis=-1)

                    # 归一化处理
                    heatmap = tf.squeeze(heatmap).numpy()
                    heatmap = np.maximum(heatmap, 0)  # ReLU
                    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + eps)

                    # 调整大小到输入图像尺寸
                    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
                    heatmap = np.uint8(255 * heatmap)
                    # 和原图叠加
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    heatmap = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)

                    results['heatmaps'][class_name][layer_name] = heatmap

            return results

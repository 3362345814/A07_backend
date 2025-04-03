import base64
import hashlib
import urllib.parse
from datetime import datetime, timezone, timedelta

import cv2
import numpy as np
import qrcode
import requests
from django.conf import settings
from jinja2 import Environment, FileSystemLoader
from jinja2.exceptions import TemplateSyntaxError, UndefinedError

from backend.service.oss_utils import OSSUtils


class QRService:
    def __init__(self):
        self.oss_utils = OSSUtils()
        self.caoliao_api_key = settings.CAOLIAO_API_KEY
        self.caoliao_api_secret = settings.CAOLIAO_API_SECRET

    def change_class_name(self, class_name):
        class_map = {
            "D": "糖尿病视网膜病变",
            "H": "高血压视网膜病变",
            "G": "青光眼",
            "C": "白内障",
            "A": "年龄相关性黄斑变性",
            "M": "近视性视网膜病变",
            "O": "其他视网膜病变"
        }
        return class_map.get(class_name, class_name)

    def generate_report_html(self, report_data, name, age, gender, report_name):
        try:
            # 使用Jinja2模板引擎
            env = Environment(loader=FileSystemLoader('templates'))
            template = env.get_template('patient_report.html')

            # 转换为东八区时间
            report_data['timestamp'] = datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')
            report_data['name'] = name
            report_data['age'] = age
            report_data['gender'] = '男' if gender == '0' else '女'

            # 渲染HTML
            html_content = template.render(report_data, changeClassName=self.change_class_name)

            report_url = self.oss_utils.upload_html_to_oss(report_name, html_content)

            return report_url
        except Exception as e:
            # Enhanced error reporting
            if isinstance(e, TemplateSyntaxError):
                error_msg = f"Template error on line {e.lineno}: {e.message}"
            elif isinstance(e, UndefinedError):
                error_msg = f"Undefined variable: {e.message}"
            else:
                error_msg = str(e)

            raise RuntimeError(f"生成报告失败: {error_msg}")

    def generate_qrcode(self, url, qr_name):
        qr = qrcode.main.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")

        img_np = np.array(img.convert('RGB'))  # 转换为RGB格式的NumPy数组
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # 上传到OSS
        qr_url = self.oss_utils.upload_to_oss(qr_name, img_np)
        return qr_url

    def generate_caoliao_qrcode(self, url, qr_name, name, age, gender):
        params = {
            'api_key': self.caoliao_api_key,
            'return_file': 'base64',
            'cliT': 'D1',
            'cliD': url,  # 二维码动态内容
            'cliF1': '眼底检查单',
            'cliF2': name,
            'cliF3': age,
            'cliF4': '男' if gender == '0' else '女',
            'cliF5': datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')
        }

        sorted_params = sorted(params.items(), key=lambda x: x[0])
        query_string = '&'.join([f"{k}={v}" for k, v in sorted_params])
        sign_string = query_string + self.caoliao_api_secret
        sign = hashlib.md5(sign_string.encode('utf-8')).hexdigest()

        # 构建请求URL
        base_url = 'https://open-api.cli.im/cli-open-platform-service/v1/labelStyle/createWithKey'
        encoded_params = {k: urllib.parse.quote_plus(str(v)) for k, v in params.items()}
        url = f"{base_url}?{'&'.join([f'{k}={v}' for k, v in encoded_params.items()])}&sign={sign}"

        # 发送请求
        response = requests.get(url)

        img_data = response.text.split('base64,')[-1]

        img_bytes = base64.b64decode(img_data)
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("图片解码失败")

        # 上传到OSS
        qr_url = self.oss_utils.upload_to_oss(qr_name, img)
        return qr_url

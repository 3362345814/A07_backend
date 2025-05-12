import cv2
import oss2
from django.conf import settings


class OSSUtils:
    def __init__(self):
        self.auth = oss2.Auth(settings.OSS_ACCESS_KEY_ID, settings.OSS_ACCESS_KEY_SECRET)
        self.bucket = oss2.Bucket(self.auth, settings.OSS_ENDPOINT, settings.OSS_BUCKET_NAME)

    def download_from_oss(self, url):
        """
        从OSS下载文件到内存
        :param url: OSS文件完整URL
        :return: 文件二进制内容
        """
        # 修改为从自定义域名中提取object key（假设绑定的域名是cdn.example.com）
        object_key = url.split(f'{settings.OSS_CUSTOM_DOMAIN}/')[-1]
        try:
            result = self.bucket.get_object(object_key)
            return result.read()
        except oss2.exceptions.NoSuchKey:
            raise ValueError("OSS文件不存在")
        except oss2.exceptions.OssError as e:
            raise RuntimeError(f"OSS访问失败: {str(e)}")

    def upload_to_oss(self, img_name, image):
        try:
            _, buf = cv2.imencode('.jpg', image)
            self.bucket.put_object(img_name, buf.tobytes())
            # 修改为返回自定义域名格式的URL
            return f"https://{settings.OSS_CUSTOM_DOMAIN}/{img_name}"
        except oss2.exceptions.OssError as e:
            raise RuntimeError(f"OSS上传失败: {str(e)}")

    def upload_html_to_oss(self, html_name, html_content):
        try:
            self.bucket.put_object(html_name, html_content, headers={
                'Content-Type': 'text/html; charset=utf-8',
                'Content-Disposition': 'inline',
                'Cache-Control': 'no-cache',
            })
            # 修改为返回自定义域名格式的URL
            return f"https://{settings.OSS_CUSTOM_DOMAIN}/{html_name}"
        except oss2.exceptions.OssError as e:
            raise RuntimeError(f"OSS上传失败: {str(e)}")

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
        # 从URL中提取object key
        object_key = url.split(f'https://{settings.OSS_BUCKET_NAME}.{settings.OSS_ENDPOINT}/')[-1]
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
            return f"https://{settings.OSS_BUCKET_NAME}.{settings.OSS_ENDPOINT}/{img_name}"
        except oss2.exceptions.OssError as e:
            raise RuntimeError(f"OSS上传失败: {str(e)}")

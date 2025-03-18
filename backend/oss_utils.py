import oss2
from django.conf import settings

def download_from_oss(url):
    """
    从OSS下载文件到内存
    :param url: OSS文件完整URL
    :return: 文件二进制内容
    """
    auth = oss2.Auth(settings.OSS_ACCESS_KEY_ID, settings.OSS_ACCESS_KEY_SECRET)
    bucket = oss2.Bucket(auth, settings.OSS_ENDPOINT, settings.OSS_BUCKET_NAME)

    # 从URL中提取object key
    object_key = url.split(f'https://{settings.OSS_BUCKET_NAME}.{settings.OSS_ENDPOINT}/')[-1]
    try:
        result = bucket.get_object(object_key)
        return result.read()
    except oss2.exceptions.NoSuchKey:
        raise ValueError("OSS文件不存在")
    except oss2.exceptions.OssError as e:
        raise RuntimeError(f"OSS访问失败: {str(e)}")
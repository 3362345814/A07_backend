import json

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .oss_utils import download_from_oss
from .img_processor import process_images
import logging

logger = logging.getLogger(__name__)

@csrf_exempt
@require_http_methods(["POST"])
def process_medical_images(request):
    try:
        # 解析请求体数据
        if request.content_type == 'application/json':
            data = json.loads(request.body)
        else:
            data = request.POST

        # 获取参数
        left_url = data.get('left_url')
        right_url = data.get('right_url')

        # 参数校验
        if not all([left_url, right_url]):
            return JsonResponse({
                "success": False,
                "error": "Missing image URLs"
            }, status=400)

        # 下载和处理逻辑保持不变
        left_image = download_from_oss(left_url)
        right_image = download_from_oss(right_url)
        result = process_images(left_image, right_image)

        if not result['success']:
            return JsonResponse(result, status=500)

        return JsonResponse({
            "success": True,
            "message": result["detail"]
        })

    except json.JSONDecodeError:
        logger.error("Invalid JSON format")
        return JsonResponse({"success": False, "error": "Invalid JSON"}, status=400)
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        return JsonResponse({
            "success": False,
            "message": "Internal server error"
        }, status=500)
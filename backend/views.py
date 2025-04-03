import json
import logging

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from backend.service.image_process import ImageProcess
from backend.service.oss_utils import OSSUtils

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
        name = data.get('name')
        age = data.get('age')
        gender = data.get('gender')

        left_name = left_url.split('/')[-1]
        right_name = right_url.split('/')[-1]

        # 参数校验
        if not all([left_url, right_url]):
            return JsonResponse({
                "success": False,
                "error": "Missing image URLs"
            }, status=400)

        oss_utils = OSSUtils()
        left_image = oss_utils.download_from_oss(left_url)
        right_image = oss_utils.download_from_oss(right_url)
        result = ImageProcess().process_images(left_image, right_image, left_name, right_name, left_url, right_url,
                                               name, age, gender)

        if not result['success']:
            return JsonResponse(result, status=500)

        return JsonResponse({
            "success": True,
            "message": {
                "predictions": result['predictions'],
                "images": result['images'],
                "suggestions": result['suggestions'],
                "drugs": result['drugs'],
                "report_html": result['report_html'],
                "qr_code": result['qr_code'],
            }
        })

    except json.JSONDecodeError:
        logger.error("Invalid JSON format")
        return JsonResponse({"success": False, "message": "Invalid JSON"}, status=400)
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        return JsonResponse({
            "success": False,
            "message": "Internal server error"
        }, status=500)

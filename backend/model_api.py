import json

from openai import OpenAI

from A07_backend.settings import MODEL_API_KEY

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=MODEL_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def get_suggestions(left_url, right_url, predictions):
    completion = client.chat.completions.create(
        model="qwen-vl-max-0125",
        messages=[
            {"role": "system", "content": [
                {"type": "text", "text": "你是一个专业的眼科医生，你需要根据以下两张眼底图的诊断结果，给出你对这两张图片的分析结果，"
                                         "格式保持为json格式，如下：{'0': '黄斑区出现微小出血点',"
                                         "'1': '视网膜血管轻度扭曲'"
                                         "'2': '视盘边界清晰'}"
                                         "对图片给出三条分析结果。"}
            ]},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": left_url}, },
                {"type": "image_url", "image_url": {"url": right_url}, },
                {"type": "text", "text": f"诊断结果为：{predictions}，其中D为糖尿病，G为青光眼，C为白内障，A为老年老年性黄斑变性，"
                                         f"H为高血压，M为近视，O为其他疾病，数字表示患病的概率，请你分析图片给出三条分析结果，"
                                         f"诊断结果仅供判断该患者患病种类，不要出现在你的分析结果中，分析结果不要出现数字。"},
            ]}
        ]
    )
    result = completion.choices[0].message.content
    # 去除{}前后的内容
    result = result[result.find('{'):result.rfind('}') + 1]
    result = result.replace("'", "\"")
    try:
        result = json.loads(result)
        return {"success": True, **result}
    except json.JSONDecodeError:
        print(result)
        return {"success": False}




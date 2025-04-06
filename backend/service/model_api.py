import json

from openai import OpenAI

from A07_backend.settings import MODEL_API_KEY


class ModelApi:
    def __init__(self):
        self.client = OpenAI(
            api_key=MODEL_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def parse_json_result(self, completion):
        result = completion.choices[0].message.content
        result = result[result.find('{'):result.rfind('}') + 1]
        result = result.replace("'", "\"")
        try:
            result = json.loads(result)
            return {"success": True, **result}
        except json.decoder.JSONDecodeError:
            return {"success": False}

    def get_suggestions(self, left_url, right_url, predictions):
        completion = self.client.chat.completions.create(
            model="qwen2.5-vl-72b-instruct",
            messages=[
                {"role": "system", "content": [
                    {"type": "text", "text": "你是一个专业的眼科医生，你需要根据以下两张眼底图的诊断结果，给出你对这两张图片的分析结果和推荐复诊时间，单位为天"
                                             "格式保持为json格式，如下："
                                             "{'suggestions': ['黄斑区出现微小出血点',"
                                             "'视网膜血管轻度扭曲',"
                                             "'视盘边界清晰'],"
                                             "'revisit_time': 5}"
                                             "对图片给出三条分析结果。"}
                ]},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": left_url}, },
                    {"type": "image_url", "image_url": {"url": right_url}, },
                    {"type": "text",
                     "text": f"诊断结果为：{predictions}，其中D为糖尿病，G为青光眼，C为白内障，A为老年老年性黄斑变性，"
                             f"H为高血压，M为近视，O为其他疾病，数字表示患病的概率，请你分析图片给出三条分析结果，"
                             f"诊断结果仅供判断该患者患病种类，不要出现在你的分析结果中，分析结果不要出现数字。"},
                ]}
            ]
        )
        return self.parse_json_result(completion)

    def get_drugs(self, predictions):
        completion = self.client.chat.completions.create(
            model="qwen2.5-14b-instruct-1m",
            messages=[
                {"role": "system", "content": [
                    {"type": "text", "text": "你是一个专业的眼科医生，你需要根据患者患病的种类，给出所有推荐的药物，并且给出药物对应的功能。"
                                             "格式保持为json格式，如下："
                                             "{drugs: [{'function': '功能1', 'drug': ['药物1', '药物2', '药物3']},"
                                             "{'function': '功能2', 'drug': ['药物1', '药物2', '药物3']},"
                                             "{'function': '功能3', 'drug': ['药物1', '药物2', '药物3']}]}"
                                             "每一个功能最多3种药物，最多3种功能，功能控制在8个字以内。"}
                ]},
                {"role": "user", "content": [
                    {"type": "text",
                     "text": f"诊断结果为：{predictions}，其中D为糖尿病，G为青光眼，C为白内障，A为老年老年性黄斑变性，"
                             f"H为高血压，M为近视，O为其他疾病，数字表示患病的概率，请给出药物功能和推荐药物。"},
                ]}
            ]
        )
        return self.parse_json_result(completion)

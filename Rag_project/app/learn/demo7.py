import sys
import os
import json
from pathlib import Path
import chromadb
from openai import OpenAI

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from demo5 import get_embeddings, collection


llm_client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY", "sk-167fff19c26f401fad8277f93a21ee35"),
    base_url="https://api.deepseek.com"
)


# ─────────────────────────────────────────
# 1. 语义检索
# ─────────────────────────────────────────
def retrieve_standards(pollution_items: list[dict], top_k: int = 15) -> str:
    """
    每条数据单独检索，若有 pdf_id 则只在对应文件里查，否则全库查。
    pollution_items 格式：
      [{"name": "烟尘", "pdf_id": "GB_13223"}, ...]
    """
    seen_ids: set[str] = set()
    contexts: list[str] = []

    for item in pollution_items:
        name = item["name"]
        pdf_id = item.get("pdf_id")

        query = f"{name} 排放限值 浓度 mg/m³"
        query_embedding = get_embeddings([query])[0]

        kwargs = dict(query_embeddings=[query_embedding], n_results=top_k)
        if pdf_id:
            kwargs["where"] = {"pdf_id": pdf_id}

        results = collection.query(**kwargs)

        for doc_id, doc, meta in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0]
        ):
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                source = meta.get("pdf_name", "未知标准")
                contexts.append(f"【来源：{source}】\n{doc}")

    return "\n\n".join(contexts)


# ─────────────────────────────────────────
# 2. 调用 LLM
# ─────────────────────────────────────────
def call_llm(prompt: str) -> str:
    response = llm_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": "你是工业排放合规专家，严格根据提供的标准文档内容进行判断，不要凭空推测限值。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────
# 3. 合规判断主函数
# ─────────────────────────────────────────
def check_compliance(pollution_data: list[dict]) -> dict:
    """
    输入格式：
    [
        {"name": "烟尘",    "value": 25,  "unit": "mg/Nm³"},
        {"name": "二氧化硫", "value": 250, "unit": "mg/Nm³"},
    ]
    """
    context = retrieve_standards(pollution_data)

    data_lines = "\n".join([
        f"- {d['name']}：实测值 {d['value']} {d['unit']}"
        for d in pollution_data
    ])

    prompt = f"""
请根据下方标准文档，逐一判断每条污染物数据是否合规。

【标准文档摘录】
{context}

【待判断数据】
{data_lines}

判断规则：
1. 在标准文档中找到对应污染物的排放限值
2. 实测值 > 限值 → 不合规；实测值 ≤ 限值 → 合规
3. 如果文档中找不到该污染物的限值，compliant 字段填 "unknown"
4. basis 字段填写限值来自哪个标准的哪张表，例如"GB 13223—2011 表1"

请严格按以下 JSON 格式输出，不要输出任何其他内容：
{{
  "results": [
    {{
      "name": "污染物名称",
      "measured_value": 实测数值,
      "unit": "单位",
      "standard_limit": 标准限值数值（找不到填null）,
      "compliant": true或false或"unknown",
      "basis": "依据来源"
    }}
  ],
  "summary": "整体结论，几条达标几条超标"
}}
"""

    raw = call_llm(prompt)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "LLM返回格式异常", "raw": raw}


# ─────────────────────────────────────────
# 4. 测试入口
# ─────────────────────────────────────────
if __name__ == "__main__":
    test_data = [
        # 指定 pdf_id：只在对应文件里检索
        {"name": "烟尘",    "value": 25,  "unit": "mg/Nm³", "pdf_id": "GB_13223"},
        {"name": "二氧化硫", "value": 450, "unit": "mg/Nm³", "pdf_id": "GB_13223"},
        # 不指定 pdf_id：全库检索
        {"name": "氮氧化物", "value": 80,  "unit": "mg/Nm³"},
    ]

    result = check_compliance(test_data)
    print(json.dumps(result, ensure_ascii=False, indent=2))

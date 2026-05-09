from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

# 准备一段长文本
text = """
第一章 大气污染物排放标准

1.1 氮氧化物排放标准
氮氧化物是大气污染的主要成分之一。工业企业排放的氮氧化物浓度不得超过100mg/m³。
超过此标准的企业将面临处罚，情节严重者将被责令停产整改。
监测频次要求每季度至少检测一次，并向环保部门报告检测结果。

1.2 二氧化硫排放标准  
二氧化硫主要来源于煤炭燃烧和工业生产过程。
排放浓度限值为50mg/m³，重点控制区域执行更严格的30mg/m³标准。
企业需安装在线监测设备，实时上传监测数据。

1.3 颗粒物排放标准
颗粒物包括PM2.5和PM10两类指标。
PM2.5年均浓度限值为35微克每立方米，日均浓度限值为75微克每立方米。
PM10年均浓度限值为70微克每立方米，日均浓度限值为150微克每立方米。
工业企业厂界颗粒物排放不得超过1.0mg/m³。
"""

chunks = splitter.split_text(text)

print(f"共切出 {len(chunks)} 块\n")
for i, chunk in enumerate(chunks):
    print(f"--- 第{i+1}块（{len(chunk)}字）---")
    print(chunk)
    print()


import pdfplumber
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _should_keep_line(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    # 过滤页眉页脚，如 "2GB 13223—2011"、"ivGB 13223—2011"
    if re.match(r'^[ivxIVX\d]{0,4}(GB|HJ)\s*\d+', line):
        return False
    # 过滤目录行（大量省略号）
    if line.count('.') > 8:
        return False
    # 过滤独立页码
    if re.match(r'^[ivxIVX\d]{1,5}$', line):
        return False
    return True


def _forward_fill(rows: list[list]) -> list[list]:
    """向下填充空白单元格（处理PDF合并单元格）。"""
    if not rows:
        return rows
    col_count = max(len(r) for r in rows)
    padded = [list(r) + [None] * (col_count - len(r)) for r in rows]
    prev = [""] * col_count
    filled = []
    for row in padded:
        new_row = []
        for i, cell in enumerate(row):
            val = str(cell).strip() if cell is not None else ""
            new_row.append(val if val else prev[i])
        filled.append(new_row)
        prev = new_row
    return filled


def _extract_table_meta(page_text: str) -> tuple[str, str]:
    """从页面文本中提取表名和单位。"""
    table_match = re.search(r'(表\s*\d+\s+[^\n]{2,40})', page_text)
    unit_match = re.search(r'单位[：:]\s*([^\s\n，]{2,20})', page_text)
    table_name = table_match.group(1).strip() if table_match else ""
    unit = unit_match.group(1).strip() if unit_match else ""
    return table_name, unit


def _table_to_chunks(rows: list[list], table_label: str, unit: str) -> list[str]:
    """将表格每行转换为自包含的结构化文本块。"""
    if len(rows) < 2:
        return []

    filled = _forward_fill(rows)
    headers = filled[0]
    unit_str = f"，单位：{unit}" if unit else ""
    prefix = f"【{table_label}{unit_str}】"

    chunks = []
    for row in filled[1:]:
        if row == headers or not any(row):
            continue

        parts = []
        for h, v in zip(headers, row):
            h, v = h.strip(), v.strip()
            if v and v != h:
                parts.append(f"{h}：{v}" if h else v)

        if len(parts) >= 2:
            chunks.append(prefix + "，".join(parts))

    return chunks


def parse_and_chunk(pdf_path: str, pdf_name: str = "") -> list[str]:
    """
    解析PDF并返回结构化文本块。
    - 表格：每行转为带完整上下文的句子（含表名、单位、所有字段）
    - 正文：按语义切块，过滤页眉页脚噪音
    """
    table_chunks: list[str] = []
    text_sections: list[str] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            raw_text = page.extract_text() or ""
            tables = page.extract_tables()

            if tables:
                table_name, unit = _extract_table_meta(raw_text)
                label = f"{pdf_name} {table_name}".strip() if pdf_name else table_name
                for table in tables:
                    table_chunks.extend(_table_to_chunks(table, label, unit))

            clean_lines = [l for l in raw_text.splitlines() if _should_keep_line(l)]
            if clean_lines:
                text_sections.append("\n".join(clean_lines))

    text_chunks: list[str] = []
    if text_sections:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=80,
            separators=["\n\n", "\n", "。", "；", "，"],
        )
        raw_chunks = splitter.split_text("\n\n".join(text_sections))
        if pdf_name:
            raw_chunks = [f"【{pdf_name}】{c}" for c in raw_chunks]
        text_chunks = raw_chunks

    return table_chunks + text_chunks


# ─── 测试入口 ────────────────────────────────────────────────────
if __name__ == "__main__":
    pdf_path = "..//Rag_project//app//file//GB 13223 火电厂大气污染物排放标准.pdf"
    pdf_name = "GB 13223—2011 火电厂大气污染物排放标准"

    chunks = parse_and_chunk(pdf_path, pdf_name)

    print(f"共生成 {len(chunks)} 个文本块\n")
    print("── 前5个表格块 ──")
    for i, c in enumerate(chunks[:5], 1):
        print(f"\n块 #{i}:\n{c}")

    with open("chunks_output.txt", "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks, 1):
            f.write(f"\n{'='*60}\n")
            f.write(f"块 #{i} (长度: {len(chunk)})\n")
            f.write(f"{'='*60}\n")
            f.write(chunk)
            f.write("\n")

    print(f"\n✅ 已保存 {len(chunks)} 个文本块到 chunks_output.txt")

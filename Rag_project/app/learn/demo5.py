"""
PDF 向量库管理模块

单个 Collection 存所有 PDF，每条记录带 pdf_id 元数据。
支持：新增、更新、删除、列表查询。
"""
import sys
import os
import json
from pathlib import Path
import chromadb
from openai import OpenAI

_HERE = Path(__file__).resolve().parent
_CHROMA_PATH = str(_HERE.parent.parent / "chroma_db")
_PDF_DIR = _HERE.parent / "file"
_REGISTRY_PATH = _HERE.parent.parent / "chroma_db" / "pdf_registry.json"

chroma_client = chromadb.PersistentClient(path=_CHROMA_PATH)
collection = chroma_client.get_or_create_collection("emission_standards")

open_client = OpenAI(
    api_key=os.environ.get("DASHSCOPE_API_KEY", "sk-088021142fda42b9bf965d3fdafd65f9"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


# ─── 注册表（记录已入库的 PDF 信息）────────────────────────────
def _load_registry() -> dict:
    if _REGISTRY_PATH.exists():
        return json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
    return {}

def _save_registry(registry: dict):
    _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REGISTRY_PATH.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")


# ─── Embedding ──────────────────────────────────────────────────
def get_embeddings(chunks: list[str]) -> list[list[float]]:
    response = open_client.embeddings.create(
        model="text-embedding-v2",
        input=chunks
    )
    return [item.embedding for item in response.data]


# ─── 增删改查 ────────────────────────────────────────────────────
def add_pdf(pdf_id: str, pdf_name: str, chunks: list[str]):
    """新增 PDF（pdf_id 已存在则报错，请用 update_pdf）。"""
    registry = _load_registry()
    if pdf_id in registry:
        raise ValueError(f"pdf_id '{pdf_id}' 已存在，请用 update_pdf() 更新")

    _write_chunks(pdf_id, pdf_name, chunks)
    registry[pdf_id] = {"pdf_name": pdf_name, "chunk_count": len(chunks)}
    _save_registry(registry)
    print(f"✅ 新增：{pdf_name}（{len(chunks)} 块）")


def update_pdf(pdf_id: str, pdf_name: str, chunks: list[str]):
    """更新已有 PDF：先删除旧数据，再写入新数据。"""
    registry = _load_registry()
    if pdf_id in registry:
        collection.delete(where={"pdf_id": pdf_id})
        print(f"🗑  已删除旧数据：{registry[pdf_id]['pdf_name']}")

    _write_chunks(pdf_id, pdf_name, chunks)
    registry[pdf_id] = {"pdf_name": pdf_name, "chunk_count": len(chunks)}
    _save_registry(registry)
    print(f"✅ 更新完成：{pdf_name}（{len(chunks)} 块）")


def delete_pdf(pdf_id: str):
    """从向量库和注册表中删除指定 PDF。"""
    registry = _load_registry()
    if pdf_id not in registry:
        print(f"⚠️  pdf_id '{pdf_id}' 不存在")
        return
    collection.delete(where={"pdf_id": pdf_id})
    pdf_name = registry.pop(pdf_id)["pdf_name"]
    _save_registry(registry)
    print(f"🗑  已删除：{pdf_name}")


def list_pdfs():
    """打印当前向量库中所有 PDF。"""
    registry = _load_registry()
    if not registry:
        print("向量库为空")
        return
    print(f"{'pdf_id':<20} {'文本块数':>6}  pdf_name")
    print("-" * 60)
    for pid, info in registry.items():
        print(f"{pid:<20} {info['chunk_count']:>6}  {info['pdf_name']}")


def _write_chunks(pdf_id: str, pdf_name: str, chunks: list[str]):
    """批量写入，每次最多 500 条（避免单次请求过大）。"""
    batch_size = 500
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start: start + batch_size]
        embeddings = get_embeddings(batch)
        collection.add(
            ids=[f"{pdf_id}_{start + i}" for i in range(len(batch))],
            documents=batch,
            embeddings=embeddings,
            metadatas=[{"pdf_id": pdf_id, "pdf_name": pdf_name} for _ in batch]
        )


# ─── 命令行入口 ──────────────────────────────────────────────────
if __name__ == "__main__":
    sys.path.insert(0, str(_HERE))
    from demo2 import parse_and_chunk
    import argparse

    parser = argparse.ArgumentParser(description="PDF 向量库管理工具")
    sub = parser.add_subparsers(dest="cmd")

    # 子命令：add / update / delete / list
    for cmd in ("add", "update"):
        p = sub.add_parser(cmd)
        p.add_argument("pdf_id", help="唯一标识符，如 GB_13223")
        p.add_argument("pdf_name", help="显示名称，如 'GB 13223—2011 火电厂排放标准'")
        p.add_argument("pdf_file", help="PDF 文件路径")

    p_del = sub.add_parser("delete")
    p_del.add_argument("pdf_id")

    sub.add_parser("list")

    args = parser.parse_args()

    if args.cmd in ("add", "update"):
        pdf_path = Path(args.pdf_file)
        if not pdf_path.exists():
            print(f"❌ 文件不存在：{pdf_path}")
            sys.exit(1)
        print(f"正在解析：{pdf_path.name} ...")
        chunks = parse_and_chunk(str(pdf_path), args.pdf_name)
        if args.cmd == "add":
            add_pdf(args.pdf_id, args.pdf_name, chunks)
        else:
            update_pdf(args.pdf_id, args.pdf_name, chunks)

    elif args.cmd == "delete":
        delete_pdf(args.pdf_id)

    elif args.cmd == "list":
        list_pdfs()

    else:
        parser.print_help()

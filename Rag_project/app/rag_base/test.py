#!/usr/bin/env python3

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path

import httpx

# base_url
BASE_URL = "https://onetoken.one"
API_PATH = "/v1/chat/completions"
# key
API_KEY = "sk-nMorkI9vSXsNORHwtnNBTwB0WYk8wEEHi2KY5qTnhddbG2CH"
MODEL = "gpt-5.4"
QUESTION = "你是什么模型？能干啥"
CONCURRENCY = 5
#总次数
TOTAL = 10

PROJECT_ROOT = Path(__file__).resolve().parent
DOC_DIR = PROJECT_ROOT / "docs"
DOC_DIR.mkdir(parents=True, exist_ok=True)


def build_payload() -> dict:
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": QUESTION}],
        "stream": False,
    }


def build_headers() -> dict:
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }


async def single_request(client: httpx.AsyncClient, idx: int) -> dict:
    start = time.perf_counter()
    started_at = datetime.now().isoformat(timespec="seconds")
    try:
        resp = await client.post(
            BASE_URL + API_PATH,
            headers=build_headers(),
            json=build_payload(),
            timeout=120.0,
        )
        elapsed = time.perf_counter() - start
        status = resp.status_code
        try:
            data = resp.json()
        except Exception:
            data = {"_raw": resp.text}

        content = ""
        usage = {}
        if isinstance(data, dict):
            choices = data.get("choices") or []
            if choices and isinstance(choices, list):
                msg = choices[0].get("message") or {}
                content = msg.get("content") or ""
            usage = data.get("usage") or {}

        return {
            "idx": idx,
            "started_at": started_at,
            "elapsed_sec": round(elapsed, 3),
            "status_code": status,
            "ok": status == 200 and bool(content),
            "content": content,
            "usage": usage,
            "error": None if status == 200 else json.dumps(data, ensure_ascii=False),
        }
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return {
            "idx": idx,
            "started_at": started_at,
            "elapsed_sec": round(elapsed, 3),
            "status_code": -1,
            "ok": False,
            "content": "",
            "usage": {},
            "error": f"{type(exc).__name__}: {exc}",
        }


async def run_warmup() -> dict:
    print("=" * 60)
    print(f"[预热] 单次请求 -> {BASE_URL}{API_PATH}  model={MODEL}")
    print("=" * 60)
    async with httpx.AsyncClient() as client:
        result = await single_request(client, idx=0)
    print(f"状态: {result['status_code']}  用时: {result['elapsed_sec']}s  ok={result['ok']}")
    if result["content"]:
        print(f"回复: {result['content'][:200]}")
    if result["error"]:
        print(f"错误: {result['error']}")
    print(f"usage: {result['usage']}")
    return result


async def run_load(total: int, concurrency: int, warmup: dict | None = None) -> tuple[list[dict], float]:
    print()
    print("=" * 60)
    print(f"[压测] 并发={concurrency}  总数={total}" + ("  (含预热)" if warmup else ""))
    print("=" * 60)
    sem = asyncio.Semaphore(concurrency)
    results: list[dict] = []
    done_counter = {"n": 0}

    start_idx = 1
    if warmup:
        warmup_entry = dict(warmup)
        warmup_entry["idx"] = 1
        results.append(warmup_entry)
        done_counter["n"] += 1
        print(f"[{done_counter['n']:>3}/{total}] #  1 OK {warmup_entry['elapsed_sec']:>6}s status={warmup_entry['status_code']}  (复用预热)")
        start_idx = 2

    if start_idx > total:
        return results, 0.0

    limits = httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(limits=limits) as client:
        async def worker(i: int):
            async with sem:
                r = await single_request(client, idx=i)
                results.append(r)
                done_counter["n"] += 1
                tag = "OK" if r["ok"] else "ERR"
                print(f"[{done_counter['n']:>3}/{total}] #{i:>3} {tag} {r['elapsed_sec']:>6}s status={r['status_code']}")

        wall_start = time.perf_counter()
        await asyncio.gather(*(worker(i) for i in range(start_idx, total + 1)))
        wall = time.perf_counter() - wall_start

    results.sort(key=lambda x: x["idx"])
    print(f"\n总墙钟用时(不含预热): {wall:.3f}s")
    return results, wall


def summarize(results: list[dict]) -> dict:
    elapsed = [r["elapsed_sec"] for r in results]
    ok_n = sum(1 for r in results if r["ok"])
    err_n = len(results) - ok_n

    def pct(p: float) -> float:
        if not elapsed:
            return 0.0
        s = sorted(elapsed)
        k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
        return s[k]

    return {
        "total": len(results),
        "ok": ok_n,
        "fail": err_n,
        "min_sec": min(elapsed) if elapsed else 0,
        "max_sec": max(elapsed) if elapsed else 0,
        "avg_sec": round(sum(elapsed) / len(elapsed), 3) if elapsed else 0,
        "p50_sec": pct(50),
        "p90_sec": pct(90),
        "p95_sec": pct(95),
        "p99_sec": pct(99),
    }


def write_report(warmup: dict, results: list[dict], wall: float, summary: dict) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = DOC_DIR / f"test_concurrent_report_{ts}.md"
    json_path = DOC_DIR / f"test_concurrent_report_{ts}.json"

    json_path.write_text(
        json.dumps(
            {
                "config": {
                    "base_url": BASE_URL,
                    "path": API_PATH,
                    "model": MODEL,
                    "question": QUESTION,
                    "concurrency": CONCURRENCY,
                    "total": TOTAL,
                },
                "warmup": warmup,
                "wall_sec": round(wall, 3),
                "summary": summary,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines: list[str] = []
    lines.append(f"# test 测试报告 ({ts})")
    lines.append("")
    lines.append("## 配置")
    lines.append(f"- URL: `{BASE_URL}{API_PATH}`")
    lines.append(f"- 模型: `{MODEL}`")
    lines.append(f"- 问题: `{QUESTION}`")
    lines.append(f"- 并发: `{CONCURRENCY}`  总请求: `{TOTAL}`")
    lines.append("")
    lines.append("## 汇总")
    lines.append(f"- 总墙钟用时: `{wall:.3f}s`")
    lines.append(f"- 成功 / 失败: `{summary['ok']} / {summary['fail']}` (共 `{summary['total']}`)")
    lines.append(f"- 单请求用时: min `{summary['min_sec']}s`  avg `{summary['avg_sec']}s`  max `{summary['max_sec']}s`")
    lines.append(f"- 分位: p50 `{summary['p50_sec']}s`  p90 `{summary['p90_sec']}s`  p95 `{summary['p95_sec']}s`  p99 `{summary['p99_sec']}s`")
    lines.append("")
    lines.append("## 明细")
    lines.append("")
    for r in results:
        tag = "OK" if r["ok"] else "ERR"
        lines.append(
            f"### #{r['idx']} [{tag}] status=`{r['status_code']}` 用时=`{r['elapsed_sec']}s` 起始=`{r['started_at']}`"
        )
        lines.append("")
        if r["usage"]:
            lines.append(f"- usage: `{json.dumps(r['usage'], ensure_ascii=False)}`")
        if r["error"]:
            err_full = r["error"]
            err_half = err_full[: max(1, len(err_full) // 2)]
            lines.append("- 错误:")
            lines.append("")
            lines.append("```")
            lines.append(err_half)
            lines.append("```")
        if r["content"]:
            content_full = r["content"]
            content_half = content_full[: max(1, len(content_full) // 2)]
            lines.append("- 回复:")
            lines.append("")
            lines.append("```")
            lines.append(content_half)
            lines.append("```")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


async def main():
    warmup = await run_warmup()
    if not warmup["ok"]:
        print("\n[!] 预热失败，跳过批量压测。请检查 URL/Key/路径。")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        md_path = DOC_DIR / f"test_concurrent_report_{ts}.md"
        md_path.write_text(
            f"# test 预热失败\n\n- URL: `{BASE_URL}{API_PATH}`\n- 状态: `{warmup['status_code']}`\n- 用时: `{warmup['elapsed_sec']}s`\n- 错误: `{warmup['error']}`\n",
            encoding="utf-8",
        )
        print(f"已写入: {md_path}")
        return

    results, wall = await run_load(TOTAL, CONCURRENCY, warmup=warmup)
    summary = summarize(results)
    md_path = write_report(warmup, results, wall, summary)
    print()
    print("=" * 60)
    print(f"成功 {summary['ok']}/{summary['total']}  失败 {summary['fail']}")
    print(f"avg {summary['avg_sec']}s  p95 {summary['p95_sec']}s  max {summary['max_sec']}s")
    print(f"报告: {md_path}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

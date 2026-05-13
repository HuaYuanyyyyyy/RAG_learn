# test 测试报告 (20260512_154238)

## 配置
- URL: `https://onetoken.one/v1/chat/completions`
- 模型: `claude-sonnet-4-6`
- 问题: `你是什么模型？能干啥`
- 并发: `5`  总请求: `10`

## 汇总
- 总墙钟用时: `60.433s`
- 成功 / 失败: `6 / 4` (共 `10`)
- 单请求用时: min `15.827s`  avg `30.362s`  max `60.405s`
- 分位: p50 `27.521s`  p90 `38.454s`  p95 `60.405s`  p99 `60.405s`

## 明细

### #1 [OK] status=`200` 用时=`15.827s` 起始=`2026-05-12T15:41:22`

- usage: `{"prompt_tokens": 116, "completion_tokens": 322, "total_tokens": 438, "usage_semantic": "openai", "usage_source": "anthropic", "prompt_tokens_details": {"cached_tokens": 0, "text_tokens": 0, "audio_tokens": 0, "image_tokens": 0}, "completion_tokens_details": {"text_tokens": 0, "audio_tokens": 0, "image_tokens": 0, "reasoning_tokens": 0}, "input_tokens": 116, "output_tokens": 0, "input_tokens_details": null, "claude_cache_creation_5_m_tokens": 0, "claude_cache_creation_1_h_tokens": 0}`
- 回复:

```
# 我是 Claude Sonnet 4.6

由 Anthropic 公司开发的 AI 助手。

## 我能做什么？

### 📝 写作与创作
- 写文章、报告、故事、诗歌
- 文案撰写、润色修改
- 翻译（中英文等多语言）

### 💻 编程开发
- 编写、调试、解释代码
- 支持 Python、JavaScript、Java、C++ 等主流语言
-
```

### #2 [ERR] status=`-1` 用时=`60.405s` 起始=`2026-05-12T15:41:38`

- 错误:

```
RemoteProtocolError: Server discon
```

### #3 [OK] status=`200` 用时=`27.546s` 起始=`2026-05-12T15:41:38`

- usage: `{"prompt_tokens": 116, "completion_tokens": 337, "total_tokens": 453, "usage_semantic": "openai", "usage_source": "anthropic", "prompt_tokens_details": {"cached_tokens": 0, "text_tokens": 0, "audio_tokens": 0, "image_tokens": 0}, "completion_tokens_details": {"text_tokens": 0, "audio_tokens": 0, "image_tokens": 0, "reasoning_tokens": 0}, "input_tokens": 116, "output_tokens": 0, "input_tokens_details": null, "claude_cache_creation_5_m_tokens": 0, "claude_cache_creation_1_h_tokens": 0}`
- 回复:

```
# 我是 Claude！

**具体信息：**
- 🤖 模型：**Claude Sonnet 4.6**（由 Anthropic 开发）
- 📅 知识截止日期：**2025年8月**

---

## 我能干啥？

### 📝 写作 & 文案
- 写文章、报告、邮件、故事
- 润色和修改文字

### 💻 编程 & 技术
- 写代码（Python、JS、Java 等各种语言）
```

### #4 [OK] status=`200` 用时=`27.521s` 起始=`2026-05-12T15:41:38`

- usage: `{"prompt_tokens": 116, "completion_tokens": 353, "total_tokens": 469, "usage_semantic": "openai", "usage_source": "anthropic", "prompt_tokens_details": {"cached_tokens": 0, "text_tokens": 0, "audio_tokens": 0, "image_tokens": 0}, "completion_tokens_details": {"text_tokens": 0, "audio_tokens": 0, "image_tokens": 0, "reasoning_tokens": 0}, "input_tokens": 116, "output_tokens": 0, "input_tokens_details": null, "claude_cache_creation_5_m_tokens": 0, "claude_cache_creation_1_h_tokens": 0}`
- 回复:

```
我是 **Claude**，由 Anthropic 开发的 AI 助手。

根据系统信息，我运行的模型是 **claude-sonnet-4-6**。

---

## 我能做什么？

### 📝 写作与文字
- 写文章、报告、邮件、故事、诗歌
- 润色、翻译、校对文本
- 总结长篇内容

### 💻 编程与技术
- 写代码（Python、JavaScript、Java 等主流语言）
- 调
```

### #5 [OK] status=`200` 用时=`21.95s` 起始=`2026-05-12T15:41:38`

- usage: `{"prompt_tokens": 116, "completion_tokens": 299, "total_tokens": 415, "usage_semantic": "openai", "usage_source": "anthropic", "prompt_tokens_details": {"cached_tokens": 0, "text_tokens": 0, "audio_tokens": 0, "image_tokens": 0}, "completion_tokens_details": {"text_tokens": 0, "audio_tokens": 0, "image_tokens": 0, "reasoning_tokens": 0}, "input_tokens": 116, "output_tokens": 0, "input_tokens_details": null, "claude_cache_creation_5_m_tokens": 0, "claude_cache_creation_1_h_tokens": 0}`
- 回复:

```
# 我是 Claude Sonnet 4.6

由 Anthropic 开发的 AI 助手。

## 我能做什么？

### 📝 写作与创作
- 写文章、报告、故事、诗歌
- 润色、翻译、总结文本

### 💻 编程
- 编写、调试、解释代码
- 支持 Python、JavaScript、Java 等多种语言

### 🧠 分析与推
```

### #6 [OK] status=`200` 用时=`23.282s` 起始=`2026-05-12T15:41:38`

- usage: `{"prompt_tokens": 116, "completion_tokens": 357, "total_tokens": 473, "usage_semantic": "openai", "usage_source": "anthropic", "prompt_tokens_details": {"cached_tokens": 0, "text_tokens": 0, "audio_tokens": 0, "image_tokens": 0}, "completion_tokens_details": {"text_tokens": 0, "audio_tokens": 0, "image_tokens": 0, "reasoning_tokens": 0}, "input_tokens": 116, "output_tokens": 0, "input_tokens_details": null, "claude_cache_creation_5_m_tokens": 0, "claude_cache_creation_1_h_tokens": 0}`
- 回复:

```
# 我是 Claude Sonnet 4.6

由 Anthropic 公司开发的 AI 助手。

---

## 我能做什么？

### 📝 写作与创作
- 写文章、报告、邮件、故事、诗歌
- 内容润色、翻译、摘要

### 💻 编程开发
- 编写、调试、解释代码（Python、Java、JS 等几乎所有语言）
- 技术方案设计

### 🧠 分析与推理
- 数据分析、逻辑推理
```

### #7 [ERR] status=`-1` 用时=`38.454s` 起始=`2026-05-12T15:42:00`

- 错误:

```
RemoteProtocolError: Server discon
```

### #8 [ERR] status=`-1` 用时=`37.121s` 起始=`2026-05-12T15:42:01`

- 错误:

```
RemoteProtocolError: Server discon
```

### #9 [ERR] status=`-1` 用时=`32.868s` 起始=`2026-05-12T15:42:05`

- 错误:

```
RemoteProtocolError: Server discon
```

### #10 [OK] status=`200` 用时=`18.65s` 起始=`2026-05-12T15:42:05`

- usage: `{"prompt_tokens": 116, "completion_tokens": 313, "total_tokens": 429, "usage_semantic": "openai", "usage_source": "anthropic", "prompt_tokens_details": {"cached_tokens": 0, "text_tokens": 0, "audio_tokens": 0, "image_tokens": 0}, "completion_tokens_details": {"text_tokens": 0, "audio_tokens": 0, "image_tokens": 0, "reasoning_tokens": 0}, "input_tokens": 116, "output_tokens": 0, "input_tokens_details": null, "claude_cache_creation_5_m_tokens": 0, "claude_cache_creation_1_h_tokens": 0}`
- 回复:

```
# 我是 Claude Sonnet 4.6

由 Anthropic 公司开发的 AI 助手。

---

## 我能帮你做什么？

### 📝 写作与文字
- 写文章、报告、邮件、故事
- 润色和修改文本
- 翻译多种语言

### 💻 编程与技术
- 编写和调试代码（Python、JavaScript、Java 等）
- 解释技术概念

```

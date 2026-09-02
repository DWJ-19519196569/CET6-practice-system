# CET-6 综合练习系统

面向大学英语六级（CET-6）备考的**本地 AI 综合练习系统**。四个平行功能 Tab，覆盖四类能力：

| Tab | 能力 | 交互形态 |
|-----|------|---------|
| 每日一篇 | 听力 | AI 生成短文/对话并朗读，织入六级词，逐句可翻译 |
| 互动故事 | 阅读 | AI 流式讲故事，A/B/C/D 选择推进剧情，逐句朗读 |
| 翻译练习 | 翻译（汉译英） | AI 出中文材料 → 用户译英文 → AI 参考译文 + 逐句点评 + 打分 |
| 写作练习 | 写作 | AI 出作文题（6 类题型）→ 用户写 → AI 打分 + 逐段批注 + 范文 |

---

## 目录结构

```
CET-6/
├─ 系统/                    主程序
│  ├─ server.py            后端：全部 HTTP API + 历史归档
│  ├─ engine.py            引擎：prompt 模板 / LLMClient / TTS / 解析器
│  ├─ wordlist.py          词表模块：词块 / 滚动窗口 / 可复现采样
│  ├─ config.toml          配置：端口 / 模型 / 题型长度 / 路径 / 超时重试
│  ├─ 前端.html            单文件前端（原生 JS + SSE，无框架）
│  ├─ 启动.bat             启动入口（Windows，GBK 编码）
│  ├─ 使用说明.txt         面向使用者的说明
│  ├─ deepseek_key.txt.example  DeepSeek key 模板（真实 key 不入库）
│  └─ test_*.py            5 套测试
└─ 单词/wordlist_cet6_乱序.txt   CET-6 词表（3991 词）
```

运行后会自动生成（已 `.gitignore` 排除）：`系统/state.json`、`系统/历史记录/`、`每日一篇/`、`互动故事/`。

---

## 环境要求

- Windows + Python 3.12
- 生成模型（二选一）：
  - 本地 Qwen（`http://127.0.0.1:8081/v1`，需另开 llama.cpp）
  - 云端 DeepSeek 官方 API（需在页面顶部填 key）
- TTS：Kokoro-82M（本地 CPU 推理）

## 安装依赖

```powershell
pip install fastapi uvicorn httpx pydantic numpy soundfile kokoro imageio-ffmpeg
```

## 运行

```powershell
cd 系统
python server.py
```

浏览器打开 <http://127.0.0.1:8123>。或双击 `系统\启动.bat`（会先检测本地 Qwen、再启动服务并打开浏览器）。

## 配置

`系统\config.toml` 关键项：

- `[llm] active`：`local`（本地 Qwen，默认）/ `cloud`（云端 DeepSeek）
- `[llm.cloud]`：DeepSeek 官方 API 地址与模型列表；key 不写死在配置里，运行时读 `deepseek_key.txt`
- `[llm.generation]`：`temperature` / `max_tokens` / `timeout`（默认 180s）/ `max_retries`（默认 1）
- `[paths]`：均为**相对路径**（以 `server.py` 所在目录为基准），克隆到任意目录可直接运行

### 填入 DeepSeek key

1. 复制 `系统\deepseek_key.txt.example` 为 `系统\deepseek_key.txt`
2. 填入你的 key（`sk-...`）
3. 或在页面顶部的输入框里填写后点「保存」（也会写入 `deepseek_key.txt`）

> `deepseek_key.txt` 已被 `.gitignore` 排除，不会被提交到仓库。

---

## 测试

```powershell
cd 系统
python test_feature_regressions.py   # 14 用例：核心回归 + 翻译写作去重/归档
python test_trw.py                    # 15 用例：prompt / 解析 / API（mock）
python test_history.py                # 2 用例：归档 / 恢复 / 删除
python test_stream_parser.py          # 7 用例：流式切句 / SEGMENT 边界
python test_wordlist.py               # 7 用例：词表采样
```

---

## 说明

- 内容全部由模型现场生成；除 DeepSeek API 外不联网。
- 词表来源：[KyleBing/english-vocabulary](https://github.com/KyleBing/english-vocabulary) 公开词表（仅取 CET-6 词，3991 词）。
- 本项目仅供学习交流使用，题库与评分提示词按 CET-6 常考题型设计，非官方评分。

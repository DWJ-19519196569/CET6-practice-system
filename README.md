# CET-6 综合练习系统

面向大学英语六级（CET-6）备考的**本地 AI 综合练习系统**。四个平行功能 Tab，覆盖四类能力：

| Tab | 能力 | 交互形态 |
|-----|------|---------|
| 每日一篇 | 听力 | AI 生成短文/对话并朗读，织入六级词，逐句可翻译 |
| 互动故事 | 阅读 | AI 流式讲故事（默认织入六级词），A/B/C/D 选择推进剧情，逐句朗读 |
| 翻译练习 | 翻译（汉译英） | AI 出中文材料 → 用户译英文 → AI 参考译文 + 逐句点评 + 打分 |
| 写作练习 | 写作 | AI 出作文题（6 类题型）→ 用户写 → AI 打分 + 逐段批注 + 范文 |
| 口语练习 | 口语 | AI 出朗读题 + Kokoro 范本 → 用户朗读录音 → OpenPronounce 音素级发音评分 |
| 笔记本 | 积累 | 生词（故事点词收录）/ 翻译句式 / 写作表达 三类，支持编辑/导出/复习 |

---

## 目录结构

```
CET-6/
├─ 系统/                    主程序
│  ├─ server.py            后端：全部 HTTP API + 历史归档
│  ├─ engine.py            引擎：prompt 模板 / LLMClient / TTS / 解析器
│  ├─ wordlist.py          词表模块：词块 / 滚动窗口 / 可复现采样
│  ├─ pronounce.py         口语发音评测：评分器抽象 + OpenPronounce + 音频预处理
│  ├─ config.toml          配置：端口 / 模型 / 题型长度 / 路径 / 超时重试
│  ├─ 前端.html            单文件前端（原生 JS + SSE，无框架）
│  ├─ 启动.bat             启动入口（Windows，GBK 编码）
│  ├─ 使用说明.txt         面向使用者的说明
│  ├─ deepseek_key.txt.example  DeepSeek key 模板（真实 key 不入库）
│  └─ test_*.py            6 套测试
└─ 单词/wordlist_cet6_乱序.txt   CET-6 词表（3991 词）
```

运行后会自动生成（已 `.gitignore` 排除）：`系统/state.json`、`系统/历史记录/`、`每日一篇/`、`互动故事/`。

---

## 环境要求

- Windows + Python 3.12
- 生成模型：云端 DeepSeek 官方 API（需在页面顶部填 key；本地 Qwen 已停用，显存留给 TTS/评分模型）
- TTS：Kokoro-82M（CPU/GPU 自动检测，装 CUDA 版 PyTorch 即自动上 GPU）
- 口语评测：OpenPronounce（wav2vec2，CPU/GPU 自动检测）；需系统有 `espeak-ng`（本机已置于 `%LOCALAPPDATA%\eSpeakNG`），首次使用会从 Hugging Face 下载两个 wav2vec2 模型（约 1.2 GB ×2，已走 hf-mirror 镜像）
- 服务启动后会自动在后台把 Kokoro 与 wav2vec2 模型预热到 GPU（约 1 分钟），预热完成前生成/评分会排队等待

## 安装依赖

```powershell
pip install fastapi uvicorn httpx pydantic numpy soundfile kokoro imageio-ffmpeg
pip install openpronounce   # 口语发音评测（自动带入 librosa/scipy/scikit-learn/fastdtw/phonemizer/Levenshtein 等）
```

> GPU 加速（可选）：有 NVIDIA 显卡时，把 PyTorch 换成 CUDA 版即可自动启用，无需改任何代码。
> 国内推荐从阿里云直链下载 wheel 后本地安装：
> `pip install <torch-2.10.0+cu128-cp312-cp312-win_amd64.whl>`（下载页 <https://mirrors.aliyun.com/pytorch-wheels/cu128/>）

## 运行

```powershell
cd 系统
python server.py
```

浏览器打开 <http://127.0.0.1:8123>。或双击 `系统\启动.bat`（启动服务并打开浏览器，模型自动预热到 GPU）。

## 配置

`系统\config.toml` 关键项：

- `[llm] active`：`cloud`（云端 DeepSeek，默认）
- `[llm.cloud]`：DeepSeek 官方 API 地址与模型列表；key 不写死在配置里，运行时读 `deepseek_key.txt`
- `[llm.generation]`：`temperature` / `max_tokens` / `timeout`（默认 180s）/ `max_retries`（默认 1）
- `[speaking]`：`passage_words`（朗读题词数）/ `sample_n`（织入词数）/ `max_record_sec`（最长录音秒数）/ `scorer`（评测引擎，默认 `openpronounce`）
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
python test_speaking.py               # 口语练习：出题 / 音频预处理 / 评分归一化 / API / 历史
```

---

## 说明

- 内容全部由模型现场生成；除 DeepSeek API 外不联网。
- 口语评测参考声用本机 Kokoro（离线）；wav2vec2 模型首次使用时从 Hugging Face 下载一次（走 hf-mirror），之后完全离线。
- 词表来源：[KyleBing/english-vocabulary](https://github.com/KyleBing/english-vocabulary) 公开词表（仅取 CET-6 词，3991 词）。
- 本项目仅供学习交流使用，题库与评分提示词按 CET-6 常考题型设计，非官方评分。

# 在 Google Colab 上进行语音到语音翻译（Speech-to-Speech）

本文档演示如何在 **Google Colab** 环境中使用 pyVideoTrans 完成完整的视频语音翻译流程，并通过 **index-tts2** 模型保留原音色。同时补充了部署指南，帮助你将自建的 TTS 服务通过 Gradio 接入管线。

## 一、准备工作

1. 打开 [Google Colab](https://colab.research.google.com/)。
2. 建议使用 GPU 运行时（`启用 GPU -> 运行时 -> 更改运行时类型 -> 硬件加速器 -> GPU`）。
3. 准备好视频/音频输入文件，可以上传到 Colab 或 Google Drive。
4. 确保您已部署并启动 `index-tts2` WebUI，并能够通过公网地址访问（可使用 ngrok/cloudflared 等隧道）。

## 二、快速开始 Notebook

仓库提供了两份现成的 Notebook（点击徽章可直接在 Colab 中打开）：

- `notebooks/colab_speech_to_speech.ipynb`：英文讲解版。[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jianchang512/pyvideotrans/blob/main/notebooks/colab_speech_to_speech.ipynb)
- `notebooks/colab_speech_to_speech_zh.ipynb`：全新中文讲解版，便于培训或与团队分享。[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jianchang512/pyvideotrans/blob/main/notebooks/colab_speech_to_speech_zh.ipynb)

在 Colab 中按照以下步骤运行：

1. 克隆仓库并进入目录：
   ```python
   !git clone https://github.com/jianchang512/pyvideotrans.git
   %cd pyvideotrans
   ```
2. 在左侧“文件”树中双击打开需要的 Notebook（英文或中文版本）。
3. 按顺序执行 Notebook 各单元：
   - 克隆/更新仓库并切换目录；
   - 安装 `requirements-colab.txt` 依赖（包含 CUDA 12.1 下的 Faster-Whisper）；
   - （可选）挂载 Google Drive 以便读写文件；
   - 准备示例视频（或自行上传文件）；
   - 调用 `run_speech_to_speech` 函数执行完整的语音到语音翻译；
4. Notebook 中 `result` 字典会返回生成的视频、音频和字幕路径，可直接下载或保存至 Drive。

## 三、`colab_s2s.py` 快速接口说明

`colab_s2s.py` 封装了在 Colab 环境下的调用流程，核心函数为：

```python
from colab_s2s import run_speech_to_speech

result = run_speech_to_speech(
    input_path='/content/media/sample.mp4',
    target_language='en',
    index_tts_url='http://<your-index-tts2-endpoint>',
    whisper_model='large-v3',
    translate_backend='google',
    recognition_backend='faster-whisper',
    separate_vocals=True,
    voice_role='clone'
)
```

常用参数说明：

| 参数 | 说明 |
| --- | --- |
| `input_path` | 输入视频/音频路径（Colab 本地路径或挂载盘路径）。 |
| `target_language` | 输出语言代码，例如 `en`、`ja`、`zh-cn` 等。 |
| `index_tts_url` | index-tts2 WebUI 的可访问地址（确保 Colab 可连通）。 |
| `whisper_model` | Faster-Whisper 模型名称，默认 `large-v2`。 |
| `translate_backend` | 翻译通道，可选 `google`、`microsoft`、`deepl`、`baidu`、`tencent`、`ali` 等。 |
| `recognition_backend` | 语音识别通道，可选 `faster-whisper`、`openai-whisper`、`funasr`、`google`、`gemini` 等。 |
| `separate_vocals` | 是否启用人声/背景音乐分离，默认 `True`。 |
| `voice_role` | 配音角色，使用 `clone` 可在 index-tts2 中保持原音色。 |
| `voice_autorate` / `video_autorate` | 是否自动根据字幕对齐音频/视频节奏，默认 `True`。 |
| `use_cuda` | 是否在 GPU 上运行 Faster-Whisper，默认 `True`。 |

函数返回的 `result` 字典包含：

- `uuid`：本次任务的唯一编号；
- `elapsed`：执行时长（秒）；
- `target_dir`：输出目录；
- `translated_video`：合成后的视频文件路径；
- `translated_audio`：目标语言音频路径；
- `target_subtitle`：目标语言字幕路径；
- `source_subtitle`：原字幕路径；
- `logs`（可选）：过程日志（如果 `collect_logs=True`）。

## 四、自动音色克隆与参考片段长度

- 当 `voice_role='clone'` 且 `tts_type` 为 Index-TTS2 时，pyVideoTrans 会自动从原始视频/音频中截取参考片段，并在调用接口前压缩到默认 **≤ 8 秒**，避免因上传超长音频导致的失败。
- 参考时长可在 `cfg.ini` 或环境变量中通过 `index_tts2_ref_seconds` 调整（允许 1–30 秒）。
- 如果启用了声道分离（`separate_vocals=True`），系统会优先使用分离后的人声轨道作为参考，使 timbre 更纯净。

## 五、接入 index-tts2 或自建 TTS 服务

### 1. 部署并公开 Index-TTS2

1. 使用官方 WebUI 或 Docker 启动 Index-TTS2。
2. 通过 `ngrok`、`cloudflared`、自建反向代理等方式，将接口暴露为 Colab 可访问的公网地址。
3. 在 Colab/本地调用 `run_speech_to_speech` 时，将该地址传入 `index_tts_url`，即可完成音色克隆。

### 2. 使用 `gradio_client` 快速自检

部署完成后，建议先在本地或 Colab 中运行以下示例，确认接口连通性与必需参数：

```python
from gradio_client import Client, handle_file

client = Client("https://<your-indextts2-host>")
# 以 Voice Reference + Emotion Reference 的接口为例
demo = client.predict(
    emo_control_method="Same as the voice reference",
    prompt=handle_file('/content/sample_ref.wav'),
    text="Hello!!",
    emo_ref_path=handle_file('/content/sample_ref.wav'),
    api_name="/gen_single"
)
print(demo)
```

如需调用其他辅助接口，可参考：

- `/on_method_select`：切换情感控制方式；
- `/on_input_text_change`：检查输入文本分段情况；
- `/on_experimental_change`：开启实验性参数；
- `/update_prompt_audio`：刷新默认参考音频；
- `/gen_single`：执行实际合成（pyVideoTrans 默认调用此接口）。

### 3. 接入自建或改造后的 TTS 服务

如果你在 Index-TTS2 的基础上扩展了自定义接口，只需公开一个包含「文本输入 + 至少一个参考音频」的 Gradio Endpoint。pyVideoTrans 会自动读取 API Schema 并映射必要字段。常见场景：

1. 部署私有化版本并开启基础验证/Token —— 在 `index_tts_url` 中附加凭证即可；
2. 为不同语言准备独立模型 —— 可在 Notebook 中动态设置 `index_tts_url`，或在调用前修改 `config.params['f5tts_url']`；
3. 若 API 需要额外的情感向量或说话人 ID，可在自建接口中提供默认值，pyVideoTrans 会把这些参数视为可选项自动填充。

### 4. 常见故障排查

- **接口结构变化**：pyVideoTrans 会在每次调用时刷新 Gradio Schema，若报缺参，请刷新 WebUI 或检查终端日志；
- **上传文件失败**：确认 Endpoint 运行在 HTTPS，或在配置中禁用 SSL 校验；
- **声音不稳定**：尝试缩短参考片段，或预处理 3–5 秒干净语音。

## 六、使用 index-tts2 的注意事项

1. WebUI 必须开放 HTTP 接口，并能被 Colab 网络访问。建议使用安全隧道工具（例如 cloudflared、ngrok）。
2. 参考音频应保持在 10 秒以内、清晰无噪声；在 `voice_role='clone'` 时软件会自动截取原音频片段作为参考。
3. 若 index-tts2 的接口字段发生变化，`colab_s2s.py` 会自动读取 Gradio API 定义并适配，若仍报缺少参数，可刷新 WebUI 或检查日志。

## 七、常见问题

- **依赖安装较慢**：Colab 首次下载模型与依赖时耗时较久，可挂载 Drive 缓存模型目录。
- **index-tts2 连接失败**：确认 URL 是否可公开访问，若在本地运行请使用隧道工具映射到公网。
- **显存不足**：尝试更换较小的 Whisper 模型（如 `medium`/`small`）或关闭 `separate_vocals`。
- **日志调试**：`result['logs']` 中包含每个阶段的状态信息，便于排查。

## 八、目录结构调整

新增的主要文件：

- `colab_s2s.py`：Colab 专用工具函数集合；
- `notebooks/colab_speech_to_speech.ipynb`：英文示例；
- `notebooks/colab_speech_to_speech_zh.ipynb`：中文引导版本；
- 文档 `docs/google_colab_s2s.md`（即本文）提供操作说明。

通过上述步骤，即可在 Colab 环境完成从语音识别、翻译、到 index-tts2 配音的完整流程，并生成带字幕的视频输出。

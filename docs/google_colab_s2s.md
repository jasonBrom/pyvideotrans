# 在 Google Colab 上进行语音到语音翻译（Speech-to-Speech）

本文档演示如何在 **Google Colab** 环境中使用 pyVideoTrans 完成完整的视频语音翻译流程，并通过 **index-tts2** 模型保留原音色。

## 一、准备工作

1. 打开 [Google Colab](https://colab.research.google.com/)。
2. 建议使用 GPU 运行时（`启用 GPU -> 运行时 -> 更改运行时类型 -> 硬件加速器 -> GPU`）。
3. 准备好视频/音频输入文件，可以上传到 Colab 或 Google Drive。
4. 确保您已部署并启动 `index-tts2` WebUI，并能够通过公网地址访问（可使用 ngrok/cloudflared 等隧道）。

## 二、快速开始 Notebook

仓库提供了现成的 Notebook：`notebooks/colab_speech_to_speech.ipynb`。

1. 在 Colab 中运行以下命令克隆仓库：
   ```python
   !git clone https://github.com/jianchang512/pyvideotrans.git
   %cd pyvideotrans
   ```
2. 打开 `notebooks/colab_speech_to_speech.ipynb` Notebook（Colab 左侧“文件”树中双击即可）。
3. 按顺序执行 Notebook 各单元：
   - 克隆/更新仓库并切换目录；
   - 安装 `requirements-colab.txt` 依赖（包含 CUDA 12.1 下的 Faster-Whisper）；
   -（可选）挂载 Google Drive 以便读写文件；
   - 准备示例视频（或者自行上传文件）；
   - 调用 `run_speech_to_speech` 函数执行完整的语音到语音翻译；
4. Notebook 中 `result` 字典会返回生成的视频、音频和字幕路径，可直接下载或保存至 Drive。

## 三、`colab_s2s.py` 快速接口说明

新增加的 `colab_s2s.py` 封装了在 Colab 环境下的调用流程，核心函数为：

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

## 四、使用 index-tts2 的注意事项

1. WebUI 必须开放 HTTP 接口，并能被 Colab 网络访问。建议使用安全隧道工具（例如 cloudflared、ngrok）。
2. 参考音频应保持在 10 秒以内、清晰无噪声；在 `voice_role='clone'` 时软件会自动截取原音频片段作为参考。
3. 若 index-tts2 的接口字段发生变化，`colab_s2s.py` 会自动读取 Gradio API 定义并适配，若仍报缺少参数，可刷新 WebUI 或检查日志。

## 五、常见问题

- **依赖安装较慢**：Colab 首次下载模型与依赖时耗时较久，可挂载 Drive 缓存模型目录。
- **index-tts2 连接失败**：确认 URL 是否可公开访问，若在本地运行请使用隧道工具映射到公网。
- **显存不足**：尝试更换较小的 Whisper 模型（如 `medium`/`small`）或关闭 `separate_vocals`。
- **日志调试**：`result['logs']` 中包含每个阶段的状态信息，便于排查。

## 六、目录结构调整

新增的主要文件：

- `colab_s2s.py`：Colab 专用工具函数集合；
- `notebooks/colab_speech_to_speech.ipynb`：一步步运行示例；
- 文档 `docs/google_colab_s2s.md`（即本文）提供操作说明。

通过上述步骤，即可在 Colab 环境完成从语音识别、翻译、到 index-tts2 配音的完整流程，并生成带字幕的视频输出。

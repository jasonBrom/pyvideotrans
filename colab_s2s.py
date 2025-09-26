"""Utilities for running speech-to-speech translation inside Google Colab.

This helper keeps the orchestration logic in one place so that notebooks can
focus on user interaction.  It wires together recognition, translation and
index-tts2 cloning in a sequential pipeline that mirrors the desktop
application.
"""
from __future__ import annotations

import contextlib
import time
from pathlib import Path
from queue import Empty
from typing import Dict, List, Optional, Tuple, Union

from videotrans.configure import config
from videotrans.task.trans_create import TransCreate
from videotrans.util import tools
from videotrans import recognition, translator, tts


_TRANSLATE_NAME_TO_INDEX: Dict[str, int] = {
    'google': translator.GOOGLE_INDEX,
    'microsoft': translator.MICROSOFT_INDEX,
    'deepl': translator.DEEPL_INDEX,
    'deeplx': translator.DEEPLX_INDEX,
    'baidu': translator.BAIDU_INDEX,
    'tencent': translator.TENCENT_INDEX,
    'ali': translator.ALI_INDEX,
}

_RECOGN_NAME_TO_INDEX: Dict[str, int] = {
    'faster-whisper': recognition.FASTER_WHISPER,
    'openai-whisper': recognition.OPENAI_WHISPER,
    'funasr': recognition.FUNASR_CN,
    'google': recognition.GOOGLE_SPEECH,
    'gemini': recognition.GEMINI_SPEECH,
}


def _language_label(code: str) -> Tuple[str, str]:
    """Return language label and normalized code for pyvideotrans configs."""
    if not code:
        raise ValueError("language code must not be empty")
    normalized = code.lower().replace('_', '-')
    label = config.langlist.get(normalized)
    if not label and normalized in config.rev_langlist:
        label = config.langlist[config.rev_langlist[normalized]]
    if not label:
        label = config.langlist.get(normalized.split('-')[0])
    if not label:
        raise ValueError(f"Unsupported language code: {code}")
    return label, normalized


def _resolve_translate_backend(value: Union[str, int]) -> int:
    if isinstance(value, int):
        return value
    key = value.lower()
    if key not in _TRANSLATE_NAME_TO_INDEX:
        raise ValueError(
            f"Unknown translate backend '{value}'. Allowed keys: {sorted(_TRANSLATE_NAME_TO_INDEX)}"
        )
    return _TRANSLATE_NAME_TO_INDEX[key]


def _resolve_recogn_backend(value: Union[str, int]) -> int:
    if isinstance(value, int):
        return value
    key = value.lower()
    if key not in _RECOGN_NAME_TO_INDEX:
        raise ValueError(
            f"Unknown recognition backend '{value}'. Allowed keys: {sorted(_RECOGN_NAME_TO_INDEX)}"
        )
    return _RECOGN_NAME_TO_INDEX[key]


def _consume_logs(uuid: str) -> List[Dict[str, str]]:
    queue = config.uuid_logs_queue.get(uuid)
    if queue is None:
        return []
    messages: List[Dict[str, str]] = []
    with contextlib.suppress(Exception):
        while True:
            try:
                item = queue.get_nowait()
            except Empty:
                break
            else:
                if isinstance(item, dict):
                    messages.append(item)
    return messages


def _prepare_environment(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    Path(config.TEMP_DIR).mkdir(parents=True, exist_ok=True)
    config.exec_mode = 'colab'
    config.exit_soft = False
    config.current_status = 'ing'
    config.box_recogn = 'ing'
    config.box_trans = 'ing'
    config.box_tts = 'ing'
    config.global_msg.clear()


def _restore_environment(previous_status: Dict[str, str]) -> None:
    config.current_status = previous_status.get('current_status', 'stop')
    config.box_recogn = previous_status.get('box_recogn', 'stop')
    config.box_trans = previous_status.get('box_trans', 'stop')
    config.box_tts = previous_status.get('box_tts', 'stop')


def run_speech_to_speech(
    *,
    input_path: Union[str, Path],
    target_language: str,
    index_tts_url: str,
    whisper_model: str = 'large-v2',
    translate_backend: Union[str, int] = 'google',
    recognition_backend: Union[str, int] = 'faster-whisper',
    output_root: Union[str, Path] = '/content/pyvideotrans-output',
    separate_vocals: bool = True,
    denoise: bool = False,
    split_type: str = 'all',
    voice_role: str = 'clone',
    voice_rate: str = '+0%',
    volume: str = '+0%',
    pitch: str = '+0Hz',
    voice_autorate: bool = True,
    video_autorate: bool = True,
    use_cuda: bool = True,
    collect_logs: bool = True,
) -> Dict[str, Union[str, Dict[str, str], List[Dict[str, str]]]]:
    """Execute the full speech-to-speech translation pipeline.

    Parameters
    ----------
    input_path:
        Source video or audio file on the Colab filesystem.
    target_language:
        ISO language code such as ``en`` or ``ja``.
    index_tts_url:
        Base URL of the index-tts2 webui (e.g. ``http://127.0.0.1:7860``).
    whisper_model:
        Name of the Faster-Whisper model to use.
    translate_backend:
        Either an integer index or one of ``google``, ``microsoft``, ``deepl``
        ``deeplx``, ``baidu``, ``tencent`` or ``ali``.
    recognition_backend:
        Either an integer index or one of ``faster-whisper`` (default),
        ``openai-whisper``, ``funasr``, ``google`` or ``gemini``.
    output_root:
        Directory where result folders will be created.
    separate_vocals:
        If True, perform vocal/music separation before cloning.
    denoise:
        Apply noise removal to the recognition audio.
    split_type:
        ``all`` for full utterances or ``avg`` for even splitting.
    voice_role:
        TTS role. Use ``clone`` to keep the original timbre.
    voice_rate / volume / pitch:
        TTS prosody adjustments following pyvideotrans formatting.
    voice_autorate / video_autorate:
        Enable automatic alignment to better fit subtitles.
    use_cuda:
        Whether to run Faster-Whisper on CUDA when available.
    collect_logs:
        When True, attach the streaming logs to the return payload.
    """

    input_path = Path(input_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not index_tts_url:
        raise ValueError('index_tts_url must be provided')

    target_label, target_code = _language_label(target_language)
    _, source_code = ('Auto', 'auto')

    translate_index = _resolve_translate_backend(translate_backend)
    recogn_index = _resolve_recogn_backend(recognition_backend)

    output_root = Path(output_root).expanduser().resolve()
    prev_status = {
        'current_status': config.current_status,
        'box_recogn': config.box_recogn,
        'box_trans': config.box_trans,
        'box_tts': config.box_tts,
    }
    prev_params = {
        'f5tts_url': config.params.get('f5tts_url'),
        'f5tts_ttstype': config.params.get('f5tts_ttstype'),
        'f5tts_is_whisper': config.params.get('f5tts_is_whisper'),
    }

    start_time = time.time()
    _prepare_environment(output_root)

    config.params['f5tts_url'] = index_tts_url.strip().rstrip('/')
    config.params['f5tts_ttstype'] = 'Index-TTS2'
    config.params['f5tts_is_whisper'] = False

    obj = tools.format_video(input_path.as_posix(), target_dir=output_root.as_posix())

    cfg = {
        'target_dir': obj['target_dir'],
        'app_mode': 'biaozhun',
        'recogn_type': recogn_index,
        'model_name': whisper_model,
        'split_type': split_type,
        'remove_noise': denoise,
        'is_separate': separate_vocals,
        'translate_type': translate_index,
        'target_language': target_label,
        'target_language_code': target_code,
        'source_language': source_code,
        'source_language_code': source_code,
        'voice_role': voice_role,
        'voice_rate': voice_rate,
        'volume': volume,
        'pitch': pitch,
        'voice_autorate': voice_autorate,
        'video_autorate': video_autorate,
        'tts_type': tts.F5_TTS,
        'subtitle_type': 0,
        'cuda': bool(use_cuda),
    }
    cfg.update(obj)

    task = TransCreate(cfg=cfg)
    logs: List[Dict[str, str]] = []

    try:
        task.prepare()
        if collect_logs:
            logs.extend(_consume_logs(task.uuid))

        task.recogn()
        if collect_logs:
            logs.extend(_consume_logs(task.uuid))

        if task.shoud_trans:
            task.trans()
            if collect_logs:
                logs.extend(_consume_logs(task.uuid))

        if task.shoud_dubbing:
            task.dubbing()
            if collect_logs:
                logs.extend(_consume_logs(task.uuid))

        task.align()
        if collect_logs:
            logs.extend(_consume_logs(task.uuid))

        task.assembling()
        if collect_logs:
            logs.extend(_consume_logs(task.uuid))

        task.task_done()
        if collect_logs:
            logs.extend(_consume_logs(task.uuid))
    finally:
        config.params['f5tts_url'] = prev_params.get('f5tts_url', '')
        config.params['f5tts_ttstype'] = prev_params.get('f5tts_ttstype', 'F5-TTS')
        config.params['f5tts_is_whisper'] = prev_params.get('f5tts_is_whisper', False)
        _restore_environment(prev_status)

    target_dir = Path(task.cfg['target_dir']).resolve()
    result = {
        'uuid': task.uuid,
        'elapsed': time.time() - start_time,
        'target_dir': target_dir.as_posix(),
        'translated_video': Path(task.cfg['targetdir_mp4']).resolve().as_posix(),
        'translated_audio': Path(task.cfg['target_wav_output']).resolve().as_posix() if task.cfg.get('target_wav_output') else '',
        'target_subtitle': Path(task.cfg['target_sub']).resolve().as_posix() if task.cfg.get('target_sub') else '',
        'source_subtitle': Path(task.cfg['source_sub']).resolve().as_posix() if task.cfg.get('source_sub') else '',
    }
    if collect_logs:
        result['logs'] = logs
    return result


__all__ = [
    'run_speech_to_speech',
]

import copy
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple

from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_not_exception_type, before_log, after_log, \
    RetryError

from videotrans.configure import config
from videotrans.configure._except import NO_RETRY_EXCEPT,StopRetry
from videotrans.tts._base import BaseTTS
from videotrans.util import tools
from gradio_client import Client, handle_file

RETRY_NUMS = 2
RETRY_DELAY = 5


@dataclass
class F5TTS(BaseTTS):
    v1_local: bool = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self._index2_endpoint_cache: Optional[Tuple[Optional[str], Optional[int], List[Dict[str, Union[str, Dict]]]]] = None
        self._index2_ref_short_cache: Dict[str, str] = {}
        self.copydata = copy.deepcopy(self.queue_tts)
        api_url = config.params['f5tts_url'].strip().rstrip('/').lower()
        self.api_url = f'http://{api_url}' if not api_url.startswith('http') else api_url
        self.v1_local = True
        sepflag = self.api_url.find('/', 9)
        if sepflag > -1:
            self.api_url = self.api_url[:sepflag]

        if not re.search(r'127.0.0.1|localhost', self.api_url):
            self.v1_local = False
        elif re.search(r'^https:', self.api_url):
            self._set_proxy(type='set')

    def _exec(self):
        self._local_mul_thread()

    def _item_task_v1(self, data_item: Union[Dict, List, None]):

        speed = 1.0
        try:
            speed = 1 + float(self.rate.replace('%', '')) / 100
        except:
            pass

        text = data_item['text'].strip()
        role = data_item['role']
        data = {'ref_text': '', 'ref_wav': ''}

        if role == 'clone':
            data['ref_wav'] = data_item['ref_wav']
            if not config.params.get('f5tts_is_whisper'):
                data['ref_text'] = data_item.get('ref_text').strip()
        else:
            roledict = tools.get_f5tts_role()
            if role in roledict:
                data['ref_text'] = roledict[role]['ref_text'] if not config.params.get('f5tts_is_whisper') else ''
                data['ref_wav'] = config.ROOT_DIR + f"/f5-tts/{role}"

        if not Path(data['ref_wav']).exists():
            raise StopRetry(f'{role} 角色不存在')
        if data['ref_text'] and len(data['ref_text']) < 10:
            speed = 0.5
        try:
            client = Client(self.api_url, httpx_kwargs={"timeout": 7200}, ssl_verify=False)
        except Exception as e:
            raise StopRetry( f'{e}')
        try:
            result = client.predict(
                ref_audio_input=handle_file(data['ref_wav']),
                ref_text_input=data['ref_text'],
                gen_text_input=text,
                remove_silence=True,

                speed_slider=speed,
                api_name='/basic_tts'
            )
        except Exception as e:
            raise

        config.logger.info(f'result={result}')
        wav_file = result[0] if isinstance(result, (list, tuple)) and result else result
        if isinstance(wav_file, dict) and "value" in wav_file:
            wav_file = wav_file['value']
        if isinstance(wav_file, str) and Path(wav_file).is_file():
            self.convert_to_wav(wav_file, data_item['filename'])
        else:
            raise RuntimeError(str(result))

        if self.inst and self.inst.precent < 80:
            self.inst.precent += 0.1
        self.has_done += 1
        self._signal(text=f'{config.transobj["kaishipeiyin"]} {self.has_done}/{self.len}')

    def _item_task_spark(self, data_item: Union[Dict, List, None]):

        speed = 1.0
        try:
            speed = 1 + float(self.rate.replace('%', '')) / 100
        except:
            pass

        text = data_item['text'].strip()
        role = data_item['role']
        data = {'ref_text': '', 'ref_wav': ''}

        if role == 'clone':
            data['ref_wav'] = data_item['ref_wav']
            data['ref_text'] = data_item.get('ref_text', '')
        else:
            roledict = tools.get_f5tts_role()
            if role in roledict:
                data['ref_wav'] = config.ROOT_DIR + f"/f5-tts/{role}"

        if not Path(data['ref_wav']).exists():
            raise StopRetry( f'{role} 角色不存在')
        try:
            client = Client(self.api_url, httpx_kwargs={"timeout": 7200}, ssl_verify=False)
        except Exception as e:
            raise StopRetry( f'{e}')
        try:
            result = client.predict(
                text=text,
                prompt_text=data['ref_text'],
                prompt_wav_upload=handle_file(data['ref_wav']),
                prompt_wav_record=None,
                api_name='/voice_clone'
            )
        except Exception as e:
            raise

        config.logger.info(f'result={result}')
        wav_file = result[0] if isinstance(result, (list, tuple)) and result else result
        if isinstance(wav_file, dict) and "value" in wav_file:
            wav_file = wav_file['value']
        if isinstance(wav_file, str) and Path(wav_file).is_file():
            self.convert_to_wav(wav_file, data_item['filename'])
        else:
            raise RuntimeError(str(result))

        if self.inst and self.inst.precent < 80:
            self.inst.precent += 0.1
        self.has_done += 1
        self._signal(text=f'{config.transobj["kaishipeiyin"]} {self.has_done}/{self.len}')

    def _item_task_index(self, data_item: Union[Dict, List, None]):

        speed = 1.0
        try:
            speed = 1 + float(self.rate.replace('%', '')) / 100
        except:
            pass

        text = data_item['text'].strip()
        role = data_item['role']
        data = {'ref_wav': ''}

        if role == 'clone':
            data['ref_wav'] = data_item['ref_wav']
        else:
            roledict = tools.get_f5tts_role()
            if role in roledict:
                data['ref_wav'] = config.ROOT_DIR + f"/f5-tts/{role}"

        if not Path(data['ref_wav']).exists():
            raise StopRetry(  f'{role} 角色不存在')
        config.logger.info(f'index-tts {data=}')
        try:
            client = Client(self.api_url, httpx_kwargs={"timeout": 7200}, ssl_verify=False)
        except Exception as e:
            raise StopRetry(str(e))
        
        try:
            result = client.predict(
                prompt=handle_file(data['ref_wav']),
                emo_ref_path=handle_file(data['ref_wav']),
                text=text,
                api_name='/gen_single'
            )
        except Exception as e:
            if "Parameter emo_ref_path is not a valid" not in str(e):
                raise
            try:
                result = client.predict(
                        prompt=handle_file(data['ref_wav']),
                        text=text,
                        api_name='/gen_single'
                )
            except Exception as e:
                raise
                
        config.logger.info(f'result={result}')
        wav_file = result[0] if isinstance(result, (list, tuple)) and result else result
        if isinstance(wav_file, dict) and "value" in wav_file:
            wav_file = wav_file['value']
        if isinstance(wav_file, str) and Path(wav_file).is_file():
            self.convert_to_wav(wav_file, data_item['filename'])
        else:
            raise RuntimeError(str(result))
        if self.inst and self.inst.precent < 80:
            self.inst.precent += 0.1

        self.has_done += 1
        self._signal(text=f'{config.transobj["kaishipeiyin"]} {self.has_done}/{self.len}')

    def _classify_index2_param(self, param: Dict) -> str:
        label = (param.get('label') or '').lower()
        component = (param.get('component') or '').lower()
        python_type = (param.get('python_type') or '').lower()
        parameter_name = (param.get('parameter_name') or '').lower()

        text_keywords = {'text', 'script', 'content', 'sentence'}
        audio_keywords = {'audio', 'voice', 'wav', 'reference', 'prompt'}
        language_keywords = {'lang', 'language', 'locale'}

        if any(key in component for key in ('audio', 'file', 'upload')):
            return 'audio'
        if any(key in label for key in ('emo', 'emotion')):
            return 'emotion'
        if any(key in label for key in language_keywords) or any(key in parameter_name for key in language_keywords):
            return 'language'
        if any(key in label for key in text_keywords) or any(key in parameter_name for key in text_keywords) \
                or any(key in python_type for key in ('text', 'str')):
            if 'ref' in label or 'ref' in parameter_name:
                return 'ref_text'
            return 'text'
        if any(key in label for key in audio_keywords) or any(key in parameter_name for key in audio_keywords) \
                or 'audio' in python_type:
            return 'audio'
        if 'speaker' in label or 'speaker' in parameter_name or 'spk' in label:
            return 'speaker'
        if 'silence' in label and 'remove' in label:
            return 'remove_silence'
        if 'seed' in label or 'seed' in parameter_name:
            return 'seed'
        if 'speed' in label or 'speed' in parameter_name:
            return 'speed'
        if 'temperature' in label or 'top_p' in label or 'cfg' in label:
            return 'tuning'
        return ''

    def _resolve_index2_endpoint(self, client: 'Client') -> Tuple[Optional[str], Optional[int], List[Dict]]:
        if self._index2_endpoint_cache:
            return self._index2_endpoint_cache

        try:
            api_info = client.view_api(return_format='dict')
        except Exception as e:
            raise StopRetry(
                f"获取 Index-TTS2 接口结构失败: {e}" if config.defaulelang == 'zh' else f"Failed to fetch Index-TTS2 API schema: {e}"
            )

        endpoints: List[Tuple[Optional[str], Optional[int], Dict]] = []
        for api_name, info in api_info.get('named_endpoints', {}).items():
            name = api_name if isinstance(api_name, str) else str(api_name)
            if not name.startswith('/'):
                name = '/' + name
            endpoints.append((name, None, info))
        for fn_index, info in api_info.get('unnamed_endpoints', {}).items():
            try:
                idx = int(fn_index)
            except Exception:
                idx = None
            endpoints.append((None, idx, info))

        errors = []
        for api_name, fn_index, info in endpoints:
            parameters: List[Dict] = info.get('parameters') or []
            classified: List[Dict] = []
            text_count = 0
            audio_count = 0
            missing_required: List[str] = []
            for param in parameters:
                kind = self._classify_index2_param(param)
                if not kind and not param.get('parameter_has_default', False) \
                        and not param.get('optional', False) and not param.get('parameter_optional', False):
                    missing_required.append(param.get('label') or param.get('parameter_name') or str(param))
                if kind == 'text':
                    text_count += 1
                if kind in ('audio', 'emotion'):
                    audio_count += 1
                param = dict(param)
                param['__kind__'] = kind
                classified.append(param)

            if audio_count >= 1 and text_count >= 1 and not missing_required:
                self._index2_endpoint_cache = (api_name, fn_index, classified)
                return self._index2_endpoint_cache
            errors.append(
                f"api={api_name or fn_index}, missing={missing_required}, audio={audio_count}, text={text_count}"
            )

        raise StopRetry(
            "未找到可用的 Index-TTS2 接口，请确认 webui 已启动并开放 API: " + "; ".join(errors)
            if config.defaulelang == 'zh'
            else "No compatible Index-TTS2 endpoint detected. Please ensure the web UI exposes an API. Details: " + "; ".join(errors)
        )

    def _build_index2_arguments(self, *,
                                 parameters: List[Dict],
                                 text: str,
                                 ref_wav: str,
                                 ref_text: str,
                                 language_code: Optional[str],
                                 speed: float) -> List:
        args: List = []
        language_code = (language_code or '').strip()
        for param in parameters:
            kind = param.get('__kind__') or ''
            default = param.get('parameter_default')
            has_default = param.get('parameter_has_default', False)
            is_optional = param.get('optional', False) or param.get('parameter_optional', False)

            try:
                if kind == 'text':
                    args.append(text)
                elif kind == 'ref_text':
                    args.append(ref_text or text)
                elif kind in ('audio', 'emotion'):
                    args.append(handle_file(ref_wav))
                elif kind == 'language':
                    args.append(language_code or default or 'en')
                elif kind == 'remove_silence':
                    args.append(True)
                elif kind == 'speaker':
                    args.append(default if default is not None else 0)
                elif kind == 'seed':
                    args.append(default if default is not None else 42)
                elif kind == 'speed':
                    args.append(speed)
                elif kind == 'tuning':
                    args.append(default if default is not None else 1.0)
                else:
                    if has_default:
                        args.append(default)
                    elif is_optional:
                        args.append(None)
                    else:
                        raise StopRetry(
                            f"Index-TTS2 缺少必要参数: {param.get('label') or param.get('parameter_name')}"
                            if config.defaulelang == 'zh'
                            else f"Index-TTS2 missing required parameter: {param.get('label') or param.get('parameter_name')}"
                        )
            except StopRetry:
                raise
            except Exception as e:
                raise StopRetry(
                    f"组装 Index-TTS2 参数失败: {e}" if config.defaulelang == 'zh' else f"Failed to prepare Index-TTS2 arguments: {e}"
                )

        return args

    def _prepare_index2_reference(self, ref_wav: str) -> str:
        """Return a trimmed reference clip suitable for Index-TTS2."""
        if not ref_wav:
            return ref_wav

        ref_path = Path(ref_wav)
        if not ref_path.exists():
            return ref_wav

        cache = self._index2_ref_short_cache.get(ref_wav)
        if cache and Path(cache).exists():
            return cache

        max_seconds = config.settings.get('index_tts2_ref_seconds', 8)
        try:
            max_seconds = float(max_seconds)
        except Exception:
            max_seconds = 8.0
        max_seconds = max(1.0, min(max_seconds, 30.0))

        try:
            duration = tools.get_audio_time(ref_path.as_posix())
        except Exception:
            duration = None

        if duration is None or duration <= max_seconds + 0.1:
            return ref_wav

        cache_dir = Path(config.TEMP_DIR) / 'index2_refs'
        cache_dir.mkdir(parents=True, exist_ok=True)
        short_name = f"{tools.get_md5(ref_path.as_posix())}-{int(max_seconds)}s.wav"
        short_path = cache_dir / short_name

        if short_path.exists():
            self._index2_ref_short_cache[ref_wav] = short_path.as_posix()
            return short_path.as_posix()

        end_time = min(duration, max_seconds)
        if end_time <= 0:
            return ref_wav

        try:
            tools.cut_from_audio(
                audio_file=ref_path.as_posix(),
                ss=0,
                to=end_time,
                out_file=short_path.as_posix()
            )
        except Exception as exc:
            config.logger.warning(f'Failed to trim Index-TTS2 reference audio: {exc}')
            return ref_wav

        if short_path.exists():
            self._index2_ref_short_cache[ref_wav] = short_path.as_posix()
            return short_path.as_posix()

        return ref_wav

    def _item_task_index2(self, data_item: Union[Dict, List, None]):

        speed = 1.0
        try:
            speed = 1 + float(self.rate.replace('%', '')) / 100
        except Exception:
            pass

        text = data_item['text'].strip()
        role = data_item['role']
        data = {'ref_wav': ''}

        if role == 'clone':
            data['ref_wav'] = data_item['ref_wav']
        else:
            roledict = tools.get_f5tts_role()
            if role in roledict:
                data['ref_wav'] = config.ROOT_DIR + f"/f5-tts/{role}"

        if not Path(data['ref_wav']).exists():
            raise StopRetry(f'{role} 角色不存在' if config.defaulelang == 'zh' else f'Role {role} does not exist')

        try:
            client = Client(self.api_url, httpx_kwargs={"timeout": 7200}, ssl_verify=False)
        except Exception as e:
            raise StopRetry(str(e))

        ref_wav = self._prepare_index2_reference(data['ref_wav'])
        api_name, fn_index, parameters = self._resolve_index2_endpoint(client)
        args = self._build_index2_arguments(
            parameters=parameters,
            text=text,
            ref_wav=ref_wav,
            ref_text=data_item.get('ref_text', ''),
            language_code=self.language,
            speed=speed
        )

        try:
            result = client.predict(
                *args,
                api_name=api_name,
                fn_index=fn_index
            )
        except TypeError as e:
            self._index2_endpoint_cache = None
            raise StopRetry(
                f"Index-TTS2 调用失败: {e}. 可尝试刷新页面后重试" if config.defaulelang == 'zh'
                else f"Index-TTS2 invocation failed: {e}. Try refreshing the schema"
            )
        except Exception:
            raise

        config.logger.info(f'result={result}')
        wav_file = result[0] if isinstance(result, (list, tuple)) and result else result
        if isinstance(wav_file, dict) and "value" in wav_file:
            wav_file = wav_file['value']
        if isinstance(wav_file, str) and Path(wav_file).is_file():
            self.convert_to_wav(wav_file, data_item['filename'])
        else:
            raise RuntimeError(str(result))

        if self.inst and self.inst.precent < 80:
            self.inst.precent += 0.1

        self.has_done += 1
        self._signal(text=f'{config.transobj["kaishipeiyin"]} {self.has_done}/{self.len}')

    def _item_task_dia(self, data_item: Union[Dict, List, None]):

        speed = 1.0
        try:
            speed = 1 + float(self.rate.replace('%', '')) / 100
        except:
            pass

        text = data_item['text'].strip()
        role = data_item['role']
        data = {'ref_wav': ''}

        if role == 'clone':
            data['ref_wav'] = data_item['ref_wav']
        else:
            roledict = tools.get_f5tts_role()
            if role in roledict:
                data['ref_wav'] = config.ROOT_DIR + f"/f5-tts/{role}"
                data['ref_text'] = roledict.get('ref_text', '')

        if not Path(data['ref_wav']).exists():
            self.error = f'{role} 角色不存在'
            raise StopRetry(self.error)
        try:
            client = Client(self.api_url, httpx_kwargs={"timeout": 7200, "proxy": None}, ssl_verify=False)
        except Exception as e:
            raise StopRetry(str(e))
        try:
            result = client.predict(
                text_input=text,
                audio_prompt_input=handle_file(data['ref_wav']),
                transcription_input=data.get('ref_text', ''),
                api_name='/generate_audio'
            )
        except Exception as e:
            raise

        config.logger.info(f'result={result}')
        wav_file = result[0] if isinstance(result, (list, tuple)) and result else result
        if isinstance(wav_file, dict) and "value" in wav_file:
            wav_file = wav_file['value']
        if isinstance(wav_file, str) and Path(wav_file).is_file():
            self.convert_to_wav(wav_file, data_item['filename'])
        else:
            raise RuntimeError(str(result))

        if self.inst and self.inst.precent < 80:
            self.inst.precent += 0.1

        self.has_done += 1
        self._signal(text=f'{config.transobj["kaishipeiyin"]} {self.has_done}/{self.len}')

    def _item_task(self, data_item: Union[Dict, List, None]):

        # Spark-TTS','Index-TTS Dia-TTS
        @retry(retry=retry_if_not_exception_type(NO_RETRY_EXCEPT), stop=(stop_after_attempt(RETRY_NUMS)),
               wait=wait_fixed(RETRY_DELAY), before=before_log(config.logger, logging.INFO),
               after=after_log(config.logger, logging.INFO))
        def _run():
            ttstype = config.params.get('f5tts_ttstype')
            if self._exit():
                return
            if ttstype == 'Spark-TTS':
                self._item_task_spark(data_item)
            elif ttstype == 'Index-TTS':
                self._item_task_index(data_item)
            elif ttstype == 'Index-TTS2':
                self._item_task_index2(data_item)
            elif ttstype == 'Dia-TTS':
                self._item_task_dia(data_item)
            else:
                self._item_task_v1(data_item)




        try:
            _run()
        except RetryError as e:
            raise e.last_attempt.exception()
        except Exception as e:
            self.error = e


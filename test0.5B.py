import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

import time
import torch
import tracemalloc
import psutil
import os



def measure_cosyvoice_inference(cosyvoice, infer_func, *args, **kwargs):
    print("Start measuring inference performance...")

    # 记录 CPU 占用内存（进程级）
    process = psutil.Process(os.getpid())
    cpu_mem_before = process.memory_info().rss / 1024 / 1024  # MB

    # 开始 tracemalloc
    tracemalloc.start()
    start_snapshot = tracemalloc.take_snapshot()

    # 开始计时
    start_time = time.time()
    result = list(infer_func(*args, **kwargs))  # 运行推理
    end_time = time.time()

    # 内存追踪
    end_snapshot = tracemalloc.take_snapshot()
    stats = end_snapshot.compare_to(start_snapshot, 'lineno')
    tracemalloc.stop()

    cpu_mem_after = process.memory_info().rss / 1024 / 1024  # MB
    cpu_mem_delta = cpu_mem_after - cpu_mem_before

    mem_trace_delta = sum(stat.size_diff for stat in stats) / 1024 / 1024

    # GPU 占用（如可用）
    gpu_allocated = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    gpu_reserved = torch.cuda.memory_reserved() / 1024 / 1024 if torch.cuda.is_available() else 0

    print(f"推理耗时: {end_time - start_time:.3f} 秒")
    print(f"CPU内存变化: {cpu_mem_delta:.2f} MB (tracemalloc delta: {mem_trace_delta:.2f} MB)")
    print(f"GPU分配内存: {gpu_allocated:.2f} MB.预留内存: {gpu_reserved:.2f} MB")

    return {
        "time_sec": end_time - start_time,
        "cpu_memory_mb": cpu_mem_delta,
        "tracemalloc_delta_mb": mem_trace_delta,
        "gpu_mem_allocated_mb": gpu_allocated,
        "gpu_mem_reserved_mb": gpu_reserved,
        "result": result
    }



cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False)
prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)

# 包装推理调用
metrics = measure_cosyvoice_inference(
    cosyvoice,
    cosyvoice.inference_zero_shot,
    '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。',
    '希望你以后能够做的比我还好呦。',
    prompt_speech_16k,
    stream=False
)




# # save zero_shot spk for future usage
# assert cosyvoice.add_zero_shot_spk('希望你以后能够做的比我还好呦。', prompt_speech_16k, 'my_zero_shot_spk') is True
# for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '', '', zero_shot_spk_id='my_zero_shot_spk', stream=False)):
#     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
# cosyvoice.save_spkinfo()

# # fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L248
# for i, j in enumerate(cosyvoice.inference_cross_lingual('在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。', prompt_speech_16k, stream=False)):
#     torchaudio.save('fine_grained_control_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# # instruct usage
# for i, j in enumerate(cosyvoice.inference_instruct2('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '用四川话说这句话', prompt_speech_16k, stream=False)):
#     torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# # bistream usage, you can use generator as input, this is useful when using text llm model as input
# # NOTE you should still have some basic sentence split logic because llm can not handle arbitrary sentence length
# def text_generator():
#     yield '收到好友从远方寄来的生日礼物，'
#     yield '那份意外的惊喜与深深的祝福'
#     yield '让我心中充满了甜蜜的快乐，'
#     yield '笑容如花儿般绽放。'
# for i, j in enumerate(cosyvoice.inference_zero_shot(text_generator(), '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
#     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
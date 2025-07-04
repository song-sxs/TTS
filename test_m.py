from torchmetrics.audio import DeepNoiseSuppressionMeanOpinionScore
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
from testsparsityv2 import create_audio_tuples
import numpy as np
import torch
import pandas as pd
import os
import glob

def cal_dnsmos(waveform_tensor):
    dnsmos = DeepNoiseSuppressionMeanOpinionScore(fs=16000, personalized=False)
    score = dnsmos(waveform_tensor)  # waveform shape: [time], fs=16000

    return score


if __name__ == "__main__":
    
        # CAUSAL_RESNET_BLOCKS = (
    # # up + mid + down = 1 + 12 + 1 =14
	# ["decoder.estimator.up_blocks.0.0"] + 
	# [f"decoder.estimator.mid_blocks.{i}.0" for i in range(12)] +
    # ["decoder.estimator.down_blocks.0.0"] 
    # )

    # BASIC_TRANSFORMER_BLOCKS = (
    # # up + mid + down = 4 + 48 + 4 =56
	# [f"decoder.estimator.up_blocks.0.1.{i}" for i in range(4)] +
	# [f"decoder.estimator.mid_blocks.{i}.1.{j}" for i in range(12) for j in range(4)] +
    # [f"decoder.estimator.down_blocks.0.1.{i}" for i in range(4)]
    # )
        
    # print("原模型非零参数数量：", count_nonzero_parameters(flow_blocks))

    test_clean_100_path = "/home/linlinsong/CosyVoice/LibriTTS/test-clean"    
    pairs = create_audio_tuples(test_clean_100_path, 10)
    
    # 获取result_model目录下所有的.pth文件
    model_files = glob.glob('result_model/*.pth')
    results = []

    for model_path in model_files:
        model_name = os.path.basename(model_path)  # 获取文件名
        
        try:
            cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False)
            print(f"Loading model: {model_name}")
            state_dict = torch.load(model_path)  
            cosyvoice.model.flow.load_state_dict(state_dict)

            score_list = []
            for k in range(len(pairs)):
                prompt_speech_16k = load_wav(pairs[k][2],16000)
                for i, j in enumerate(cosyvoice.inference_zero_shot(pairs[k][0], pairs[k][1], prompt_speech_16k, stream=False)):
                    # torchaudio.save('result_audio/'+ model_name + '_shot{}_{}.wav'.format(k,i), j['tts_speech'], cosyvoice.sample_rate)
                    # score = cal_dnsmos('result_audio/'+ model_name + '_shot{}_{}.wav'.format(k,i))
                    score = cal_dnsmos(j['tts_speech'])
                    score_list.append(score)

            stacked_scores = torch.stack(score_list) 
            column_means = torch.mean(stacked_scores, dim=0)
            column_means_rounded = np.round(column_means.numpy(), 2)
            # print("model_name: ", model_name)
            # print("mean_score: ", column_means_rounded)
            results.append([model_name] + column_means_rounded.tolist())
            
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            print(f"Skipping model {model_name}")
            continue

    if results:
        df = pd.DataFrame(results, columns=["model_name", "nmos", "mos_sig", "mos_bak", "mos_ovr"])
        df.to_csv("all_model_scores.csv", index=False)
        print(f"Results saved to all_model_scores.csv with {len(results)} models")
    else:
        print("No models were successfully processed")


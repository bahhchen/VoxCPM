import sys
import torchaudio
import torch
import re
import os
import torch_complex

import soundfile as sf
import numpy as np
from voxcpm import VoxCPM

# print("PyTorch version:", torch.__version__)
# print("Torchaudio version:", torchaudio.__version__)
# # print("Torch-complex version:", torch_complex.__version__)
# print("CUDA available:", torch.cuda.is_available())
# if torch.cuda.is_available():
#     print("CUDA device:", torch.cuda.get_device_name(0))
#     print("CUDA version (runtime):", torch.version.cuda)

from chh_prompts import prompts as g_prompts

def save_waves(filename, waves, sample_rate):
    
    # 获取上级目录
    dir_path = os.path.dirname(filename)
    # 判断是否存在，不存在就创建
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # 在时间维拼接
    merged = np.concatenate(waves)

    sf.write(
        filename,
        merged,
        sample_rate
    )

def inference_zero_shot(model, txt_contents, prompts, dstwav):
    waves = []

    prompts_key0 = next(iter(prompts))
    def_speaker = prompts[prompts_key0]

    # "Speaker namexxx: 别整那些没用的了!"
    pattern = r"^Speaker\s+(\w+):\s*(.+)$"    

    for txt in txt_contents:

        m = re.match(pattern, txt)
        if m:
            speaker_name = m.group(1)   # namexxx
            text = m.group(2)           # 别整那些没用的了!
            if speaker_name in prompts:
                cur_speaker = prompts[speaker_name]
            else:
                cur_speaker = def_speaker
        else:
            text = txt
            cur_speaker = def_speaker

        # print("speaker_name:", speaker_name)
        # print("text:", text)

        wav = model.generate(
            text=text,
            prompt_wav_path=cur_speaker['prompt_wav'],      # optional: path to a prompt speech for voice cloning
            prompt_text=cur_speaker['prompt_text'],          # optional: reference text
            cfg_value=2.0,             # LM guidance on LocDiT, higher for better adherence to the prompt, but maybe worse
            inference_timesteps=10,   # LocDiT inference timesteps, higher for better result, lower for fast speed
            normalize=False,           # enable external TN tool, but will disable native raw text support
            denoise=False,             # enable external Denoise tool, but it may cause some distortion and restrict the sampling rate to 16kHz
            retry_badcase=True,        # enable retrying mode for some bad cases (unstoppable)
            retry_badcase_max_times=3,  # maximum retrying times
            retry_badcase_ratio_threshold=6.0, # maximum length restriction for bad case detection (simple but effective), it could be adjusted for slow pace speech
        )
        waves.append(wav)            

    save_waves(dstwav, waves, model.tts_model.sample_rate)   

def read_txt_speaker_paragraphs(file_path):
    """
    读取 TXT 文件，将段落按 speaker 
    """
    paragraphs = []
    speaker_pattern = re.compile(r'^Speaker\s*\S*:')  # 匹配 Speaker namexxx:

    with open(file_path, "r", encoding="utf-8-sig") as f:
        noSpeaker = []
        for line in f:
            line = line.strip()
            if not line:
                continue  # 忽略空行

            if speaker_pattern.match(line):
                if len(noSpeaker) > 0:
                    paragraphs.append("。".join(noSpeaker))
                # 新段落
                paragraphs.append(line)
                noSpeaker = []
            else:
                noSpeaker.append(line)

        if len(noSpeaker) > 0:
            paragraphs.append("。".join(noSpeaker))

    return paragraphs


def voxcpm_example():
    model = VoxCPM.from_pretrained(hf_model_id = "./chh/pretrained_models/VoxCPM1.5", 
                                    zipenhancer_model_id = "./chh/modelscope/speech_zipenhancer_ans_multiloss_16k_base",
                                    # optimize = False,
                                    local_files_only = True)

    # Non-streaming
    # wav = model.generate(
    #     text="波奇酱你搁这儿呢啊! 虽然不知道你咋整的, 我还是买了一裤兜子甜水呢! 卧槽! 撩了的吉他小妹儿! 喜多, 你怎么搁这儿呢?",
    #     prompt_wav_path=g_prompts['YM_woman']['prompt_wav'],      # optional: path to a prompt speech for voice cloning
    #     prompt_text=g_prompts['YM_woman']['prompt_text'],          # optional: reference text
    #     cfg_value=2.0,             # LM guidance on LocDiT, higher for better adherence to the prompt, but maybe worse
    #     inference_timesteps=10,   # LocDiT inference timesteps, higher for better result, lower for fast speed
    #     normalize=False,           # enable external TN tool, but will disable native raw text support
    #     denoise=False,             # enable external Denoise tool, but it may cause some distortion and restrict the sampling rate to 16kHz
    #     retry_badcase=True,        # enable retrying mode for some bad cases (unstoppable)
    #     retry_badcase_max_times=3,  # maximum retrying times
    #     retry_badcase_ratio_threshold=6.0, # maximum length restriction for bad case detection (simple but effective), it could be adjusted for slow pace speech
    # )

    # sf.write("output.wav", wav, model.tts_model.sample_rate)
    # print("saved: output.wav")


    txt_contents = [
        'Speaker YM_woman: 波奇酱你搁这儿呢啊! 虽然不知道你咋整的, 我还是买了一裤兜子甜水呢! 卧槽! 撩了的吉他小妹儿! 喜多, 你怎么搁这儿呢?',
        'Speaker YM_woman: 卧槽! 这谁啊?',
        'Speaker ZL2_woman: 别整那些没用的了!',
    ]
    txt_contents = read_txt_speaker_paragraphs("./chh/books/1996 终极实验 - 罗伯特·索耶 - fix/0004.1996 终极实验 - 罗伯特·索耶.引子.txt")
    inference_zero_shot(model, txt_contents, g_prompts, './chh/output/引子.wav')

def main():
    voxcpm_example()


if __name__ == '__main__':
    main()
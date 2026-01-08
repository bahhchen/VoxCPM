import sys
import torchaudio
import torch
import re
import os
import torch_complex

import soundfile as sf
import numpy as np
from voxcpm import VoxCPM

from chh.ebook import split_text_by_length
from chh.prompts import prompts as g_prompts
from chh.book_reader import read_book, read_txt_speaker_paragraphs

import json

# print("PyTorch version:", torch.__version__)
# print("Torchaudio version:", torchaudio.__version__)
# # print("Torch-complex version:", torch_complex.__version__)
# print("CUDA available:", torch.cuda.is_available())
# if torch.cuda.is_available():
#     print("CUDA device:", torch.cuda.get_device_name(0))
#     print("CUDA version (runtime):", torch.version.cuda)


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

def inference_zero_shot(model, txt_contents, prompts, dstwav, steps = 10):
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
            if speaker_name not in prompts:
            #     cur_speaker = prompts[speaker_name]
            # else:
                # cur_speaker = def_speaker
                speaker_name = prompts_key0
        else:
            text = txt
            # cur_speaker = def_speaker
            speaker_name = prompts_key0

        # print("speaker_name:", speaker_name)
        # print("text:", text)

        segments = split_text_by_length(text, 512)
        for txt_segment in segments:
            print(f"INFO synthesis text：{txt_segment}")
            cur_speaker = prompts['s_' + speaker_name] if len(txt_segment) <= 6 else prompts['l_' + speaker_name]
            wav = model.generate(
                text=txt_segment,
                prompt_wav_path=cur_speaker['prompt_wav'],      # optional: path to a prompt speech for voice cloning
                prompt_text=cur_speaker['prompt_text'],          # optional: reference text
                in_prompt_cache = cur_speaker['prompt_cache'],
                cfg_value=2.0,             # LM guidance on LocDiT, higher for better adherence to the prompt, but maybe worse
                inference_timesteps=steps,   # LocDiT inference timesteps, higher for better result, lower for fast speed
                normalize=True,           # enable external TN tool, but will disable native raw text support
                denoise=False,             # enable external Denoise tool, but it may cause some distortion and restrict the sampling rate to 16kHz
                retry_badcase=True,        # enable retrying mode for some bad cases (unstoppable)
                retry_badcase_max_times=3,  # maximum retrying times
                retry_badcase_ratio_threshold=6.0, # maximum length restriction for bad case detection (simple but effective), it could be adjusted for slow pace speech
            )
            waves.append(wav)
            # r = 1 #2 if len(txt_segment) <= 5 else 1
            # for chunk in model.generate_streaming(
            #     text=txt_segment,
            #     prompt_wav_path=cur_speaker['prompt_wav'],      # optional: path to a prompt speech for voice cloning
            #     prompt_text=cur_speaker['prompt_text'],          # optional: reference text
            #     in_prompt_cache = cur_speaker['prompt_cache'],
            #     cfg_value=2,             # LM guidance on LocDiT, higher for better adherence to the prompt, but maybe worse
            #     inference_timesteps=steps,   # LocDiT inference timesteps, higher for better result, lower for fast speed
            #     normalize=True,           # enable external TN tool, but will disable native raw text support
            #     denoise=False,             # enable external Denoise tool, but it may cause some distortion and restrict the sampling rate to 16kHz
            #     retry_badcase=False,        # enable retrying mode for some bad cases (unstoppable)
            #     retry_badcase_max_times=3,  # maximum retrying times
            #     retry_badcase_ratio_threshold=6.0, # maximum length restriction for bad case detection (simple but effective), it could be adjusted for slow pace speech
            # ):
            #     waves.append(chunk)
                      
    save_waves(dstwav, waves, model.tts_model.sample_rate)   


def initVoxCPM():
    model = VoxCPM.from_pretrained(hf_model_id = "./chh/pretrained_models/VoxCPM1.5", 
                                    zipenhancer_model_id = "./chh/modelscope/speech_zipenhancer_ans_multiloss_16k_base",
                                    # optimize = False,
                                    local_files_only = True)
    
    # 增加缓存
    for key, value in g_prompts.items():
        value['prompt_cache'] = model.tts_model.build_prompt_cache(value['prompt_text'], value['prompt_wav'])
    
    return model
    
# 合成小说 
def inference_book(txt_path, wav_path, chapter_idx = 0, end = '---end---'):
  
    chapters = read_book(txt_path, wav_path, chapter_idx, end)
    # 目录方式
    if len(chapters) < 1:
        return

    model = initVoxCPM()

    # 合成
    for chapter in chapters:
        inference_zero_shot(model, chapter['contents'], g_prompts, chapter['wav'])
    
# 生成提示音
def inference_prompt(model, name, lcontent, scontent):
    txt_contents =[
        'Speaker '+name+':于是，我说道：'+lcontent,
    ]
    lwav = './chh/assets2/zh-l_' + name +'.wav'
    inference_zero_shot(model, txt_contents, g_prompts, lwav, steps=30)
     
    txt_contents =[
        'Speaker '+name+':于是，我说道：'+scontent,
    ]
    swav = './chh/assets2/zh-s_' + name +'.wav'
    inference_zero_shot(model, txt_contents, g_prompts, swav, steps=30)

    res = {}
    res['l_' + name] = {
        'prompt_wav' : lwav,
        'prompt_text' : lcontent,
    }
    res['s_' + name] = {
        'prompt_wav' : swav,
        'prompt_text' : scontent,
    }
    print("res:", json.dumps(res, ensure_ascii=False, indent=4))


    
def voxcpm_example():
    model = initVoxCPM()

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

    # txt_contents = read_txt_speaker_paragraphs("./chh/books/终极实验/0004.终极实验.引子.txt")
    txt_contents =[
        # 'Speaker hb_man:桑德拉睁开眼睛，似乎费了些时间才看清楚。',
        'Speaker s_zyq_woman:是你',
        'Speaker s_zta_woman:什么',
        'Speaker s_ldh_man:走',
        'Speaker s_hb_man:滚',
    ]
    if 1:
        inference_zero_shot(model, txt_contents, g_prompts, './chh/output/test.wav')
    else :
        # inference_prompt(model, 'zly_woman', '公车缓缓驶过街道，窗外景物像电影镜头般掠过，乘客安静地坐在座位上。', '我站在天桥上。')
        # inference_prompt(model, 'fbb_woman', '秋天的树林里，落叶铺满小径，踩上去发出沙沙声，空气中带着淡淡的泥土香。', '切换到夜间模式。')
        # inference_prompt(model, 'bl_woman', '清晨的市场熙熙攘攘，叫卖声、讨价声此起彼伏，生活气息充满街头巷尾。', '给我讲个故事。')
        # inference_prompt(model, 'wyb_man', '山间小路蜿蜒曲折，溪水潺潺流淌，林间充满清新的泥土气息和鸟鸣声。', '花园里开满鲜花。')
        # inference_prompt(model, 'zlc_man', '在海滩上，孩子们堆沙堡、追逐嬉戏，海浪拍打岸边发出悦耳的节奏声。', '孩子们在院子跑。')
        # inference_prompt(model, 'xz_man', '清晨阳光透过薄雾洒在湖面上，水波荡漾，倒映出远山和飞鸟的剪影。', '公园里人影稀疏。')
        # inference_prompt(model, 'st_man', '河边垂钓的人静静坐着，偶尔抛出钓竿，水面泛起微微涟漪，宁静而悠闲。', '小河潺潺流淌。')
        inference_prompt(model, 'lcw_man', '夜晚的咖啡馆里，轻柔的音乐伴随咖啡香，窗外路灯闪烁，行人匆匆而过。', '太阳落山了。')
        # inference_prompt(model, 'zjl_man', '小河弯弯曲曲流向远方，水草随波摇曳，鱼儿偶尔跃出水面，微风吹动水面泛光。', '雪花飘落街道。')
    

def main():
    # voxcpm_example()
    inference_book('./chh/books/终极实验/', './chh/output/终极实验/')


if __name__ == '__main__':
    main()
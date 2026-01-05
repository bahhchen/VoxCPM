
# 分两次执行下载
# from huggingface_hub import snapshot_download
# snapshot_download("openbmb/VoxCPM1.5", local_dir='./chh/pretrained_models/VoxCPM1.5')
# snapshot_download("openbmb/VoxCPM-0.5B", local_dir='./chh/pretrained_models/VoxCPM-0.5B')

from modelscope import snapshot_download
snapshot_download('iic/speech_zipenhancer_ans_multiloss_16k_base', local_dir='./chh/modelscope/speech_zipenhancer_ans_multiloss_16k_base')
snapshot_download('iic/SenseVoiceSmall', local_dir='./chh/modelscope/SenseVoiceSmall')
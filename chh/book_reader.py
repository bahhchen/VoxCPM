
import re
import os

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
            if line == '---end---':
                break
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

# 读取小说 
def read_book(txt_path, wav_path, chapter = 0, chapter2 = 0xfffffff):
  
    books = []

    # 目录方式
    if not os.path.isdir(txt_path):
        return books

    filelist = os.listdir(txt_path)
    for i, file_name in enumerate(filelist):
        if i >= chapter2:
            break
        # print(f"文件名： {i}:{file_name}")
        if i < chapter:
            continue
       
        name_without_ext = os.path.splitext(file_name)[0]
        wav_filename = os.path.join(wav_path, f"{name_without_ext}.wav")
        if os.path.exists(wav_filename):
            continue

        txt_filename = os.path.join(txt_path, file_name)
        txt_contents = read_txt_speaker_paragraphs(txt_filename)
        books.append({"contents": txt_contents, "wav": wav_filename})

    return books
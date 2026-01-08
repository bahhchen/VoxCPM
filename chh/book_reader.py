
import re
import os


# 中文标点 -> 英文标点映射
punct_map = {
    # '，': ',',
    # '。': '.',
    # '！': '!',
    # '？': '?',
    # '：': ':',
    # '；': ';',
    # '（': '(',
    # '）': ')',
    '【': '[',
    '】': ']',
    '·':'-'
    # '“': '"',
    # '”': '"',
    # '‘': "'",
    # '’': "'",
    # '、': ',',
    # '《': '<',
    # '》': '>',
}

# 特殊情况: "——" 和 "…" 需要单独处理
def normalize_punctuation(text: str) -> str:
    # 先替换多字符标点
    # text = text.replace("——", "--").replace("…", "...")
    text = text.replace("……", "，")
    text = text.replace("…", "，")
    
    # 单字符映射用 translate
    # trans_table = str.maketrans(punct_map)
    # text = text.translate(trans_table)

    # 删除非中英文和数字的字符，替换为空格
    # text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\s\.,!?;:()\-—\"'<>]", " ", text)

    return text

def read_txt_speaker_paragraphs(file_path, end):
    """
    读取 TXT 文件，将段落按 speaker 
    """
    paragraphs = []
    speaker_pattern = re.compile(r'^Speaker\s*\S*:')  # 匹配 Speaker namexxx:

    isend = False
    with open(file_path, "r", encoding="utf-8-sig") as f:
        noSpeaker = []
        for line in f:
            line = line.strip()
            if end and line == end:
                isend = True
                break
            if not line:
                continue  # 忽略空行

            nortext = normalize_punctuation(line)

            if speaker_pattern.match(nortext):
                if len(noSpeaker) > 0:
                    paragraphs.append(" ".join(noSpeaker))
                # 新段落
                paragraphs.append(nortext)
                noSpeaker = []
            else:
                noSpeaker.append(nortext)

        if len(noSpeaker) > 0:
            paragraphs.append(" ".join(noSpeaker))

    return paragraphs if ((not end) or isend) else None

# 读取小说 
def read_book(txt_path, wav_path, chapter = 0, end = None):
  
    books = []

    # 目录方式
    if not os.path.isdir(txt_path):
        return books

    filelist = os.listdir(txt_path)
    for i, file_name in enumerate(filelist):
        # print(f"文件名： {i}:{file_name}")
        if i < chapter:
            continue
       
        name_without_ext = os.path.splitext(file_name)[0]
        wav_filename = os.path.join(wav_path, f"{name_without_ext}.wav")
        if os.path.exists(wav_filename):
            continue

        txt_filename = os.path.join(txt_path, file_name)
        txt_contents = read_txt_speaker_paragraphs(txt_filename, end)
        if txt_contents == None:
            break
        books.append({"contents": txt_contents, "wav": wav_filename})

    return books
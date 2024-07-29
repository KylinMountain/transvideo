import lzma
import os
import subprocess
from datetime import timedelta

import ChatTTS
import m3u8
import numpy as np
import pybase16384 as b14
import pysrt
import requests
import torch
from pydub import AudioSegment

from my_translator_agent import translate_agent
from test_chattts import wav_to_mp3


def download_video(data_dir: str, url: str) -> str:
    filename = url.split("/")[-1].split(".")[0]
    output_video = os.path.join(data_dir, filename + ".mp4")
    if os.path.exists(output_video):
        return output_video

    command = ["ffmpeg", "-i", url, "-c", "copy", output_video]
    try:
        p = subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f'Error occurred when downloading video: {e}')
        return ""

    return output_video if p.returncode == 0 else ""


def download_subtitle(data_dir: str, lesson_url: str) -> str:
    filename = ""
    try:
        obj = m3u8.load(lesson_url)
        subtitle_m3u8 = obj.media[0].uri if len(obj.media) else ""
        newurl = m3u8.urljoin(obj.base_uri, subtitle_m3u8)
        obj = m3u8.load(newurl)
        vtt = obj.files[0] if len(obj.files) else ""
        newurl = m3u8.urljoin(obj.base_uri, vtt)
        resp = requests.get(newurl)
        filename = os.path.join(data_dir, vtt)
        with open(filename, 'wb') as file:
            file.write(resp.content)
    except requests.exceptions.RequestException as e:
        print(f"An error occurred when downloading subtitle: {e}")
    return filename


def remove_webvtt_header(vtt_path):
    with open(vtt_path, 'r') as file:
        lines = file.readlines()
    with open(vtt_path, 'w') as file:
        if lines[0].startswith("WEBVTT"):
            for line in lines[1:]:
                file.write(line)
    return vtt_path


def convert_srt(vtt_path: str):
    vtt_path = remove_webvtt_header(vtt_path)
    srt_path = vtt_path.split(".")[0] + ".srt"
    if os.path.exists(srt_path):
        return srt_path
    pysrt.open(vtt_path, encoding="utf-8").save(srt_path, encoding="utf-8")
    return srt_path


def merge_video(data_dir: str, video_path: str, subtitle_path: str, translated_audio: str,
                delay_time: float = 0) -> str:
    video_paths = video_path.split("/")[-1].split(".")
    output_video = os.path.join(data_dir, video_paths[0] + "_zh" + "." + video_paths[1])
    if os.path.exists(output_video):
        return output_video

    command = ["ffmpeg", "-y", "-i", video_path, "-i", translated_audio, "-vf",
               f"subtitles={subtitle_path}:force_style='PlayResX=640,PlayResY=360,MarginV=10,Alignment=2',setpts=PTS+{delay_time}/TB",
               "-c:v", "libx264", "-c:a", "aac", "-strict", "experimental", "-map", "0:v", "-map", "1:a", output_video
               ]

    # command = ["ffmpeg", "-y", "-i", video_path, "-i", translated_audio, "-c:v", "libx264", "-c:a", "aac",
    # "-strict", "experimental", "-map", "0:v", "-map", "1:a", output_video ]
    try:
        p = subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f'Error occurred when merging video: {e}')
        return ""
    return output_video if p.returncode == 0 else ""


def translate_subtitle(subtitle_path: str) -> str:
    output_file = subtitle_path.split(".")[0] + "_zh.srt"
    if os.path.exists(output_file):
        return output_file

    subtitles = pysrt.open(subtitle_path, encoding="utf-8")
    translated_subtitles = pysrt.SubRipFile()
    messages = []
    length = 0
    end_punctuation = {'.', '?', '!', '。', '？', '！', ','}
    for i, subtitle in enumerate(subtitles):
        length += len(subtitle.text.strip())
        if subtitle.text.strip():
            length += len(subtitle.text.strip())
            messages.append(subtitle)
            if i == len(subtitles) - 1 or (length > 500 and (subtitle.text.strip()[-1] in end_punctuation)):

                text = ""
                for sub in messages:
                    text += sub.__str__() + "\n"
                translated_text = translate_agent(text)
                translated_subtitles.extend(pysrt.from_string(translated_text))
                messages.clear()
                length = 0
        else:
            if messages:
                text = ""
                for sub in messages:
                    text += sub.__str__() + "\n"
                translated_text = translate_agent(text)
                translated_subtitles.extend(pysrt.from_string(translated_text))
                messages.clear()
                length = 0
            translated_subtitles.append(subtitle)
    # save translated_subtitles to file
    translated_subtitles.save(output_file, encoding="utf-8")
    return output_file


def add_timedelta_to_subriptime(subrip_time, delta):
    """将 timedelta 对象添加到 SubRipTime 对象"""
    total_milliseconds = subrip_time.ordinal + int(delta.total_seconds() * 1000)
    return pysrt.SubRipTime.from_ordinal(total_milliseconds)


def adjust_chinese_subtitles(subtitle_path: str, characters_per_second=2.5) -> str:
    output_file = subtitle_path.split(".")[0] + "_adjust.srt"
    if os.path.exists(output_file):
        return output_file

    new_subs = pysrt.SubRipFile()
    subtitles = pysrt.open(subtitle_path, encoding="utf-8")
    for sub in subtitles:
        if sub.text:  # 不是空行
            duration = time_to_ms(sub.end - sub.start) / 1000
            total_chars = len(sub.text)
            max_chars_per_chunk = int(duration * characters_per_second)

            num_chunks = max(1, (total_chars + max_chars_per_chunk - 1) // max_chars_per_chunk)  # 向上取整
            chunk_size = (total_chars + num_chunks - 1) // num_chunks  # 每个块的字符数

            chunk_start = sub.start

            for i in range(num_chunks):
                chunk_duration = timedelta(seconds=duration * (i + 1) / num_chunks)
                chunk_end = add_timedelta_to_subriptime(sub.start, chunk_duration)
                chunk_text = sub.text[i * chunk_size:(i + 1) * chunk_size]

                sub = pysrt.SubRipItem(index=len(new_subs) + 1, start=chunk_start, end=chunk_end, text=chunk_text)
                new_subs.append(sub)
                chunk_start = chunk_end
        else:  # 是空行，直接添加
            sub.index = len(new_subs) + 1
            new_subs.append(sub)

    # 保存新的中文字幕文件
    new_subs.save(output_file, encoding='utf-8')
    return output_file


def time_to_ms(time):
    return time.hours * 3600000 + time.minutes * 60000 + time.seconds * 1000 + time.milliseconds


def compress_and_encode(tensor):
    np_array = tensor.numpy().astype(np.float16)
    compressed = lzma.compress(np_array.tobytes(), format=lzma.FORMAT_RAW,
                               filters=[{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}])
    encoded = b14.encode_to_string(compressed)
    return encoded


def change_audio_speed(audio, speed=1.0):
    # 调整音频的帧率以改变播放速度
    audio_with_altered_frame_rate = audio._spawn(audio.raw_data, overrides={
        "frame_rate": int(audio.frame_rate * speed)
    })
    # 将帧率恢复到正常，以便播放音频
    return audio_with_altered_frame_rate.set_frame_rate(audio.frame_rate)


def tts(subtitle_path: str) -> str:
    output_audio = subtitle_path.split(".")[0] + ".mp3"
    if os.path.exists(output_audio):
        return output_audio

    chat = ChatTTS.Chat()
    chat.load(compile=False)
    spk = torch.load("asset/seed_1332_restored_emb.pt", map_location=torch.device('cpu')).detach()
    spk_emb_str = compress_and_encode(spk)
    print(spk_emb_str)  # save it for later timbre recovery

    params_infer_code = ChatTTS.Chat.InferCodeParams(
        prompt="[speed_9]",
        spk_emb=spk_emb_str,  # add sampled speaker
        temperature=.0003,  # using custom temperature
        top_P=0.7,  # top P decode
        top_K=20,  # top K decode
    )

    full_audio = AudioSegment.silent(duration=0)
    subtitles = pysrt.open(subtitle_path, encoding="utf-8")
    for i, subtitle in enumerate(subtitles):
        text = subtitle.text
        start_time = time_to_ms(subtitle.start)
        end_time = time_to_ms(subtitle.end)
        duration_ms = end_time - start_time
        # 语速如何控制？
        if text:
            wavs = chat.infer([text.replace('\n', ' ')], use_decoder=True, params_infer_code=params_infer_code)
            mp3_filename = f"temp_{i}.mp3"
            wav_to_mp3(mp3_filename, wavs[0])
            speech = AudioSegment.from_mp3(mp3_filename)
            # add silent time
            if speech.duration_seconds * 1000 < duration_ms:
                # silent = AudioSegment.silent(duration=duration_ms - speech.duration_seconds * 1000)
                # speech += silent
                print(f"{subtitle.index}, 增加静默 audio generated : {speech.duration_seconds - duration_ms / 1000}")
            else:
                # 尝试倍速调整和字幕时间匹配进行语速控制
                speed = speech.duration_seconds * 1000 / duration_ms
                print(f"{subtitle.index}, 需要倍速 {speed}")
                # if speed > 2:
                #     speed = 1.5
                # speech = change_audio_speed(speech, speed=speed)

        else:
            # 空字幕时候
            speech = AudioSegment.silent(duration=duration_ms)

        if i == 0:
            full_audio = full_audio + AudioSegment.silent(duration=start_time)
        if i > 0:
            previous_end_time = time_to_ms(subtitles[i - 1].end)
            pause_duration = start_time - previous_end_time
            if pause_duration > 0:
                pause = AudioSegment.silent(duration=pause_duration)
                full_audio += pause

        full_audio += speech
        if os.path.exists(f"temp_{i}.mp3"):
            os.remove(f"temp_{i}.mp3")

    full_audio.export(output_audio, format="mp3")
    return output_audio


def main():
    data_dir = "data"
    # course_base_url = "https://dyckms5inbsqq.cloudfront.net/Llamaindex-Truera/llamaindex-truera-c1"
    #
    # lessons = [
    #     "llamaindex-truera_c1_01/llamaindex-truera_c1_01_master.m3u8",
    #     "llamaindex-truera_c1_02/llamaindex-truera_c1_02_master.m3u8",
    #     "llamaindex-truera_c1_03/llamaindex-truera_c1_03_master.m3u8",
    #     "llamaindex-truera_c1_04/llamaindex-truera_c1_04_master.m3u8",
    #     "llamaindex-truera_c1_05/llamaindex-truera_c1_05_master.m3u8",
    #     "llamaindex-truera_c1_06/llamaindex-truera_c1_06_master.m3u8"
    # ]

    course_base_url = "https://dyckms5inbsqq.cloudfront.net/OpenAI/ChatGPT_Prompt_Engineering_for_Developer"
    lessons = [
        "prompt_eng_01/prompt_eng_01_master.m3u8",
        # "prompt_eng_02/prompt_eng_02_master.m3u8"
    ]

    # course_base_url = "https://dyckms5inbsqq.cloudfront.net/LlamaIndex/llamaindex-c2"
    # lessons = ["llamaindex_c2_01/llamaindex_c2_01_master.m3u8",
    #            "llamaindex_c2_02/llamaindex_c2_02_master.m3u8",
    #            "llamaindex_c2_03/llamaindex_c2_03_master.m3u8",
    #            "llamaindex_c2_04/llamaindex_c2_04_master.m3u8",
    #            "llamaindex_c2_05/llamaindex_c2_05_master.m3u8",
    #            "llamaindex_c2_06/llamaindex_c2_06_master.m3u8",
    #            ]
    for lesson in lessons:
        lesson1_url = os.path.join(course_base_url, lesson)

        video = download_video(data_dir, lesson1_url)
        vtt = download_subtitle(data_dir, lesson1_url)
        srt = convert_srt(vtt)
        translated_srt = translate_subtitle(subtitle_path=srt)
        translated_audio = tts(translated_srt)
        # adjust_subtitles = adjust_chinese_subtitles(translated_srt, characters_per_second=3)
        print(translated_audio)

        transvideo = merge_video(data_dir, video, translated_srt, translated_audio, delay_time=0)
        print(transvideo)


if __name__ == '__main__':
    main()

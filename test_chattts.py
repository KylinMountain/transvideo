import lzma
import wave
from io import BytesIO

import ChatTTS
import numpy as np
import pybase16384 as b14
import torch
from pydub import AudioSegment


def float32_to_int16(audio: np.ndarray) -> np.ndarray:
    """ Convert float32 audio array to int16 """
    audio = np.clip(audio, -1, 1)  # Ensure all values are in [-1, 1]
    audio = (audio * 32767).astype(np.int16)  # Convert to int16
    return audio


def wav_to_mp3(output_path: str, wav_arr: np.ndarray, sample_rate: int = 24000):
    # Convert float32 array to int16
    wav_arr_int16 = float32_to_int16(wav_arr)

    # Create a temporary buffer to store WAV data
    buf = BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # Sample width in bytes
        wf.setframerate(sample_rate)  # Sample rate in Hz
        wf.writeframes(wav_arr_int16.tobytes())

    # Load WAV data into an AudioSegment
    buf.seek(0)
    audio_segment = AudioSegment.from_wav(buf)

    # Export the AudioSegment as an MP3 file
    mp3_buf = BytesIO()
    audio_segment.export(mp3_buf, format="mp3")

    # Return MP3 data
    mp3_buf.seek(0)

    with open(output_path, "wb") as f:
        f.write(mp3_buf.getbuffer())
    return output_path


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


def tensor_to_str(spk_emb):
    # 将Tensor转换为NumPy数组
    spk_emb_np = spk_emb.cpu().numpy().astype(np.float16)
    # 将NumPy数组编码为字符串
    spk_emb_str = b14.encode_to_string(lzma.compress(spk_emb_np, format=lzma.FORMAT_RAW, filters=[
        {"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}]))
    return spk_emb_str


def numpy_array_to_mp3(numpy_array, output_file):
    # 将 numpy 数组规范化到 [-1, 1] 范围
    numpy_array = np.clip(numpy_array, -1, 1)
    numpy_array = (numpy_array * 32767).astype(np.int16)
    # 将 numpy 数组转换为 int16 类型的 WAV 格式字节流
    byte_stream = numpy_array.tobytes()
    # 创建一个新的 AudioSegment 对象
    audio_segment = AudioSegment(
        data=byte_stream,
        sample_width=2,
        frame_rate=24000,
        channels=1
    )
    # 将 AudioSegment 对象保存为 MP3 文件
    audio_segment.export(output_file, format="mp3")


if __name__ == '__main__':
    chat = ChatTTS.Chat()
    chat.load()

    spk = torch.load("asset/seed_1332_restored_emb.pt", map_location=torch.device('cpu')).detach()
    spk_emb_str = tensor_to_str(spk)
    print(spk_emb_str)  # save it for later timbre recovery

    params_infer_code = ChatTTS.Chat.InferCodeParams(
        spk_emb=spk_emb_str,  # add sampled speaker
        temperature=.0003,  # using custom temperature
        top_P=0.7,  # top P decode
        top_K=20,  # top K decode
    )

    text = "你好，我是X二零四六，欢迎关注LLM深潜：Agent框架与应用揭秘"
    wavs = chat.infer([text], use_decoder=True, params_infer_code=params_infer_code)
    wav_array = wavs[0]

    mp3_filename = "output-llm-agent-introduction.mp3"
    numpy_array_to_mp3(wav_array, mp3_filename)

    # wav_to_mp3(mp3_filename, wav_array)
    # speech = AudioSegment.from_mp3(mp3_filename)
    #
    # speed = speech.duration_seconds * 1000 / (51950 - 49384)
    # print(f"speed: {speed}")
    #
    # speech_speedup = change_audio_speed(speech, speed=speed)
    # speech.export(mp3_filename, format="mp3")
    # print(f"Audio saved to {mp3_filename}")
    # speech_speedup = change_audio_speed2(speech, speed=speed)
    # speech_speedup.export("output_speeded_audio.mp3", format="mp3")

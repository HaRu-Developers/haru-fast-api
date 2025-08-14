from io import BytesIO
import wave

def add_wav_header(audio_data: bytes, sample_rate: int, sample_width: int, channels: int) -> bytes:
    """
    원시 PCM 오디오 데이터에 WAV 헤더를 추가합니다.
    :param audio_data: 원시 PCM 오디오 바이트.
    :param sample_rate: 샘플 레이트 (예: 16000).
    :param sample_width: 샘플 너비 (바이트 단위, 예: 2 for 16-bit).
    :param channels: 채널 수 (예: 1 for mono).
    :return: WAV 파일 형식의 바이트 데이터.
    """
    with BytesIO() as buf:
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)
        return buf.getvalue()
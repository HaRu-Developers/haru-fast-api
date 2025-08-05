from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
import os
import towav

load_dotenv()
app = FastAPI()
elevenlabs = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

SAMPLE_RATE = 16000

@app.post("/stt")
async def stt(request: Request):
    
    audio_PCM = await request.body()
    
    audio_data = towav.add_wav_header(audio_PCM, SAMPLE_RATE, 2, 1)
    
    transcription = elevenlabs.speech_to_text.convert(
        file=audio_data,
        model_id="scribe_v1", # Model to use, for now only "scribe_v1" is supported
        tag_audio_events=False, # Tag audio events like laughter, applause, etc.
        language_code="kor", # Language of the audio file. If set to None, the model will detect the language automatically.
        diarize=True, # Whether to annotate who is speaking
    )
    
    print(transcription.text)
    
    # 시간 순서대로 모든 발언을 담을 리스트
    utterances = []
    
    current_utterance_text = ""
    current_utterance_start_time = None
    last_speaker_id = None

    # 단어 목록을 순회하며 발언 마디를 구분
    for i, word in enumerate(transcription.words):
        if word.type not in {"word", "spacing"}:
            continue

        speaker = word.speaker_id or "unknown"
        
        # 첫 단어가 아니고, 화자가 변경되었을 경우
        if i > 0 and speaker != last_speaker_id:
            # 이전 발언 마디를 리스트에 추가
            if current_utterance_text.strip():
                utterances.append({
                    "speaker_id": last_speaker_id,
                    "text": current_utterance_text.strip(),
                    "start": current_utterance_start_time
                })
            
            # 새로운 발언을 위해 변수 초기화
            current_utterance_text = ""
            current_utterance_start_time = None

        # 현재 단어 처리
        if current_utterance_start_time is None:
            current_utterance_start_time = word.start
        
        current_utterance_text += word.text
        
        last_speaker_id = speaker
        
    # 루프 종료 후 마지막으로 처리 중이던 발언 추가
    if current_utterance_text.strip():
        utterances.append({
            "speaker_id": last_speaker_id,
            "text": current_utterance_text.strip(),
            "start": current_utterance_start_time
        })
    
    return JSONResponse(content={
        "message": "End of speech",
        "utterances": utterances
    })
    
# python -m uvicorn elevenlabs_server:app --host 0.0.0.0 --port 8000 --reload

# {
#   "message": "End of speech",
#   "utterances": [
#     {
#       "speaker_id": "speaker_0",
#       "text": "첫 번째 발언입니다.",
#       "start": 0.119
#     },
#     {
#       "speaker_id": "speaker_1",
#       "text": "두 번째 발언입니다.",
#       "start": 2.105
#     },
#     {
#       "speaker_id": "speaker_0",
#       "text": "세 번째 발언입니다.",
#       "start": 5.678
#     }
#   ]
# }
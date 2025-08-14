from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from sentence_transformers import SentenceTransformer, util
from gliner import GLiNER
import os
import re
from module import towav
import io

# 환경 변수 로드
load_dotenv()

# FastAPI 앱 생성
app = FastAPI(title="Haru FastAPI Server", version="1.0.0")

# ElevenLabs 클라이언트 초기화
elevenlabs = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

# 상수 정의
SAMPLE_RATE = 16000

# --- [NLP 모델 로드] ---
try:
    # 1. 벡터 임베딩 모델 (한국어 범용)
    embedding_model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
    
    # 2. Named Entity Recognition 모델 (한국어 전용 GLiNER)
    ner_model = GLiNER.from_pretrained("taeminlee/gliner_ko")

    print("NLP 모델 로드 완료.")
except Exception as e:
    print(f"모델 로드 중 오류 발생: {e}")
    embedding_model = None
    ner_model = None

# --- [Pydantic 모델 정의] ---
class ScoringRequest(BaseModel):
    speech_id: int
    utterance: str
    has_agenda: bool
    agenda_text: Optional[str] = None
    recent_utterances: Optional[List[str]] = Field(default_factory=list)

class ScoringResponse(BaseModel):
    speech_id: int
    score: float
    isQuestionNeeded: bool

# --- [STT 관련 함수들] ---
@app.post("/stt")
async def stt(request: Request):
    """음성을 텍스트로 변환하고 발화를 구분합니다."""
    audio_PCM = await request.body()
    
    audio_data = towav.add_wav_header(audio_PCM, SAMPLE_RATE, 2, 1)
    
    transcription = elevenlabs.speech_to_text.convert(
        file=io.BytesIO(audio_data),
        model_id="scribe_v1",
        tag_audio_events=False,
        language_code="kor",
        diarize=True,
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

# --- [점수 계산 관련 함수들] ---
def calculate_similarity_score(text1: str, text2: str) -> float:
    """두 텍스트 간의 코사인 유사도 점수를 계산합니다."""
    if embedding_model is None:
        return 0.0
    
    embeddings = embedding_model.encode([text1, text2], convert_to_tensor=True)
    cosine_similarity = util.cos_sim(embeddings[0], embeddings[1])
    return cosine_similarity.item()

def get_similarity_score(request: ScoringRequest) -> float:
    """회의 안건지 또는 최근 발화와의 유사도 점수를 계산합니다."""
    score = 0.0
    
    if request.has_agenda and request.agenda_text:
        # Case 1: 안건지가 있는 경우
        similarity = calculate_similarity_score(request.utterance, request.agenda_text)
        if similarity >= 0.9: score = 12
        elif similarity >= 0.8: score = 7
        elif similarity >= 0.7: score = 3
    elif not request.has_agenda and request.recent_utterances:
        # Case 2: 안건지가 없는 경우, 최근 발화와의 맥락 유사도 계산
        max_similarity = 0.0
        for recent_utterance in request.recent_utterances:
            max_similarity = max(max_similarity, calculate_similarity_score(request.utterance, recent_utterance))
            
        if max_similarity >= 0.85: score = 10
        elif max_similarity >= 0.7: score = 7
    
    return score

def get_keyword_score(utterance: str) -> float:
    """질문 유발 키워드 포함 여부에 따른 점수를 계산합니다."""
    score = 0.0
    found_keywords = set()

    if any(k in utterance for k in ["제안", "추천", "생각", "아이디어"]):
        found_keywords.add("proposal")
    
    if any(k in utterance for k in ["정하자", "결정", "선택", "합의"]):
        found_keywords.add("decision")
    
    if any(k in utterance for k in ["알려줘", "설명", "왜", "어떻게"]):
        found_keywords.add("info")

    for keyword_type in found_keywords:
        if keyword_type == "proposal": score += 5
        elif keyword_type == "decision": score += 3
        elif keyword_type == "info": score += 4
    return score

def get_named_entity_score(utterance: str) -> float:
    """Named Entity 포함 개수에 따른 점수를 계산합니다."""
    if ner_model is None:
        return 0.0

    labels = ["사람", "장소", "날짜", "시간", "프로젝트명", "결정사항", "이슈", "목표", "예산", "제품", "서비스"] 
    
    entities = ner_model.predict_entities(utterance, labels)
    ner_count = len(entities)
    
    if ner_count >= 2:
        return 5
    elif ner_count >= 1:
        return 3
    return 0

def get_format_score(utterance: str) -> float:
    """문장 형식(의문문, 요청)에 따른 점수를 계산합니다."""
    score = 0.0
    
    interrogative_words = ["누가", "무엇을", "왜", "어떻게", "언제", "어디"]
    if "?" in utterance or any(utterance.strip().startswith(q) for q in interrogative_words):
        score += 4
    
    if any(k in utterance for k in ["정리해줘", "결정할까요", "요약해줘"]):
        score += 3
        
    return score

def get_length_score(utterance: str) -> float:
    """발화 길이에 따른 점수를 계산합니다."""
    score = 0.0
    
    utterance_len = len(utterance.strip())
    word_count = len(re.findall(r'\b\S+\b', utterance))
    
    if utterance_len >= 10 and word_count >= 4:
        score += 2
    elif word_count <= 3:
        score -= 3
        
    return score

@app.post("/api/v1/score_utterance", response_model=ScoringResponse)
async def score_utterance(request: ScoringRequest):
    """주어진 발화(utterance)에 대해 질문이 필요한지 여부를 판단하는 점수를 계산합니다."""
    total_score = 0.0
    
    # 각 함수 호출 및 점수 할당
    similarity_score = get_similarity_score(request)
    keyword_score = get_keyword_score(request.utterance)
    named_entity_score = get_named_entity_score(request.utterance)
    format_score = get_format_score(request.utterance)
    length_score = get_length_score(request.utterance)

    # 발화에 대한 각 점수 출력
    print(f"\n[Scoring Utterance]: {request.utterance}")
    print(f"Similarity Score: {similarity_score}")
    print(f"Keyword Score: {keyword_score}")
    print(f"Named Entity Score: {named_entity_score}")
    print(f"Format Score: {format_score}")
    print(f"Length Score: {length_score}")

    # 총점 계산
    total_score = similarity_score + keyword_score + named_entity_score + format_score + length_score

    # 총점 출력
    print(f"\nTotal Score: {total_score}")

    # 질문 필요 여부 결정
    is_question_needed = total_score >= 15.0
    print(f"Is Question Needed: {is_question_needed}")
    
    return ScoringResponse(
        speech_id=request.speech_id,
        score=total_score, 
        isQuestionNeeded=is_question_needed
    )

# --- [헬스체크 엔드포인트] ---
@app.get("/health")
async def health_check():
    """서버 상태를 확인하는 엔드포인트"""
    return {"status": "healthy", "message": "Haru FastAPI Server is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

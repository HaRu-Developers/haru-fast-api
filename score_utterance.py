from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional
from sentence_transformers import SentenceTransformer, util
from gliner import GLiNER
import re # get_length_score 함수에서 사용됩니다.

app = FastAPI()

# --- [서버 시작 시 모델 로드] ---
try:
    # 1. 벡터 임베딩 모델 (한국어 범용)
    embedding_model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
    
    # 2. Named Entity Recognition 모델 (한국어 전용 GLiNER)
    model = GLiNER.from_pretrained("taeminlee/gliner_ko")

    print("NLP 모델 로드 완료.")
except Exception as e:
    print(f"모델 로드 중 오류 발생: {e}")
    embedding_model = None
    model = None

# --- [Pydantic 모델 정의] ---
class ScoringRequest(BaseModel):
    speech_id: int  # speech_id 필드 추가
    utterance: str
    has_agenda: bool
    agenda_text: Optional[str] = None
    recent_utterances: Optional[List[str]] = Field(default_factory=list)

class ScoringResponse(BaseModel):
    speech_id: int  # speech_id 필드 추가
    score: float
    isQuestionNeeded: bool

# --- [기준 구현 함수] ---
def calculate_similarity_score(text1: str, text2: str) -> float:
    """두 텍스트 간의 코사인 유사도 점수를 계산합니다."""
    if embedding_model is None:
        return 0.0
    
    # 텍스트를 임베딩 벡터로 변환
    embeddings = embedding_model.encode([text1, text2], convert_to_tensor=True)
    
    # 두 벡터 간의 코사인 유사도 계산
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
            
        if max_similarity >= 0.85: score = 7
        elif max_similarity >= 0.7: score = 5
    
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
    if model is None:
        return 0.0

    labels = ["사람", "장소", "날짜", "시간", "프로젝트명", "결정사항", "이슈", "목표", "예산", "제품", "서비스"] 
    
    entities = model.predict_entities(utterance, labels)
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

# --- [메인 API 엔드포인트] ---
@app.post("/api/v1/score_utterance", response_model=ScoringResponse)
async def score_utterance(request: ScoringRequest):
    """
    주어진 발화(utterance)에 대해 질문이 필요한지 여부를 판단하는 점수를 계산합니다.
    """
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
        speech_id=request.speech_id,  # 요청받은 speech_id를 그대로 응답에 포함
        score=total_score, 
        isQuestionNeeded=is_question_needed
    )
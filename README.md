# Haru FastAPI Server

통합된 FastAPI 서버로 STT(Speech-to-Text)와 발화 점수 계산 기능을 제공합니다.

## 설치 및 실행

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정
`.env` 파일을 생성하고 다음 내용을 추가하세요:
```
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
```

### 3. 서버 실행
```bash
# 방법 1: Python으로 직접 실행
python main.py

# 방법 2: uvicorn으로 실행
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API 엔드포인트

### 1. 헬스체크
- **GET** `/health`
- 서버 상태 확인

### 2. 음성-텍스트 변환 (STT)
- **POST** `/stt`
- 음성 파일을 텍스트로 변환하고 발화를 구분
- **요청**: PCM 오디오 데이터 (바이너리)
- **응답**: 
```json
{
  "message": "End of speech",
  "utterances": [
    {
      "speaker_id": "speaker_0",
      "text": "첫 번째 발언입니다.",
      "start": 0.119
    }
  ]
}
```

### 3. 발화 점수 계산
- **POST** `/api/v1/score_utterance`
- 발화에 대한 질문 필요성 점수 계산
- **요청**:
```json
{
  "speech_id": 1,
  "utterance": "회의 안건에 대해 제안하고 싶습니다.",
  "has_agenda": true,
  "agenda_text": "프로젝트 진행 방향 논의",
  "recent_utterances": ["이전 발화 내용"]
}
```
- **응답**:
```json
{
  "speech_id": 1,
  "score": 15.0,
  "isQuestionNeeded": true
}
```

## API 문서

서버 실행 후 다음 URL에서 자동 생성된 API 문서를 확인할 수 있습니다:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 기존 파일들과의 차이점

### 이전 구조 (문제점)
- `elevenlabs_server.py`: 독립적인 FastAPI 앱 (포트 8000)
- `score_utterance.py`: 독립적인 FastAPI 앱 (별도 포트 필요)
- 각각 별도 프로세스로 실행해야 함

### 현재 구조 (개선점)
- `main.py`: 통합된 FastAPI 앱
- 모든 엔드포인트가 하나의 서버에서 제공
- 단일 포트(8000)에서 모든 기능 사용 가능
- 코드 중복 제거 및 유지보수성 향상

## 점수 계산 기준

발화 점수는 다음 요소들을 종합하여 계산됩니다:

1. **유사도 점수** (0-15점)
   - 안건지와의 유사도 또는 최근 발화와의 맥락 유사도

2. **키워드 점수** (0-12점)
   - 제안/추천 키워드: +5점
   - 결정/합의 키워드: +3점
   - 정보 요청 키워드: +4점

3. **Named Entity 점수** (0-5점)
   - 2개 이상: +5점
   - 1개: +3점

4. **형식 점수** (0-7점)
   - 의문문: +4점
   - 요청문: +3점

5. **길이 점수** (-3~+2점)
   - 적절한 길이: +2점
   - 너무 짧음: -3점

**총점 10점 이상**일 때 질문이 필요하다고 판단됩니다.

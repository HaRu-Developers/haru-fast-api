FROM python:3.10-slim

# 1. 작업 디렉토리 설정
WORKDIR /app

# 2. 시스템 패키지 설치 (OpenCV가 필요로 하는 공유 라이브러리)
RUN apt-get update && apt-get install -y

# 3. requirements.txt만 먼저 복사 (캐시 활용)
COPY ./fast-api-server/requirements.txt .

# 4. pip 패키지 설치
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 5. 실제 코드 복사
COPY ./fast-api-server .

# 6. PYTHONPATH 환경변수 설정
ENV PYTHONPATH=/app

# 7. FastAPI 서버 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
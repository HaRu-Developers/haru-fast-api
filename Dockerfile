FROM python:3.10-slim

# 1) 작업 디렉토리
WORKDIR /app

# 2) (선택) 최소 베이스 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# 3) requirements만 먼저 복사
COPY ./fast-api-server/requirements.txt .

# 4) pip 업그레이드
RUN pip install --upgrade pip

# 5) 🔽 CPU용 PyTorch 먼저 설치 (권장: 공식 인덱스 사용)
#    - GPU가 아니라면 아래 CPU 인덱스가 가장 깔끔합니다.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1

# 6) 나머지 파이썬 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 7) 실제 코드 복사
COPY ./fast-api-server .

# 8) PYTHONPATH
ENV PYTHONPATH=/app

# 9) FastAPI 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
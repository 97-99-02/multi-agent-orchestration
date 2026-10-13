"""설계 문서의 고정값을 한곳에 모은다. 값을 바꾸면 설계 문서도 같이 고친다."""

# 2.1 LLM 배분 기준
MODEL_ANALYZE = "gpt-4.1-mini"   # 탐색, 기술 요약, 시장성, 경쟁사, 팀, 보고서
# (탐색에 gpt-4.1-nano를 썼으나 해외 기업을 국내로 잘못 판정하고 대표자·제품 칸을 비워 mini로 통일)
MODEL_JUDGE = "gpt-4.1"          # 투자 판단 (생성 모델과 다른 모델)
# 분석·판단 LLM의 출력 상한과 재시도. 드물게 같은 내용을 반복 생성해 32,768토큰까지 가다 실패한 적이 있어
# 상한을 두어 빨리 실패시키고 한 번 더 시도한다 (결과는 JSON이라 4,096토큰이면 충분)
LLM_MAX_TOKENS = 4096
LLM_ATTEMPTS = 2
# 같은 입력이면 최대한 같은 응답이 나오게 한다 (OpenAI seed: 완전한 보장은 아님). temperature=0만으로는
# 같은 기업 분석이 호출마다 달라져 점수가 흔들렸다
LLM_SEED = 42

# 2.3 임베딩: 후보 3종 비교 실험(eval/embedding) 결과 MRR@5·Hit@3 최고, 인코딩 최속
EMBEDDING_MODEL = "nlpai-lab/KURE-v1"

# 2.4 Vector DB
CHROMA_DIR = "vectorstore"
DOC_TYPES = ("tech", "market")   # 메타데이터 doc_type 값

# 2.1 탐색 규칙
SEARCH_QUERIES = [          # 실행마다 후보가 크게 바뀌지 않도록 고정. 세부 분야별로 한 개 이상
    "국내 AI 반도체 스타트업 시리즈 투자 유치",
    "NPU 스타트업 투자 유치 시리즈",
    "LLM 추론 전용 칩 스타트업 투자 유치",       # 데이터센터
    "온디바이스 엣지 AI 반도체 스타트업 투자",    # 엣지
    "차량용 AI 반도체 스타트업 투자 유치",        # 차량용
    "CXL 메모리 반도체 스타트업 투자 유치",       # 인프라
]
ALLOWED_ROUNDS = ("Seed", "Pre-A", "Series A", "Series B", "Series C")
MAX_CANDIDATES = 5

# 3.2 평가 항목과 비중 (합계 100)
WEIGHTS = {
    "tech": 20,         # 기술·제품 성숙도 (양산 전 기업은 최대 3점이라 30에서 하향)
    "market": 30,       # 시장성
    "team": 25,         # 창업자·팀
    "traction": 10,     # 실적·고객 검증
    "competition": 10,  # 경쟁 우위
    "deal": 5,          # 투자조건
}

# 3.4 투자 결정 규칙
INVEST_THRESHOLD = 70
CORE_ITEMS = ("team", "tech")   # 1점 이하이면 보류 (핵심 역량 미달)
CORE_MIN_SCORE = 2              # 이 점수 미만(1점 이하)이면 보류
CORE_CAUTION_SCORE = 2           # 핵심 항목이 이 점수 이하이면 보류가 아니어도 보고서에 주의 코멘트를 남긴다

# 4.1 사실 검증 재작성 횟수
MAX_REPORT_RETRY = 1

# Supervisor 종료 보장 (반복 상한)
MAX_REWORK = 1          # 관점당 재작업 횟수 (기업 단위). 공개 정보가 없으면 다시 찾아도 없다
MAX_QUALITY_LOOPS = 2   # 품질 평가 미달로 보고서를 다시 쓰는 횟수
# 보고서 품질 평가 기준 (agents/quality.py 코드 검사)
QUALITY_CITED_RATIO = 0.8        # Groundedness: 분석 장에서 숫자가 든 문단 중 [n] 인용이 있는 비율
QUALITY_MAX_UNVERIFIED = 2       # Groundedness: 사실 검증이 남긴 '(확인 필요)' 표시 허용 개수
QUALITY_MIN_PUBLISHERS = 3       # 편향 통제: 인용된 출처의 발행처 수
QUALITY_MAX_SOURCE_SHARE = 0.4   # 편향 통제: 출처 하나가 전체 인용에서 차지하는 비율 상한
QUALITY_MIN_SECTION_CHARS = 200  # 관점 커버리지: 관점 장의 최소 분량 (표 제외)

# supervisor 실행 횟수 상한. 후보 1곳에 약 5회(탐색, 분석, 재작업, 판단 뒤마다 1회) × 최대 5곳 = 25,
# 보고서 이후 약 12회(작성·사실 검증·재작성·품질 평가 루프) = 37 → 여유를 두어 40
MAX_STEPS = 40
# LangGraph 단계 상한(마지막 안전장치). supervisor와 하위 노드가 번갈아 실행되므로 supervisor 1회당 2단계
# + 여유. 실제 종료는 MAX_STEPS 등 위 상한이 담당하고, 이 값에 걸리면 설계 오류다
RECURSION_LIMIT = 2 * MAX_STEPS + 10
MAX_RESUME = 1          # 실행 중 예외로 멈췄을 때 마지막 체크포인트에서 이어가는 횟수

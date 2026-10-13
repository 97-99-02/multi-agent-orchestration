"""보고서 생성 에이전트 (reporter) — 담당: 이진우

하는 일:
    1) reporter_node  : State에 쌓인 분석 결과로 투자 보고서(마크다운)를 만든다. (LangGraph 노드)
    2) export_report_pdf: 검증이 끝난 보고서를 제출용 PDF로 저장한다. (그래프 실행 후 app.py에서 호출)

핵심 설계 — "숫자는 코드가, 문장은 AI가":
    - 문장(해설): AI(LLM)가 쓴다.
    - 점수표·환산 점수: 코드가 State에서 직접 만들어 끼운다.
    - REFERENCE·[n] 번호: 코드가 State의 sources로 만든다. (AI가 출처를 지어낼 수 없다)
    - 분량·순서·SUMMARY 틀 같은 과제 조건: 코드가 점검하고 어기면 한 번 고쳐 쓰게 한다.

파일 구성(위에서 아래로):
    [A] 프롬프트  [B] 평가표 상수·계산  [C] 점수표  [D] 출처(REFERENCE)
    [E] 과제 조건 점검  [F] 보고서 생성 노드  [G] PDF 저장(대시보드·디자인)
    사실 검증(verifier.py)도 [B]~[D]의 도구를 가져다 쓴다.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from datetime import date
from pathlib import Path

from urllib.parse import urlparse

from reportlab.graphics.shapes import Circle, Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFError, TTFont
from reportlab.platypus import (CondPageBreak, Flowable, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)
from reportlab.platypus.doctemplate import LayoutError

from agents.competitor import NO_COMPARISON, claim_text
from agents.judge import RUBRIC
from config import (ALLOWED_ROUNDS, CORE_CAUTION_SCORE, CORE_ITEMS, CORE_MIN_SCORE, INVEST_THRESHOLD,
                    MAX_CANDIDATES, MODEL_ANALYZE)
from config import WEIGHTS as TEAM_WEIGHTS

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════
# [A] 프롬프트 — 문구만 고칠 때는 이 부분만 수정
# ═════════════════════════════════════════════════════════════
# ── 보고서 생성 ──────────────────────────────────────────────
# 시스템 프롬프트: LLM에게 '역할'과 '절대 지킬 규칙'을 알려주는 부분
SYSTEM_PROMPT = """당신은 AI 반도체 스타트업 투자 보고서를 쓰는 애널리스트입니다.
독자는 바쁜 투자위원입니다. 1분 안에 결론과 근거를 파악할 수 있게 쓰세요.
반드시 아래 규칙을 지키세요.
1. 제공된 [분석 자료]에 있는 내용과 숫자만 사용합니다. 자료에 없는 숫자·사실은 절대 만들지 않습니다.
2. 숫자는 자료에 적힌 표기 그대로 옮깁니다. (반올림·단위 변환·추정 금지)
   '수십억 달러', '수천억 원' 같은 어림 표현을 쓰지 말고 자료의 수치를 그대로 씁니다.
   다른 단위로 바꾼 값을 괄호로 덧붙이지 않습니다. (예: '백만달러' 수치를 '억 달러'로 바꿔 함께 쓰지 않음)
3. 자료에 정보가 없으면 '정보 부족'이라고 씁니다.
4. 출력은 마크다운이며, 아래 [목차]의 순서와 제목('## ' 형식)을 그대로 따릅니다.
5. 맨 앞은 'SUMMARY'입니다. 900자 이내이며, 투자 추천 보고서는 반드시 아래 틀을 그대로 지킵니다.
   **결론:** 결정과 환산 점수를 담은 한 문장. [분석 자료]의 selection.candidates 후보 이름을 자료 표기 그대로
   모두 쓰고, 그중에서 이 기업이 선정되었다는 것을 한 구절로 밝힙니다.
   (예: 후보 A, B, C 중 선정된 B는 환산 점수 76점으로 투자를 추천한다.)
   **핵심 근거**
   - 근거 (정확히 3개, 줄마다 끝에 출처 번호)
   **핵심 리스크**
   - 리스크 (정확히 3개, 줄마다 끝에 그 리스크를 다룬 출처 번호)
   **확인 필요**
   - 추가 확인이 필요한 사항 (1~2개). 점수를 다시 쓰는 것이 아니라 무엇을 확인해야 하는지 씁니다.
     [분석 자료]에 summary_caution이 있으면 그 내용(항목명, 점수, 확인할 다음 점수 조건)을 한 항목으로 반드시 포함합니다.
     항목명·점수는 summary_caution에 적힌 값을 그대로 씁니다. summary_caution이 없으면 그 항목은 쓰지 않고 항목 점수도 쓰지 않습니다.
     보고서 어디에서든 항목 점수를 쓸 때는 scores의 점수와 같아야 합니다.
   [분석 자료]의 키 이름(summary_caution, team_analysis 등)은 본문 어디에도 쓰지 않습니다.
6. 'REFERENCE' 장은 쓰지 않습니다. (코드가 실제 사용 출처로 자동 첨부합니다.)
7. 본문 전체는 한글 기준 약 10,000~14,000자로 씁니다. (PDF는 대시보드·REFERENCE 포함 7~10장)
   1~4장(관점 장)은 [분석 자료]에 있는 사실을 빠짐없이 2~4문단으로 자세히 씁니다. 분량을 채우려고 자료에 없는 내용이나
   같은 말 반복을 넣지 않습니다. 자료가 적은 장은 짧아도 됩니다.
8. '투자 판단 결과' 장에는 점수표를 직접 쓰지 말고, 표가 들어갈 자리에 {{SCORE_TABLE}} 한 줄만 쓰세요.
   그 아래에 항목별 근거를 문장으로 해설합니다. (표는 코드가 자동으로 채웁니다.)
9. 수치나 사실을 말하는 문장 끝에는 [분석 자료]의 sources_for_citation에 있는 번호를 [1]처럼 붙이세요.
   번호가 여러 개면 [1][2]처럼 하나씩 따로 씁니다. ([1,2]·[1-2]처럼 묶지 않습니다)
   대괄호 안에는 숫자만 씁니다. [team_analysis]처럼 자료 이름을 넣지 않습니다.
   목록에 없는 번호는 절대 쓰지 말고, 근거 출처가 없으면 번호를 붙이지 않습니다.
10. '경쟁 구도' 장에는 경쟁사 목록·차별점·경쟁 리스크를 직접 나열하지 말고, 그 자리에 {{COMPETITOR_TABLE}} 한 줄만 쓰세요.
    (구도 그림과 차별점·리스크 목록은 코드가 자동으로 채웁니다.) 그 아래에는 이 경쟁 구도가 투자 판단에
    주는 의미를 2~3문장으로 해설합니다. 목록 내용을 그대로 반복하지 않습니다.
    해설 문장에는 sources_for_citation 중 used_by에 competitor가 있는 출처 번호를 붙입니다.
11. 장과 장 사이에 '---' 같은 구분선을 쓰지 않습니다.
12. 자료에 붙은 '(2026-04-24 기준)' 같은 날짜 표기는 지우지 말고 그대로 옮깁니다.
    '올해', '내년', '연내' 같은 상대 시점 표현은 쓰지 않습니다.
13. 경쟁사가 대기업·상장사인지 스타트업인지는 competitor_analysis의 tier만 따릅니다.
    ('선도 기업' = 대기업 또는 상장사, '동급 기업' = 비상장 스타트업)
    sources_for_citation의 제목은 인용 번호를 고르는 용도일 뿐이며, 제목의 표현('신진기업' 등)으로 기업을 규정하지 않습니다.
14. 평가 기준·보류 조건을 설명할 때는 [분석 자료]의 evaluation_rules에 적힌 그대로 씁니다. 기준을 추측해 바꿔 쓰지 않습니다.
15. '한계점' 장에는 평가 대상 기업에 대한 조언(연구개발 강화, 자금 조달 계획 등)을 쓰지 않습니다.
    이 평가 자체의 한계(공개 정보 부족으로 점수가 제한된 항목, evaluation_rules의 평가 방식에서 오는 한계,
    웹 검색·문서 범위의 한계)와 평가 방법의 개선 방향을 씁니다.
16. 기술 요약의 기존 7개 항목은 기업 웹 사실이고 industry_context는 보고서 기반 업계 해석입니다.
    기업 사실은 웹 출처를, 업계 해석은 연결된 기업 웹 출처와 보고서 출처를 함께 인용합니다.
    conditions_missing은 비교 조건이 부족하다는 뜻이므로 기술 우위·업계 순위로 표현하지 않습니다.
    업계 보고서의 수치를 기업의 양산·매출·계약 실적으로 바꾸지 않습니다.
17. 1~4장은 평가 관점별 장입니다. 장 제목의 관점 이름(기술 성숙도, 시장성, 이해관계자, 도메인 적용)을 그대로 씁니다.
    '이해관계자' 장은 창업자·핵심 인력(team_analysis)과 투자사·투자 조건(company의 investors, 투자 금액·기업가치)을,
    '도메인 적용' 장은 고객·PoC·공급 계약 등 실적(tech_summary의 public_revenue_contracts)과 적용 분야(세부 분야)를 다룹니다.
    자료가 없으면 그 장에 '정보 부족'이라고 명시하고 없는 내용을 채우지 않습니다.
    '시장성' 장은 세부 분야 전체 수치(sub_market=false)를 먼저 쓰고, sub_market=true 수치(PC·스마트폰 등 하위 시장)는
    '하위 시장 참고 수치'라고 구분해 씁니다. 같은 범위의 수치가 출처마다 다르면 둘 다 출처 번호와 함께 씁니다.
    시장 수치의 연도는 figures의 year만 씁니다. year가 비어 있으면(year_note: 연도 미상) 연도를 지어내지 말고 '연도 미상'이라고 씁니다.
    forecast=true 수치는 '~로 전망된다'처럼 전망으로 쓰고, '성장하고 있다' 같은 현재형으로 쓰지 않습니다.
18. '최적화된 성능', '혁신적인', '획기적인', '탁월한' 같은 회사 홍보 표현을 옮기지 말고, 수치·사실로만 서술합니다.
19. '사업 리스크' 장의 리스크마다 근거 출처 번호를 붙입니다. (market_risks, competitive_risks, 기술 약점 등 자료에 있는 리스크)
    scores.risks는 sources에 적힌 출처의 번호를 붙이고, 리스크를 다루지 않는 출처 번호를 붙이지 않습니다.
    유리한 근거만 고르지 말고, 자료에 있는 약점·리스크·반대 근거도 해당 장에 씁니다.
20. [분석 자료]에 halted가 있으면 평가가 중간에 멈춘 것이므로 '한계점' 장 첫 문장에 그 사유를 그대로 씁니다.
21. 같은 사실(양산·테이프아웃 시점, 성능·투자 수치 등)은 모든 장에서 같은 값·시점으로 씁니다.
    자료에 시점이 여러 개면 가장 최근 자료의 내용을 쓰고, 계획·목표는 '목표'라고 밝힙니다.
    제조·양산·테이프아웃·실증은 자료에 완료로 확인된 경우에만 완료형이나 현재형('~로 제조되며')으로 씁니다.
    확인되지 않았으면 '~로 설계 중', '~ 계획'으로 쓰고, MOU·협약 체결을 실증 완료로 쓰지 않습니다.
22. '경쟁 구도' 장은 표 자리 아래에 '**확인된 사실**', '**비교 결과**', '**투자 관점 해석**' 소제목 줄로 세 부분을 나눠 씁니다.
    - 확인된 사실: competitor_analysis의 claims·differentiation에 있는 수치·사실만 출처 번호와 함께 씁니다.
      speaker가 대상 기업·경쟁사면 '~가 발표했다'처럼 발표 주체를 밝힙니다.
    - 비교 결과: comparable=true인 수치만 우열을 씁니다. 같은 조건의 비교 자료가 없으면 '같은 조건의 비교 자료가 없어
      우열을 판단할 수 없다'고 쓰고 not_comparable 항목을 씁니다.
    - 투자 관점 해석: scores의 경쟁 우위 점수와 근거(기준표 상한 포함)에 맞춰 1~2문장으로 씁니다. 경쟁사 분석 결과를
      긍정적인 투자 의견으로 바꾸지 않고, 경쟁사가 앞선 점도 함께 씁니다.
    자료에 없는 평가('충분히 경쟁할 수 있다', '가장 위협적', '최초가 될 가능성')를 덧붙이지 않습니다.
23. 약어(HBM, CXL 등)는 [분석 자료]에 풀어 쓴 표기가 있을 때만 풀어 쓰고, 없으면 약어 그대로 씁니다.
"""

# 투자 추천 기업이 있을 때의 목차
TOC_RECOMMEND = """## SUMMARY (위 5번 틀 그대로)
## 1. 사업 아이디어·기술 성숙도 (핵심 칩, 공정, 개발 단계, 성능)
## 2. 시장성 (세부 분야 시장 규모·성장률, 수요 요인)
## 3. 이해관계자 (창업자·핵심 인력, 투자사·투자 조건)
## 4. 도메인 적용 (고객·PoC·공급 계약 등 실적, 적용 분야)
## 5. 경쟁 구도 (자리 {{COMPETITOR_TABLE}} + 위 22번의 세 부분: 확인된 사실 / 비교 결과 / 투자 관점 해석)
## 6. 투자 판단 결과 (점수표 자리 {{SCORE_TABLE}} + 항목별 근거 해설)
## 7. 사업 리스크 (시장·기술·규제·경쟁, 리스크마다 출처 번호)
## 8. 한계점 ('정보 부족' 항목, '확인 필요' 수치, 보류·제외 기업과 사유)"""

# 모든 후보가 보류/제외되었을 때의 목차
TOC_NO_RECOMMEND = """## SUMMARY
## 1. 평가 개요 (평가한 후보와 evaluation_rules의 평가 조건)
## 2. 후보별 평가 결과 (표 자리 {{CANDIDATE_TABLE}} 한 줄 + 보류 사유의 공통점 1~2문장. 후보별 목록은 따로 쓰지 않음)
## 3. 사업 리스크 (보류 사유에서 드러난 리스크만. 자료에 없는 일반론은 쓰지 않음)
## 4. 한계점 및 향후 개선 방향 (위 15번 규칙)"""

# 사용자 프롬프트 틀: {toc}, {mode}, {materials}, {fix_notes} 자리에 값이 채워진다
USER_TEMPLATE = """[보고서 유형] {mode}

[목차]
{toc}

[분석 자료 - JSON]
{materials}

{fix_notes}
위 자료만 근거로 보고서를 작성하세요."""

# 품질 평가가 미달로 되돌려 보냈을 때 붙는 추가 지시 (agents/quality.py의 fix_instructions)
QUALITY_FIX_TEMPLATE = """[품질 수정 요청] 직전 보고서가 품질 평가에서 아래 항목이 미달이었습니다.
사실을 새로 만들지 말고 [분석 자료] 안에서 고치세요. 근거가 없으면 '정보 부족'으로 쓰세요.
{items}
"""

# 사실 검증에서 불일치가 나와 '재작성'할 때만 붙는 추가 지시
FIX_NOTES_TEMPLATE = """[수정 요청] 직전 보고서에서 아래 항목이 [분석 자료]와 일치하지 않았습니다.
자료에 있는 값으로 고치거나, 자료에서 확인되지 않으면 그 내용을 빼고 '정보 부족'으로 쓰세요.
{items}
"""

# 과제 조건(분량·순서·SUMMARY 틀) 위반으로 '형식만 고쳐 쓸' 때 붙는 추가 지시
# 재작성 때 지적된 장만 고치게 하는 지시. 나머지 장은 코드가 직전 보고서 내용으로 되돌린다
PARTIAL_REWRITE_TEMPLATE = """[부분 재작성] 아래 [직전 보고서]에서 다음 장만 고치세요: {chapters}
나머지 장은 [직전 보고서]의 내용을 글자 그대로(출처 번호 포함) 옮기세요. 표 자리 표시({{{{SCORE_TABLE}}}} 등)도 그대로 둡니다.

[직전 보고서]
{previous}
"""

FORMAT_FIX_TEMPLATE = """[형식 수정 요청] 직전 보고서가 아래 과제 조건을 지키지 않았습니다. 내용은 유지하고 형식만 고치세요.
{items}
"""


# ═════════════════════════════════════════════════════════════
# 설정값(상수)
# ═════════════════════════════════════════════════════════════
LLM_MODEL = MODEL_ANALYZE    # 팀 설정(config.py)을 따른다
LLM_TEMPERATURE = 0.0          # 0 = 매번 거의 같은 답(재현성). 사실 기반 글이라 창의성 불필요
MAX_FORMAT_FIXES = 1           # 과제 조건 위반 시 형식만 고쳐 쓰는 횟수(사실 검증 재작성과 별개)

# 평가표(설계서 3장) — 바뀌면 여기만 고친다. 순서는 보고서 표의 표시 순서이기도 하다
# 채점 항목 키는 config.WEIGHTS(영문)를 따르고, 보고서에는 아래 한글 이름으로 표시한다
ITEM_LABELS = {
    "tech": "기술·제품 성숙도",
    "market": "시장성",
    "team": "창업자·팀",
    "traction": "실적·고객 검증",
    "competition": "경쟁 우위",
    "deal": "투자조건",
}
WEIGHTS = {ITEM_LABELS[k]: w for k, w in TEAM_WEIGHTS.items()}  # 한글 이름 → 비중
SCORE_MAX = 5                    # 항목 점수는 1~5점
DECISION_THRESHOLD = INVEST_THRESHOLD  # 설계서 3.4: 환산 점수 70점 이상이면 투자 (config.py)
DECISION_INVEST = "투자"         # 투자 판단 에이전트가 쓰는 결정 값
INSUFFICIENT_MAX_SCORE = 2       # 설계서 3.5: 정보 부족 항목은 최대 2점
INSUFFICIENT_LABEL = "정보 부족"  # 근거 문장에 이 말이 있으면 정보 부족으로 본다
COMPANY_NAME_KEY = "name"        # State의 company 안 기업명 키 (팀원 State와 다르면 여기만 수정)

# 분석 결과가 담긴 State 키들 (설계 문서의 State 표 기준)
ANALYSIS_KEYS = (
    "company", "tech_summary", "market_analysis",
    "competitor_analysis", "team_analysis", "scores",
)

# 과제 조건 점검
SUMMARY_MAX_CHARS = 900     # A4 반 페이지 ≈ 한글 900자 (대략치. 조에서 조정 가능)
PAGE_CHARS = 1800           # A4 한 페이지 ≈ 한글 1,800자 (대략치)
MAX_PAGES = 10              # 과제 조건: 최대 10장 (Agent 과제)
MIN_PAGES = 7               # 조 기준: 최소 7장 (PDF 조판 간격을 키워 맞춘다, 내용은 늘리지 않음)
SUMMARY_BULLETS = 3         # SUMMARY 틀: 핵심 근거·핵심 리스크 각 3개
CONFIRM_BULLETS = (1, 2)    # SUMMARY 틀: 확인 필요 1~2개
SUMMARY_LABELS = ("**결론:**", "**핵심 근거**", "**핵심 리스크**", "**확인 필요**")

# 출처 표기 (설계서 5.2) — 종류 → (화면 이름, 필요한 필드, 표기 틀)
SOURCE_FORMATS = {
    "report": ("기관 보고서", ("publisher", "year", "title"),
               "{publisher}({year}). {title}. {url}"),
    "paper": ("학술 논문", ("authors", "year", "title", "journal", "volume_issue", "pages"),
              "{authors}({year}). {title}. {journal}, {volume_issue}, {pages}."),
    "web": ("웹페이지", ("author", "date", "title", "site", "url"),
            "{author}({date}). {title}. {site}, {url}"),
}
OTHER_LABEL = "기타"
UNKNOWN_DATE = "날짜 미상"
REFERENCE_HEADING = "## REFERENCE"

PLACEHOLDER = "{{SCORE_TABLE}}"                      # AI가 쓰는 '표 자리' 표시
COMPETITOR_PLACEHOLDER = "{{COMPETITOR_TABLE}}"      # 5장 경쟁사 표 자리
CANDIDATE_PLACEHOLDER = "{{CANDIDATE_TABLE}}"        # '추천 기업 없음' 보고서 2장 후보 표 자리
SECTION1_PATTERN = re.compile(r"^##\s*1\.[^\n]*\n", re.MULTILINE)   # 1장 기술 성숙도
SECTION2_PATTERN = re.compile(r"^##\s*2\.[^\n]*\n", re.MULTILINE)
COMPETITOR_SECTION_PATTERN = re.compile(r"^##\s*5\.[^\n]*\n", re.MULTILINE)   # 5장 경쟁 구도
SCORE_SECTION_PATTERN = re.compile(r"^##\s*6\.[^\n]*\n", re.MULTILINE)        # 6장 투자 판단 결과
HR_PATTERN = re.compile(r"^[ \t]*(?:[-*_][ \t]*){3,}$\n?", re.MULTILINE)  # AI가 넣는 '---' 구분선
# AI가 묶어 쓴 [6,7]·[8-11]. 세 자리 이하만 (대괄호 안 연도 범위 [2024-2030]은 인용이 아니다)
GROUPED_CITATION_PATTERN = re.compile(r"\[(\d{1,3}(?:\s*[,~\-]\s*\d{1,3})+)\]")
LEADING_TIER = "선도 기업"                            # 경쟁 구도에서 왼쪽(진입 위협)에 놓을 구분
PEER_TIER = "동급 기업"                               # 오른쪽(직접 경쟁)에 놓을 구분
TARGET_TIER = "평가 대상"
LANDSCAPE_HEADER = "구분"                             # 경쟁 구도 표의 첫 칸 제목 (PDF는 이 표를 구도 그림으로 그린다)
STRENGTH_HEADER, RISK_HEADER = "차별점(강점)", "경쟁 리스크"   # 2칸 패널로 그리는 표의 제목
# 코드가 State로 만드는 보조 표의 첫 칸 제목 (PDF 열 폭 TABLE_FRACS의 키이기도 하다)
TIMELINE_HEADER = "보도일"                            # 1장 개발 단계 타임라인
RUBRIC_POSITION_HEADER = "채점 항목"                  # 6장 기준표 대비 위치
SCORE_BASIS_HEADER = "판단 항목"                      # 6장 항목별 판단 근거
COMPARISON_HEADER = "경쟁사"                          # 5장 경쟁 구도 그림 아래 비교 근거 표 (PDF)
TIMELINE_CAPTION = "**개발 단계 보도 시점** (기사 게시일 순, 사건 시점과 다를 수 있음)"
RUBRIC_POSITION_CAPTION = "**기준표 대비 위치** (현재 점수 구간과 다음 점수 조건)"
SCORE_BASIS_CAPTION = "**항목별 판단 근거**"
MIN_TIMELINE_ROWS = 2        # 1행이면 시점 비교가 안 되어 타임라인을 넣지 않는다
MAX_TABLE_CITATIONS = 4      # 표 출처 칸에 보일 최대 번호 수 (많으면 칸이 번호로 가득 찬다)
REFERENCE_SECTION_PATTERN = re.compile(r"\n?##\s*REFERENCE.*", re.DOTALL | re.IGNORECASE)
H2_PATTERN = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
TABLE_UNAVAILABLE = "(채점 항목이 누락되어 점수표를 만들 수 없습니다.)"


# ═════════════════════════════════════════════════════════════
# 1) 평가표 계산
# ═════════════════════════════════════════════════════════════
def score_items(scores: dict) -> dict:
    """scores["items"]({영문 키: {score, evidence, ...}})를 보고서용 한글 키로 바꿔 돌려준다."""
    items = scores.get("items") or {}
    return {label: items[key] for key, label in ITEM_LABELS.items() if key in items}


def item_contributions(scores: dict) -> dict[str, float] | None:
    """항목별 '기여 점수' = 점수 ÷ 5 × 비중. 채점 항목이 하나라도 없으면 None."""
    items = score_items(scores)
    if not items:
        return None
    contributions = {}
    for name, weight in WEIGHTS.items():
        try:
            contributions[name] = items[name]["score"] / SCORE_MAX * weight
        except KeyError:
            logger.warning("[채점] 채점 항목 누락: %s", name)
            return None
    return contributions


def recompute_total(scores: dict) -> float | None:
    """항목 점수로 환산 점수를 직접 다시 계산한다. Σ(점수 ÷ 5 × 비중)"""
    contributions = item_contributions(scores)
    return None if contributions is None else round(sum(contributions.values()), 1)


def find_insufficient_items(scores: dict) -> list[str]:
    """'정보 부족'으로 표시된 채점 항목 이름을 찾는다.

    판단 기준(둘 중 하나): items[항목]["insufficient"]가 True이거나, 근거 문장에 '정보 부족'이 있음.
    """
    found = []
    for name, info in score_items(scores).items():
        if info.get("insufficient") or INSUFFICIENT_LABEL in str(info.get("evidence", "")):
            found.append(name)
    return found


# ═════════════════════════════════════════════════════════════
# 2) 점수표 만들기 — 숫자는 코드가
# ═════════════════════════════════════════════════════════════
def build_score_table(scores: dict) -> str:
    """항목별 점수·비중·기여 점수 표와 결정 한 줄을 마크다운으로 만든다."""
    contributions = item_contributions(scores)
    if contributions is None:
        return TABLE_UNAVAILABLE

    total = recompute_total(scores)
    lines = ["| 항목 | 점수 | 비중 | 기여 점수 |", "|---|---|---|---|"]
    for name, weight in WEIGHTS.items():
        score = score_items(scores)[name]["score"]
        lines.append(f"| {name} | {score}/{SCORE_MAX} | {weight}% | {contributions[name]:.1f} |")
    lines.append(f"| **환산 점수** | | 100% | **{total:.1f}** |")
    lines.append("")
    lines.append(f"결정: **{scores.get('decision', '미정')}** "
                 f"(환산 점수 {total:.1f}점, 투자 기준 {DECISION_THRESHOLD}점 이상)")
    lines += [""] + build_decision_checks(scores)
    return "\n".join(lines)


def cautions_of(scores: dict) -> list[dict]:
    """투자 판단이 남긴 주의 목록 [{item, label, score, insufficient, comment}]. 없거나 형식이 다르면 빈 목록.

    핵심 항목(기술·팀)이 낮아도 보류가 아닐 때 독자에게 알리려는 것. insufficient가 True면 정보 부족으로 제한된 점수,
    False면 실제로 낮게 평가한 점수다.
    """
    found = scores.get("cautions")
    return [c for c in found if isinstance(c, dict) and c.get("comment")] if isinstance(found, list) else []


def caution_summary_hint(scores: dict) -> str:
    """SUMMARY '확인 필요'에 쓸 주의 문구: 항목명·점수와 확인할 다음 점수 조건. 주의가 없으면 빈 문자열.
    왜: '기술·제품 성숙도 2점 확인 필요'처럼 점수만 되풀이해 무엇을 확인할지 알 수 없었다."""
    levels = rubric_levels()
    hints = []
    for c in cautions_of(scores):
        label, score = c.get("label", c.get("item", "")), c.get("score", 0)
        nxt = levels.get(c.get("item", ""), [])[score] if isinstance(score, int) and 0 < score < 5 else ""
        reason = "정보 부족으로 제한" if c.get("insufficient") else "낮은 평가"
        hints.append(f"{label} {score}점({reason}): {score + 1}점 조건인 '{nxt}' 충족 여부 확인 필요" if nxt
                     else f"{label} {score}점({reason}) 근거 확인 필요")
    return ", ".join(hints)


def build_decision_checks(scores: dict) -> list[str]:
    """설계서 3.4 보류 조건을 하나씩 점검한 결과. 점수만으로는 안 보이는 '왜 투자(보류)인가'를 보여 준다.

    글머리표(-)를 쓰지 않는다: 바로 뒤에 오는 AI의 항목별 해설 목록과 섞여 보이지 않게 하려는 것.
    """
    items = score_items(scores)
    core = [(ITEM_LABELS[k], items[ITEM_LABELS[k]]["score"]) for k in CORE_ITEMS if ITEM_LABELS[k] in items]
    lines = []
    if core:
        shown = ", ".join(f"{name} {score}점" for name, score in core)
        ok = all(score >= CORE_MIN_SCORE for _, score in core)
        lines.append(f"보류 조건 ① 핵심 역량 {CORE_MIN_SCORE}점 미만: {shown} → **{'해당 없음' if ok else '해당'}**")
        # 2점처럼 보류는 아니지만 낮은 핵심 항목은 투자 판단이 남긴 주의 코멘트를 옮긴다 (주의: 보류 조건은 아님)
        lines += [f"   ⚠ 주의: {c['comment']}" for c in cautions_of(scores)]
    legal = scores.get("legal_risk")
    if isinstance(legal, dict):
        verdict = "미해소 → 해당" if legal.get("unresolved") else "해당 없음"
        evidence = str(legal.get("evidence", "")).replace("\n", " ").strip()
        lines.append(f"보류 조건 ② 해소되지 않은 법률 리스크: **{verdict}**" + (f" ({evidence})" if evidence else ""))
    return lines


# 분석 에이전트가 자기 검색 결과 목록 기준으로 붙인 번호 ([3], [2,5] 등). 보고서 REFERENCE 번호와 다르다
AGENT_REF_PATTERN = re.compile(r"\s*\[\d{1,3}(?:\s*[,~\-]\s*\d{1,3})*\]")
# 보고서 LLM이 출처 번호 대신 자료 이름을 쓴 경우 ([team_analysis] 등)
KEY_CITATION_PATTERN = re.compile(r"\[[a-z][a-z_]*\]")
# 자료 이름만 적힌 글머리표 줄 ('- summary_caution 없음' 등)
KEY_NAME_LINE_PATTERN = re.compile(r"^[ \t]*[-*][ \t]*[a-z]+_[a-z_]*[ \t]*(?:없음)?[ \t]*\n?", re.MULTILINE)


# 기술 요약 필드명 → 보고서 표기. LLM이 "'weaknesses' 항목"처럼 필드명을 그대로 쓰는 경우를 고친다
FIELD_LABELS = {"core_chips": "핵심 칩", "development_stage": "개발 단계", "performance_metrics": "성능 수치",
                "public_revenue_contracts": "공개 매출·계약", "weaknesses": "약점", "strengths": "강점",
                # 분석 자료 키 이름 (quality.INTERNAL_NAMES와 같은 목록. judge 근거 문장이 그대로 쓰는 경우가 있다)
                "tech_summary": "기술 요약", "market_analysis": "시장성 분석", "competitor_analysis": "경쟁사 분석",
                "team_analysis": "팀 분석", "summary_caution": "주의 항목", "info_insufficient": "정보 부족",
                "industry_context": "업계 해석", "sub_market": "하위 시장", "segment_specific": "세부 분야 수치",
                "sources_for_citation": "출처 목록", "evaluation_rules": "평가 기준",
                "missing_fields": "누락 항목", "field_status": "항목 상태"}
# 밑줄이 있는 필드명은 어디서나, 일반 영어 단어이기도 한 weaknesses·strengths는 따옴표 안이거나 바로 뒤에
# 한글이 올 때만 필드명으로 본다 (경쟁사 표의 영문 설명 속 단어는 그대로 둔다)
FIELD_NAME_PATTERN = re.compile(
    r"(?<![A-Za-z_])(" + "|".join(k for k in FIELD_LABELS if "_" in k) + r")(?![A-Za-z_])"
    r"|(?<![A-Za-z_])(weaknesses|strengths)(?=['’\"]|\s*[가-힣])")
# judge 근거의 '(segment_specific=true, sub_market=false)' 같은 자료 설정값 표기
FIELD_FLAG_PATTERN = re.compile(r"\s*\(?\s*[a-z]+_[a-z_]*\s*=\s*(?:true|false)(?:\s*,\s*[a-z]+_[a-z_]*\s*=\s*(?:true|false))*\s*\)?",
                                re.IGNORECASE)


# 이름을 바꾼 뒤 남는 'info_insufficient: true'의 ': true' (Judge 사유가 자료를 JSON 모양 그대로 인용한 경우)
FIELD_VALUE_PATTERN = re.compile(r"\s*:\s*(?:true|false)(?![A-Za-z])", re.IGNORECASE)   # 뒤에 한글이 오면 \b가 안 맞는다


def translate_field_names(body: str) -> str:
    body = FIELD_FLAG_PATTERN.sub("", body)
    body = FIELD_NAME_PATTERN.sub(lambda m: FIELD_LABELS[m.group(1) or m.group(2)], body)
    return FIELD_VALUE_PATTERN.sub("", body)


def strip_agent_refs(value):
    """문자열 안의 에이전트 검색 결과 번호를 지운다 (dict·list는 안쪽까지)."""
    if isinstance(value, str):
        return AGENT_REF_PATTERN.sub("", value)
    if isinstance(value, list):
        return [strip_agent_refs(v) for v in value]
    if isinstance(value, dict):
        return {k: strip_agent_refs(v) for k, v in value.items()}
    return value


def _cell(text) -> str:
    """표 칸에 넣을 글. 줄바꿈과 '|'는 표를 깨뜨리므로 바꾼다. 에이전트 검색 결과 번호는 지운다
    (그대로 두면 보고서 번호로 다시 매겨져 다른 출처를 가리키거나 '[출처 확인 필요]'가 된다)."""
    text = strip_agent_refs(str(text)) if text not in (None, "") else ""
    return (text or "–").replace("|", "/").replace("\n", " ").strip()


EXCLUDED_PREFIX = "다른 세부 분야라 비교에서 제외"
NOT_COMPARABLE_PREFIX = "직접 비교할 수 없는 항목"


def _claims_cell(claims: list, sources: list[dict], target: str, owner: str = "") -> str:
    """경쟁사 분석의 주장들을 표 칸 하나로: 문장(발표 주체) + 보고서 출처 번호. 예전 결과(문자열)는 그대로."""
    parts = []
    for c in claims:
        if isinstance(c, dict):
            cite = _citations_for_urls(sources, [c.get("url", "")]) if c.get("url") else ""
            parts.append(f"{_cell(claim_text(c, target, owner))} {cite}".strip())
        else:
            parts.append(_cell(c))
    return " ".join(parts)


def build_competitor_table(state: dict, sources: list[dict] | None = None) -> str:
    """5장 경쟁 구도: ① 구도 표(평가 대상·선도 기업·동급 기업) ② 차별점·경쟁 리스크 표. 값은 competitor_analysis 그대로.

    마크다운에서는 표로 읽히고, PDF에서는 첫 칸 제목('구분'/'차별점')을 보고 구도 그림과 2칸 패널로 그린다.
    비교·차별점은 경쟁사 분석이 확인한 주장만, 발표 주체와 출처 번호를 붙여 쓴다.
    """
    analysis = state.get("competitor_analysis") or {}
    company = state.get("company") or {}
    sources = sources or []
    target = company.get(COMPANY_NAME_KEY) or ""
    competitors = sorted(analysis.get("competitors") or [], key=lambda c: c.get("tier") != LEADING_TIER)
    if not competitors:
        return "(같은 세부 분야에서 비교할 경쟁사를 찾지 못했습니다.)"
    country = "한국" if company.get("is_domestic") else ""
    lines = [f"| {LANDSCAPE_HEADER} | 기업 | 국가 | 제품 | 소개 | 비교 근거 |", "|---|---|---|---|---|---|",
             f"| {TARGET_TIER} | {_cell(target)} | {_cell(country)} | {_cell(company.get('product'))} | – | – |"]
    for c in competitors:
        claims = c.get("claims")
        compared = (_claims_cell(claims, sources, target, c.get("name", "")) if claims
                    else NO_COMPARISON if claims is not None else _cell(c.get("comparison")))
        lines.append(f"| {_cell(c.get('tier'))} | {_cell(c.get('name'))} | {_cell(c.get('country'))} "
                     f"| {_cell(c.get('product'))} | {_cell(c.get('summary'))} | {compared} |")

    strengths, risks = analysis.get("differentiation") or [], analysis.get("competitive_risks") or []
    if strengths or risks:
        lines += ["", f"| {STRENGTH_HEADER} | {RISK_HEADER} |", "|---|---|"]
        for i in range(max(len(strengths), len(risks))):
            left = _claims_cell([strengths[i]], sources, target) if i < len(strengths) else ""
            right = _cell(risks[i]) if i < len(risks) else ""
            lines.append(f"| {left} | {right} |")
    for prefix, items in ((NOT_COMPARABLE_PREFIX, analysis.get("not_comparable")),
                          (EXCLUDED_PREFIX, analysis.get("excluded_other_segment"))):
        if items:
            lines += ["", f"{prefix}: {'; '.join(_cell(x) for x in items)}"]
    return "\n".join(lines)


def build_candidate_table(state: dict) -> str | None:
    """'추천 기업 없음' 보고서 2장의 후보 표: 평가 순서대로 결과·환산 점수·사유. 후보 정보가 없으면 None.

    (투자 추천 보고서에는 넣지 않는다: 첫 투자 결정에서 평가를 멈춰 뒤 후보는 점수가 없다)
    """
    if not state.get("candidates"):
        return None
    reasons = {r.get("company"): r.get("reason") for r in state.get("rejected") or [] if isinstance(r, dict)}
    lines = ["| 순서 | 후보 | 결과 | 환산 점수 | 사유 |", "|---|---|---|---|---|"]
    for i, (name, total, status) in enumerate(candidate_rows(state), 1):
        score = f"{total:.1f}" if total is not None else "–"
        lines.append(f"| {i} | {_cell(name)} | {status} | {score} | {_cell(reasons.get(name))} |")
    return "\n".join(lines)


def insert_score_table(body: str, scores: dict) -> str:
    """본문의 자리표시자를 점수표로 바꾼다. 자리표시자가 없으면 6장 제목 아래에 넣는다."""
    table = build_score_table(scores)
    if PLACEHOLDER in body:
        return body.replace(PLACEHOLDER, table)
    match = SCORE_SECTION_PATTERN.search(body)
    if match:
        logger.warning("[표 생성] 자리표시자가 없어 6장 제목 아래에 표를 삽입")
        return body[: match.end()] + "\n" + table + "\n" + body[match.end():]
    logger.warning("[표 생성] 삽입 위치를 찾지 못해 본문 끝에 표를 추가")
    return body + "\n\n" + table


def _insert_table(body: str, placeholder: str, table: str, section: re.Pattern, name: str) -> str:
    """자리표시자를 표로 바꾼다. 자리표시자가 없으면 해당 장 제목 바로 아래에 넣고, 장도 없으면 넣지 않는다."""
    if placeholder in body:
        return body.replace(placeholder, table)
    match = section.search(body)
    if match:
        logger.warning("[표 생성] %s 자리표시자가 없어 장 제목 아래에 삽입", name)
        return body[: match.end()] + "\n" + table + "\n" + body[match.end():]
    logger.warning("[표 생성] %s를 넣을 장을 찾지 못해 넣지 않음", name)
    return body


def insert_competitor_table(body: str, state: dict, sources: list[dict] | None = None) -> str:
    """5장의 자리표시자를 경쟁사 표로 바꾼다. 자리표시자가 없으면 5장 제목 아래에 넣는다.
    (번호만 나열한 출처 줄은 사실 검증이 주장 없는 문장으로 잡아, 출처 번호는 주장마다 붙인다)"""
    return _insert_table(body, COMPETITOR_PLACEHOLDER, build_competitor_table(state, sources),
                         COMPETITOR_SECTION_PATTERN, "경쟁사 표")


def _citations_for_urls(sources: list[dict], urls: list[str]) -> str:
    """url들에 해당하는 collect_sources 번호를 '[n][m]'으로. 못 찾은 url은 건너뛴다 (표의 출처 칸)."""
    by_url = {}
    for s in sources:
        for raw in s.get("raw", []):
            if raw.get("url"):
                by_url.setdefault(raw["url"], s["n"])
    numbers = list(dict.fromkeys(by_url[u] for u in urls if u in by_url))[:MAX_TABLE_CITATIONS]
    return " ".join(f"[{n}]" for n in numbers)   # 띄어 써야 PDF 좁은 칸에서 번호 중간이 아니라 번호 사이에서 줄이 바뀐다


def _source_cell(sources: list[dict], urls: list[str]) -> str:
    """표 출처 칸. 보고서 출처 번호라서 _cell을 거치지 않는다 (_cell은 [n]을 에이전트 검색 번호로 보고 지운다)."""
    return _citations_for_urls(sources, urls) or "–"


def build_timeline_table(state: dict, sources: list[dict]) -> str | None:
    """1장 개발 단계 타임라인: tech_summary 근거 중 개발 단계 주장을 기사 게시일 순으로. 없으면 None.

    왜: 시점이 다른 기사의 개발 단계 문장이 본문 한 문단에 섞이면 양산 시점이 서로 모순돼 보인다.
    날짜는 사건 시점이 아니라 기사 게시일이라 제목을 '보도일'로 둔다.
    """
    evidence = (state.get("tech_summary") or {}).get("evidence") or []
    rows, seen = [], set()
    for e in evidence:
        if isinstance(e, dict) and e.get("field") == "development_stage" and e.get("text") and e["text"] not in seen:
            seen.add(e["text"])   # 같은 문장이 여러 출처로 기록되면 처음 것만
            rows.append(e)
    if len(rows) < MIN_TIMELINE_ROWS:
        return None
    rows.sort(key=lambda e: (not e.get("date"), e.get("date") or ""))   # 날짜 미상은 맨 뒤
    lines = [TIMELINE_CAPTION, "", f"| {TIMELINE_HEADER} | 내용 | 출처 |", "|---|---|---|"]
    lines += [f"| {_cell(e.get('date') or UNKNOWN_DATE)} | {_cell(translate_field_names(e['text']))} "
              f"| {_source_cell(sources, [e.get('url', '')])} |" for e in rows]
    return "\n".join(lines)


def rubric_levels() -> dict[str, list[str]]:
    """judge.RUBRIC 표를 {영문 항목 키: [1점 문구, …, 5점 문구]}로 읽는다. 기준표 원문은 judge.py 한곳에만 둔다."""
    levels = {}
    for line in RUBRIC.splitlines()[2:]:   # 제목 줄과 구분선 다음부터
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        levels[cells[0].split()[0]] = cells[1:]
    return levels


def build_score_basis_tables(scores: dict, sources: list[dict]) -> str | None:
    """6장 점수표 아래: ① 기준표 대비 위치(현재 구간·다음 점수 조건) ② 항목별 판단 근거·출처. 채점 항목이 없으면 None.

    왜: 점수표만으로는 '왜 5점이 아니라 4점인가'가 안 보인다. 기준표 문구로 보여 주면 결정을 점수·기준으로만 설명한다.
    한 표에 6칸을 넣으면 A4에서 근거 칸이 너무 좁아 두 표로 나눈다.
    """
    items, levels = scores.get("items") or {}, rubric_levels()
    position, basis = [], []
    for key, label in ITEM_LABELS.items():
        info = items.get(key)
        score = info.get("score") if isinstance(info, dict) else None
        if not isinstance(score, (int, float)) or not 1 <= score <= SCORE_MAX or key not in levels:
            continue
        score = int(score)
        upper = levels[key][score] if score < SCORE_MAX else "–"
        position.append(f"| {label} | {score}/{SCORE_MAX} | {_cell(levels[key][score - 1])} | {_cell(upper)} |")
        evidence = translate_field_names(str(info.get("evidence") or ""))   # 표는 본문 이름 변환 뒤에 들어간다
        if info.get("insufficient"):
            evidence = f"(정보 부족으로 최대 {INSUFFICIENT_MAX_SCORE}점) {evidence}"
        urls = [s.get("url", "") for s in info.get("sources") or [] if isinstance(s, dict)]
        basis.append(f"| {label} | {_cell(evidence)} | {_source_cell(sources, urls)} |")
    if not position:
        return None
    return "\n".join([RUBRIC_POSITION_CAPTION, "",
                      f"| {RUBRIC_POSITION_HEADER} | 점수 | 해당 기준 | 다음 점수 조건 |", "|---|---|---|---|",
                      *position, "", SCORE_BASIS_CAPTION, "",
                      f"| {SCORE_BASIS_HEADER} | 판단 근거 | 출처 |", "|---|---|---|", *basis])


def _append_to_section(body: str, section: re.Pattern, block: str, name: str) -> str:
    """장 끝(다음 '## ' 제목 바로 앞)에 block을 넣는다. 장을 못 찾으면 넣지 않는다."""
    match = section.search(body)
    if not match:
        logger.warning("[표 생성] %s를 넣을 장을 찾지 못해 넣지 않음", name)
        return body
    nxt = H2_PATTERN.search(body, match.end())
    end = nxt.start() if nxt else len(body)
    return body[:end].rstrip() + "\n\n" + block + "\n\n" + body[end:]


def insert_timeline_table(body: str, state: dict, sources: list[dict]) -> str:
    table = build_timeline_table(state, sources)
    return _append_to_section(body, SECTION1_PATTERN, table, "개발 단계 타임라인") if table else body


def insert_score_basis_tables(body: str, scores: dict, sources: list[dict]) -> str:
    """점수표(insert_score_table가 넣은 것) 바로 아래에 넣는다. 점수표가 없으면 넣지 않는다."""
    tables = build_score_basis_tables(scores, sources)
    score_table = build_score_table(scores)
    if not tables or score_table == TABLE_UNAVAILABLE or score_table not in body:
        if tables:
            logger.warning("[표 생성] 점수표를 찾지 못해 채점 근거 표를 넣지 않음")
        return body
    return body.replace(score_table, score_table + "\n\n" + tables, 1)


# ═════════════════════════════════════════════════════════════
# 3) 출처(REFERENCE) 만들기 — 출처는 코드가
# ═════════════════════════════════════════════════════════════
def format_source(src: dict) -> tuple[str, str]:
    """출처 하나를 (종류 이름, 표기 문자열)로 바꾼다. 필드가 부족하면 대체 표기를 쓴다."""
    kind = src.get("kind")  # state.Source: "web" | "report"
    site = src.get("site") or urlparse(src.get("url", "")).netloc.removeprefix("www.")
    src = {**src, "year": src.get("year") or (src.get("date") or "")[:4],
           "author": src.get("author") or src.get("publisher") or site,  # 언론사를 모르면 도메인
           "date": src.get("date") or (UNKNOWN_DATE if kind == "web" else ""),  # 게시일 없는 검색 결과
           "site": site, "url": src.get("url", "")}
    if kind in SOURCE_FORMATS:
        label, required, template = SOURCE_FORMATS[kind]
        missing = [f for f in required if not src.get(f)]
        if not missing:
            return label, template.format(**src).strip()
        logger.warning("[REFERENCE] '%s' 출처의 필드 부족 %s → 대체 표기 사용", kind, missing)

    fallback = src.get("citation") or f'{src.get("title", "")} {src.get("url", "")}'.strip()
    label = SOURCE_FORMATS[kind][0] if kind in SOURCE_FORMATS else OTHER_LABEL
    return label, fallback


def collect_sources(state: dict) -> list[dict]:
    """투자 결정 기업의 출처만 모아 중복을 없애고 번호(n)를 매긴다.

    번호는 본문의 [n] 인용과 REFERENCE 목록을 연결하는 열쇠다.
    raw: 이 항목으로 합쳐진 State 출처 전부 (같은 보고서의 여러 쪽 조각 등). 사실 검증이 근거 원문(snippet)을 찾을 때 쓴다.
    """
    company_name = (state.get("company") or {}).get(COMPANY_NAME_KEY)
    if not company_name:  # '추천 기업 없음' 보고서에는 출처를 붙이지 않는다
        return []

    collected, by_key = [], {}
    for src in state.get("sources", []):
        if src.get("company") != company_name:
            continue
        label, text = format_source(src)
        if not text:
            continue
        # 같은 출처가 여러 노드에서 기록돼도 한 번만. 웹 기사는 노드마다 날짜 표기가 달라질 수 있어 URL로 비교한다
        # (RAG 문서는 여러 쪽이 같은 URL을 쓰지만 표기도 같으므로 표기로 비교한다)
        key = src.get("url") if src.get("kind") == "web" and src.get("url") else text
        if key in by_key:
            by_key[key]["raw"].append(src)
            continue
        by_key[key] = {"n": len(collected) + 1, "label": label, "text": text,
                       "title": src.get("title", ""), "raw": [src]}
        collected.append(by_key[key])
    return collected


def build_reference_section(sources: list[dict]) -> str:
    """REFERENCE 장을 종류별로 묶어 만든다. 번호는 본문 인용 번호와 같다."""
    if not sources:
        return f"\n\n{REFERENCE_HEADING}\n- (사용한 출처 없음)\n"
    labels = [v[0] for v in SOURCE_FORMATS.values()] + [OTHER_LABEL]  # 표시 순서 고정
    parts = [REFERENCE_HEADING]
    for label in labels:
        group = [s for s in sources if s["label"] == label]
        if group:
            parts.append(f"### {label}")
            parts.extend(f'- [{s["n"]}] {s["text"]}' for s in group)
    return "\n\n" + "\n".join(parts) + "\n"


def keep_cited_sources(body: str, sources: list[dict]) -> tuple[str, list[dict]]:
    """본문에서 [n]으로 인용한 출처만 남기고 번호를 1부터 다시 매긴다. (본문 번호도 함께 바꾼다)

    왜: 분석 에이전트가 모은 출처 중 본문에서 쓰지 않은 것은 어떤 문장의 근거인지 알 수 없다.
    번호는 REFERENCE 표시 순서(종류별 묶음 → 본문 첫 인용 순)로 매겨 목록이 1, 2, 3… 차례로 읽히게 한다.
    출처 목록에 없는 번호(AI가 지어낸 번호)는 그대로 둬 사실 검증이 잡게 한다.
    """
    by_n = {s["n"]: s for s in sources}
    first_seen = list(dict.fromkeys(int(m.group(1)) for m in CITATION_PATTERN.finditer(body)))
    cited = [by_n[n] for n in first_seen if n in by_n]
    order = [v[0] for v in SOURCE_FORMATS.values()] + [OTHER_LABEL]
    cited.sort(key=lambda s: order.index(s["label"]) if s["label"] in order else len(order))  # 안정 정렬: 묶음 안은 첫 인용 순
    renumber = {s["n"]: i for i, s in enumerate(cited, 1)}
    body = CITATION_PATTERN.sub(
        lambda m: f"[{renumber[int(m.group(1))]}]" if int(m.group(1)) in renumber else m.group(0), body)
    return body, [{**s, "n": renumber[s["n"]]} for s in cited]


def printed_reference_numbers(report: str) -> set[int]:
    """보고서 REFERENCE 장에 실제로 찍힌 출처 번호. (사실 검증이 본문 [n]을 이것과 대조한다)"""
    _, sep, reference = report.partition(REFERENCE_HEADING)
    return {int(n) for n in re.findall(r"^- \[(\d+)\]", reference, re.MULTILINE)} if sep else set()


def cited_source_map(report: str, state: dict) -> dict[int, list[dict]]:
    """REFERENCE에 찍힌 번호 → 그 항목으로 묶인 State 출처 목록(근거 원문 snippet 포함).

    번호는 보고서 생성 때 다시 매겨지므로, REFERENCE에 찍힌 표기 문자열로 State 출처를 되찾는다.
    """
    by_text = {s["text"]: s["raw"] for s in collect_sources(state)}
    _, sep, reference = report.partition(REFERENCE_HEADING)
    if not sep:
        return {}
    return {int(n): by_text.get(text.strip(), [])
            for n, text in re.findall(r"^- \[(\d+)\] (.+)$", reference, re.MULTILINE)}


# ═════════════════════════════════════════════════════════════
# 4) 과제 조건 점검
# ═════════════════════════════════════════════════════════════
def _sections(report: str) -> list[tuple[str, str]]:
    """'## 제목' 기준으로 (제목, 내용) 목록을 만든다."""
    matches = list(H2_PATTERN.finditer(report))
    return [(m.group(1), report[m.end(): matches[i + 1].start() if i + 1 < len(matches) else len(report)])
            for i, m in enumerate(matches)]


def _count_bullets(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip().startswith(("- ", "* ")))


def _summary_structure_violations(summary: str) -> list[str]:
    """투자 추천 SUMMARY가 고정 틀(결론 / 핵심 근거 3 / 핵심 리스크 3 / 확인 필요 1~2)인지 본다."""
    missing = [label for label in SUMMARY_LABELS if label not in summary]
    if missing:
        return [f"SUMMARY 틀에 다음 항목이 없습니다: {', '.join(missing)}"]

    # 소제목 순서가 틀리면 split으로 나눌 수 없어 ValueError로 그래프가 멈췄다. 위치로 순서를 먼저 확인한다
    heads = ["**핵심 근거**", "**핵심 리스크**", "**확인 필요**"]
    pos = [summary.index(h) for h in heads]
    if pos != sorted(pos):
        return ["SUMMARY 소제목은 결론 → 핵심 근거 → 핵심 리스크 → 확인 필요 순서여야 합니다."]
    basis = summary[pos[0] + len(heads[0]):pos[1]]
    risk = summary[pos[1] + len(heads[1]):pos[2]]
    confirm = summary[pos[2] + len(heads[2]):]
    problems = []
    for name, part in (("핵심 근거", basis), ("핵심 리스크", risk)):
        if _count_bullets(part) != SUMMARY_BULLETS:
            problems.append(f"SUMMARY '{name}'는 정확히 {SUMMARY_BULLETS}개여야 합니다(현재 {_count_bullets(part)}개).")
    low, high = CONFIRM_BULLETS
    if not low <= _count_bullets(confirm) <= high:
        problems.append(f"SUMMARY '확인 필요'는 {low}~{high}개여야 합니다(현재 {_count_bullets(confirm)}개).")
    return problems


# 보고서 문장의 '<채점 항목> N점' 표기. 기준 설명('2점 미만', '4점 조건', '5점 만점')은 점수 주장이 아니라 뺀다
SCORE_CLAIM_PATTERN = re.compile(
    r"(?P<label>" + "|".join(re.escape(v) for v in ITEM_LABELS.values()) + r")(?:은|는|이|가|도)?\s*"
    r"(?P<n>[1-5])\s*점(?!\s*(?:미만|이하|이상|초과|조건|구간|만점|이\s*되|으로\s*올리|에\s*해당하려면))")


def score_claim_mismatches(report: str, scores: dict) -> list[dict]:
    """본문에 쓴 '<채점 항목> N점'이 실제 항목 점수와 다른 곳. [{surface, label, claimed, actual, pos}]
    LLM이 프롬프트 예시('기술·제품 성숙도 2점(정보 부족으로 제한)')를 베껴 점수표(4점)와 다른 점수를 SUMMARY에 쓴 적이 있다.
    사실 검증의 수치 대조는 '2점'을 정보 부족 상한 값으로 보고 통과시켜, 항목과 점수를 묶어서 따로 대조한다."""
    items = score_items(scores)
    body = report.partition(REFERENCE_HEADING)[0]
    found = []
    for m in SCORE_CLAIM_PATTERN.finditer(body):
        actual = (items.get(m.group("label")) or {}).get("score")
        if isinstance(actual, (int, float)) and int(m.group("n")) != int(actual):
            found.append({"surface": m.group(0), "label": m.group("label"), "claimed": int(m.group("n")),
                          "actual": int(actual), "pos": m.start()})
    return found


COMPETITION_PARTS = ("**확인된 사실**", "**비교 결과**", "**투자 관점 해석**")   # 5장 세 부분 (규칙 22)


def check_format(report: str, structured_summary: bool = False,
                 summary_names: list[str] | tuple = (), summary_cautions: list[str] | tuple = (),
                 scores: dict | None = None) -> list[str]:
    """과제 조건 위반 목록을 돌려준다. 비어 있으면 모두 통과.

    점검: SUMMARY 맨 앞·길이, REFERENCE 맨 뒤, 전체 MAX_PAGES장 이내(글자 수 근사),
    structured_summary=True면 SUMMARY 고정 틀까지, summary_names가 있으면 SUMMARY에 그 이름(후보)이 모두 있는지,
    summary_cautions가 있으면 SUMMARY '확인 필요'에 그 항목 이름이 있는지,
    scores가 있으면 본문의 '<채점 항목> N점' 표기가 실제 점수와 같은지.
    (실제 페이지 수는 PDF 저장 때 다시 센다)
    """
    sections = _sections(report)
    if not sections:
        return ["'## ' 제목(장)이 하나도 없습니다. 목차 순서대로 장 제목을 쓰세요."]

    violations = []
    first_title, first_body = sections[0]
    last_title, _ = sections[-1]

    if "SUMMARY" not in first_title.upper():
        violations.append(f"맨 앞 장이 SUMMARY가 아닙니다(현재: '{first_title}').")
    else:
        if len(first_body.strip()) > SUMMARY_MAX_CHARS:
            violations.append(f"SUMMARY가 너무 깁니다({len(first_body.strip())}자). {SUMMARY_MAX_CHARS}자 이내로 줄이세요.")
        if structured_summary:
            violations += _summary_structure_violations(first_body)
        missing_names = [n for n in summary_names if n not in first_body]
        if missing_names:
            violations.append(f"SUMMARY 결론에 평가 후보 이름이 빠졌습니다: {', '.join(missing_names)}. "
                              f"후보 이름을 모두 쓰고 그중 선정되었다고 밝히세요.")
        confirm = first_body.split("**확인 필요**", 1)[1] if "**확인 필요**" in first_body else ""
        missing_cautions = [c for c in summary_cautions if c not in confirm]
        if missing_cautions:
            violations.append(f"SUMMARY '확인 필요'에 핵심 항목 주의가 빠졌습니다: {', '.join(missing_cautions)}. "
                              f"항목 이름과 점수를 한 항목으로 쓰세요(1~2개 항목 안에서).")

    if "REFERENCE" not in last_title.upper():
        violations.append(f"맨 뒤 장이 REFERENCE가 아닙니다(현재: '{last_title}').")
    if structured_summary:   # 투자 추천 보고서: 경쟁 구도 장을 사실·비교·해석으로 나눴는지
        competition = next((body for title, body in sections if "경쟁 구도" in title), None)
        missing = [p.strip("*") for p in COMPETITION_PARTS if competition is not None and p not in competition]
        if missing:
            violations.append(f"'경쟁 구도' 장에 소제목이 빠졌습니다: {', '.join(missing)}. "
                              f"확인된 사실 / 비교 결과 / 투자 관점 해석으로 나눠 쓰세요.")

    limit = PAGE_CHARS * MAX_PAGES
    if len(report) > limit:
        violations.append(f"전체 분량이 {len(report)}자로 {MAX_PAGES}장 기준({limit}자)을 넘습니다. 줄이세요.")
    for m in score_claim_mismatches(report, scores or {}):
        violations.append(f"'{m['surface']}'는 점수표의 실제 점수({m['label']} {m['actual']}점)와 다릅니다. "
                          f"점수는 점수표와 같게 쓰고, 근거 없는 점수는 쓰지 마세요.")
    return violations


# ═════════════════════════════════════════════════════════════
# 5) 보고서 생성 노드
# ═════════════════════════════════════════════════════════════
def validate_state(state: dict) -> None:
    """필요한 State 키가 없으면 바로 멈춘다(fail-fast).

    왜: 재료가 빠진 채로 LLM을 부르면 돈만 쓰고 엉터리 보고서가 나온다.
    """
    if _is_recommend_mode(state):
        missing = [k for k in ANALYSIS_KEYS if not state.get(k)]
        if missing:
            raise KeyError(f"보고서 작성에 필요한 State 키가 비어 있습니다: {missing}")


def _is_recommend_mode(state: dict) -> bool:
    """투자 결정 기업이 있는가? (있으면 일반 보고서, 없으면 '추천 기업 없음' 보고서)"""
    return (state.get("scores") or {}).get("decision") == DECISION_INVEST


def build_materials(state: dict, sources: list[dict]) -> dict:
    """LLM에게 줄 '분석 자료' 묶음을 만든다. (자료에 없는 건 LLM이 못 쓰게 하려는 것)"""
    if _is_recommend_mode(state):
        # 분석 에이전트가 문장에 남긴 자기 검색 결과 번호([3] 등)는 보고서 출처 번호와 다르므로 지운다
        materials = {key: strip_agent_refs(state[key]) for key in ANALYSIS_KEYS}
        if isinstance(materials.get("tech_summary"), dict):
            materials["tech_summary"] = {k: v for k, v in materials["tech_summary"].items()
                                        if k != "diagnostics"}
        # LLM이 [n]을 붙일 수 있도록 '번호 붙은 출처 목록'을 보여 준다(표기 원문은 코드가 관리)
        materials["sources_for_citation"] = [
            {"n": s["n"], "type": s["label"], "title": s["title"],
             "used_by": sorted({r.get("node", "") for r in s.get("raw", [])} - {""})} for s in sources
        ]
        # SUMMARY 결론에 '후보 A, B, C 중 선정'을 쓰게 하려는 것. (평가 순서·사유 표는 코드가 6장에 넣는다)
        materials["selection"] = {"candidates": state.get("candidates") or [],
                                  "selected": state["company"].get(COMPANY_NAME_KEY)}
        hint = caution_summary_hint(state["scores"])
        if hint:   # 핵심 항목이 낮은 채로 투자 결정이 났을 때 SUMMARY '확인 필요'에 반드시 알린다
            materials["summary_caution"] = hint
    else:
        materials = {"candidates": state.get("candidates") or []}
    materials["rejected"] = state.get("rejected", [])  # 보류·제외 기업과 사유는 한계점에 항상 쓰인다
    materials["evaluation_rules"] = evaluation_rules()  # AI가 기준을 추측해 틀리게 쓰지 않도록
    if state.get("last_error"):   # supervisor 단계 한도 등으로 평가가 중간에 멈춘 경우 (규칙 19)
        materials["halted"] = state["last_error"]
    return materials


def evaluation_rules() -> dict:
    """보고서에 쓸 평가 기준 (설계서 2.1·3.4·3.5, 값은 config.py). 보고서 문장이 기준을 설명할 때 이것만 따른다."""
    core = ", ".join(ITEM_LABELS[k] for k in CORE_ITEMS)   # 항목 이름에 '·'가 있어 쉼표로 구분
    return {
        "평가 대상 조건": f"국내 AI 반도체 설계 기업, 비상장, 투자 단계 {ALLOWED_ROUNDS[0]}~{ALLOWED_ROUNDS[-1]}, Exit 전",
        "투자 기준": f"환산 점수 {DECISION_THRESHOLD}점 이상이고 보류 조건에 해당하지 않음",
        "보류 조건": [f"{core} 중 하나라도 {CORE_MIN_SCORE}점 미만(핵심 역량 미달)",
                  "핵심 기술 관련 법률 리스크가 해소되지 않음",
                  f"환산 점수 {DECISION_THRESHOLD}점 미만"],
        "주의 규칙": f"{core}가 {CORE_CAUTION_SCORE}점 이하이면 보류 사유는 아니지만 보고서에 주의로 알린다",
        "정보 부족 규칙": f"공개 정보로 확인할 수 없는 항목은 최대 {INSUFFICIENT_MAX_SCORE}점",
        "평가 방식": f"발굴한 후보(최대 {MAX_CANDIDATES}곳)를 순서대로 평가하고, 처음으로 투자 결정이 나오면 멈춘다",
    }


def split_grouped_citations(body: str) -> str:
    """[6,7]·[8-11]을 [6][7]·[8][9][10][11]로 푼다.
    왜: 번호를 다시 매길 때 [n] 형식만 바뀌어, 묶어 쓴 번호는 옛 번호로 남아 REFERENCE와 어긋났다
    (사실 검증도 '6,7'을 자료에 없는 수치로 표시했다).
    """
    def expand(m):
        numbers = []
        for part in re.split(r"\s*,\s*", m.group(1)):
            bounds = re.split(r"\s*[~\-]\s*", part)
            if len(bounds) == 2 and int(bounds[0]) <= int(bounds[1]) <= int(bounds[0]) + 20:
                numbers += range(int(bounds[0]), int(bounds[1]) + 1)
            else:
                numbers += [int(b) for b in bounds]
        return "".join(f"[{n}]" for n in numbers)
    return GROUPED_CITATION_PATTERN.sub(expand, body)


def assemble_report(state: dict, body: str, sources: list[dict]) -> str:
    """AI가 쓴 본문에 코드가 만든 부분(점수표, REFERENCE)을 합쳐 완성본을 만든다."""
    body = REFERENCE_SECTION_PATTERN.sub("", body).rstrip()  # AI가 쓴 REFERENCE는 지운다(진짜는 코드가)
    body = split_grouped_citations(body)                     # [6,7] → [6][7] (번호 다시 매기기 전에)
    body = KEY_CITATION_PATTERN.sub("", body)                 # [team_analysis] 같은 자료 이름 표기 삭제
    body = KEY_NAME_LINE_PATTERN.sub("", body)                # '- summary_caution 없음' 같은 줄 삭제
    body = translate_field_names(body)                       # 'weaknesses' 같은 필드명을 한글로
    body = HR_PATTERN.sub("", body)                          # '---' 구분선은 PDF에 글자로 찍히므로 지운다
    if _is_recommend_mode(state):
        body = insert_score_table(body, state["scores"])
        body = insert_score_basis_tables(body, state["scores"], sources)
        body = insert_competitor_table(body, state, sources)
        body = insert_timeline_table(body, state, sources)
        body, sources = keep_cited_sources(body, sources)   # REFERENCE에는 본문에서 인용한 출처만
    else:  # 추천 없음 보고서: 점수표·경쟁사 표 대신 2장에 후보별 결과 표(코드가 State에서 만든다)
        body = body.replace(PLACEHOLDER, "").replace(COMPETITOR_PLACEHOLDER, "")
        table = build_candidate_table(state)
        if table:
            body = _insert_table(body, CANDIDATE_PLACEHOLDER, table, SECTION2_PATTERN, "후보 표")
    body = body.replace(CANDIDATE_PLACEHOLDER, "")   # 표를 못 만든 경우 자리표시자만 지운다
    return body + build_reference_section(sources)


def _get_llm():
    """LLM 객체를 만든다. API 키가 없으면 여기서 바로 알려준다."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("환경변수 OPENAI_API_KEY가 설정되어 있지 않습니다.")
    from langchain.chat_models import init_chat_model  # 필요할 때만 import (테스트 시 없어도 되게)

    return init_chat_model(LLM_MODEL, model_provider="openai", temperature=LLM_TEMPERATURE)


# ── 지적된 장만 다시 쓰기 ─────────────────────────────────────
# 재작성 때 보고서 전체를 새로 쓰면 지적받지 않은 장에서 새 오류가 생겼다 (예: 없던 양산 시점 문장).
# 직전 보고서에서 지적된 장만 새로 쓰고, 나머지 장은 코드가 직전 내용으로 되돌린다.
TABLE_CHAPTERS = ("경쟁 구도", "투자 판단 결과")   # 표를 코드가 넣는 장: 항상 새 본문을 쓴다
AUTO_NOTE_HEADINGS = ("### 자동 검증 결과", "### 품질 평가 미달 항목")   # 검증·평가 노드가 붙인 부분 (다시 붙는다)
CODE_TABLE_CAPTIONS = (TIMELINE_CAPTION, RUBRIC_POSITION_CAPTION, SCORE_BASIS_CAPTION)   # assemble_report가 다시 넣는 보조 표
UNVERIFIED_MARKS = ("(확인 필요)", "[출처 확인 필요]")


def _chapter_key(title: str) -> str:
    """'5. 경쟁 구도 (설명)' → '경쟁 구도'. 장을 직전 보고서와 맞출 때 쓴다."""
    return re.sub(r"\s*\(.*?\)\s*$", "", re.sub(r"^\s*\d+\.\s*", "", title)).strip()


def _cut_block(text: str, heading: str) -> str:
    """heading부터 다음 '## '/'### ' 제목 전까지 지운다."""
    start = text.find(heading)
    if start < 0:
        return text
    nxt = re.search(r"^#{2,3} ", text[start + len(heading):], re.MULTILINE)
    end = start + len(heading) + nxt.start() if nxt else len(text)
    return text[:start].rstrip() + "\n" + text[end:]


def _strip_code_tables(text: str) -> str:
    """코드가 넣은 보조 표(제목 줄과 바로 아래 표)를 뺀다.
    왜: 1장은 직전 내용으로 되돌아가는데 타임라인이 남아 있으면 assemble_report가 한 번 더 넣어 표가 둘이 된다."""
    out, skipping = [], False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped in CODE_TABLE_CAPTIONS:
            skipping = True
            continue
        if skipping and (not stripped or stripped.startswith("|")):
            continue
        skipping = False
        out.append(line)
    return "\n".join(out)


def previous_body(state: dict, sources: list[dict]) -> str:
    """직전 보고서 본문을 재작성 기준으로 만든다: 출처 번호를 sources_for_citation 번호로 되돌리고,
    검증·평가 노드가 붙인 부분과 코드가 넣은 표는 빼고 표 자리 표시로 바꾼다."""
    body, _, reference = (state.get("report") or "").partition(REFERENCE_HEADING)
    by_text = {s["text"]: s["n"] for s in sources}
    back = {int(k): by_text.get(t.strip()) for k, t in re.findall(r"^- \[(\d+)\] (.+)$", reference, re.MULTILINE)}
    body = CITATION_PATTERN.sub(lambda m: f"[{back[int(m.group(1))]}]" if back.get(int(m.group(1))) else "", body)
    for heading in AUTO_NOTE_HEADINGS:
        body = _cut_block(body, heading)
    chapters = []
    for title, text in _sections(body):
        text = _strip_code_tables(text)
        if any(k in title for k in TABLE_CHAPTERS):   # 코드가 넣은 표를 지우고 자리 표시로
            kept = [ln for ln in text.splitlines() if not ln.strip().startswith("|")
                    and not ln.startswith((EXCLUDED_PREFIX, NOT_COMPARABLE_PREFIX))]
            mark = COMPETITOR_PLACEHOLDER if "경쟁 구도" in title else PLACEHOLDER
            text = "\n" + mark + "\n" + "\n".join(kept).strip() + "\n"
        chapters.append(f"## {title}\n{text.strip()}\n")
    return "\n".join(chapters).strip()


def flagged_chapters(prev: str, fix_text: str, mismatch_values: list[str]) -> set[str]:
    """수정 지시가 가리키는 장: 장 이름·'N장'이 지시에 나오거나, 불일치 값·확인 필요 표시가 들어 있는 장."""
    flagged = set()
    for title, text in _sections(prev):
        key, num = _chapter_key(title), re.match(r"\s*(\d+)\.", title)
        if (key in fix_text or (num and f"{num.group(1)}장" in fix_text)
                or any(v and v in text for v in mismatch_values) or any(m in text for m in UNVERIFIED_MARKS)):
            flagged.add(key)
    return flagged


def merge_rewrite(new_body: str, prev: str, flagged: set[str]) -> str:
    """새 본문에서 지적되지 않은 장(표 장 제외)을 직전 보고서 내용으로 되돌린다."""
    before = {_chapter_key(t): x for t, x in _sections(prev)}
    head = new_body[: H2_PATTERN.search(new_body).start()] if H2_PATTERN.search(new_body) else ""
    out = []
    for title, text in _sections(new_body):
        key = _chapter_key(title)
        if key in before and key not in flagged and not any(k in key for k in TABLE_CHAPTERS):
            text = before[key]
        out.append(f"## {title}\n{text.strip()}\n")
    return (head + "\n".join(out)).strip()


def partial_rewrite_plan(state: dict, sources: list[dict]) -> tuple[str, set[str]]:
    """(직전 본문, 다시 쓸 장). 부분 재작성 대상이 아니면 ("", set()).
    대상: 직전 보고서(report_version - 1)가 사실 검증·품질 평가에서 미달이고, 근거 부족 재조사로 자료가 바뀐 경우가 아님."""
    if not state.get("report"):
        return "", set()
    previous = (state.get("report_version") or 0) - 1
    verify, evaluation = state.get("verify_result") or {}, state.get("eval_result") or {}
    verify_failed = verify.get("report_version") == previous and not verify.get("passed", True)
    eval_failed = evaluation.get("report_version") == previous and not evaluation.get("passed", True)
    if not (verify_failed or eval_failed) or evaluation.get("evidence_gaps"):
        return "", set()   # 재조사 후에는 자료와 점수가 바뀌어 전체를 새로 쓴다
    prev = previous_body(state, sources)
    fix_text = (evaluation.get("fix_instructions") or "") if eval_failed else ""
    values = [m.get("value", "") for m in verify.get("mismatches") or []] if verify_failed else []
    flagged = flagged_chapters(prev, fix_text, values)
    return (prev, flagged) if flagged else ("", set())


def _verify_fix_notes(state: dict) -> str:
    """사실 검증이 불일치를 찾아 되돌려 보냈다면, 그 목록을 프롬프트에 붙인다."""
    result = state.get("verify_result") or {}
    mismatches = result.get("mismatches", [])
    if result.get("passed", True) or not mismatches:
        return ""
    items = "\n".join(f'- "{m["value"]}" (문맥: {m["context"]})' for m in mismatches)
    return FIX_NOTES_TEMPLATE.format(items=items)


def _quality_fix_notes(state: dict) -> str:
    """직전 보고서(report_version - 1)가 품질 평가에서 미달이었다면 수정 지시를 프롬프트에 붙인다."""
    result = state.get("eval_result") or {}
    previous = (state.get("report_version") or 0) - 1
    if result.get("passed", True) or result.get("report_version") != previous or not result.get("fix_instructions"):
        return ""
    return QUALITY_FIX_TEMPLATE.format(items=result["fix_instructions"])


def reporter_node(state: dict, llm=None) -> dict:
    """보고서 생성 노드. State를 받아 '바꿀 키만' dict로 돌려준다.

    LangGraph 규칙: 노드는 State 전체가 아니라 갱신할 키만 반환하면 된다.
    llm 인자는 테스트 때 가짜 LLM을 끼우려는 용도(평소엔 None).
    """
    logger.info("[보고서 생성] 시작")
    validate_state(state)

    recommend = _is_recommend_mode(state)
    mode = "투자 추천 보고서" if recommend else "추천 기업 없음 보고서"
    logger.info("[보고서 생성] 유형: %s", mode)

    # 재작성 횟수·한도는 supervisor가 관리한다 (제어 키). 여기서는 몇 번째 보고서인지 기록만 한다
    logger.info("[보고서 생성] report_version=%s, 사실 검증 재작성=%s회, 품질 재작성=%s회",
                state.get("report_version"), state.get("retry_count", 0), state.get("quality_loops", 0))

    sources = collect_sources(state) if recommend else []
    logger.info("[보고서 생성] 사용 가능한 출처 %d건", len(sources))
    materials = json.dumps(build_materials(state, sources), ensure_ascii=False, indent=2)
    llm = llm or _get_llm()

    verify_notes, format_notes, report = _verify_fix_notes(state) + _quality_fix_notes(state), "", ""
    prev, flagged = partial_rewrite_plan(state, sources)
    if flagged:   # 지적된 장만 다시 쓴다 (나머지 장은 아래 merge_rewrite가 직전 내용으로 되돌림)
        logger.info("[보고서 생성] 부분 재작성: %s", sorted(flagged))
        verify_notes += PARTIAL_REWRITE_TEMPLATE.format(chapters=", ".join(sorted(flagged)), previous=prev)
    for attempt in range(MAX_FORMAT_FIXES + 1):
        user_prompt = USER_TEMPLATE.format(
            mode=mode, toc=TOC_RECOMMEND if recommend else TOC_NO_RECOMMEND,
            materials=materials, fix_notes=verify_notes + format_notes,
        )
        logger.info("[보고서 생성] LLM 호출 (시도 %d)", attempt + 1)
        response = llm.invoke([("system", SYSTEM_PROMPT), ("user", user_prompt)])
        body = response.content.strip()
        if flagged:
            body = merge_rewrite(body, prev, flagged)
        report = assemble_report(state, body, sources)

        violations = check_format(report, structured_summary=recommend,   # 과제 조건을 코드로 점검
                                  summary_names=(state.get("candidates") or []) if recommend else (),
                                  summary_cautions=[c.get("label", "") for c in cautions_of(state["scores"])
                                                    if c.get("label")] if recommend else (),
                                  scores=state["scores"] if recommend else None)
        if not violations:
            break
        logger.warning("[보고서 생성] 과제 조건 위반 %d건: %s", len(violations), violations)
        format_notes = FORMAT_FIX_TEMPLATE.format(items="\n".join(f"- {v}" for v in violations))
    else:
        logger.warning("[보고서 생성] 형식 수정 후에도 위반이 남았습니다. 현재 본문을 유지합니다.")

    logger.info("[보고서 생성] 완료 (%d자)", len(report))
    return {"report": report}


# ═════════════════════════════════════════════════════════════
# [G] PDF 저장 — 1쪽: 제목 밴드 → SUMMARY → 결정 대시보드, 이후 본문·REFERENCE
#     한글 글꼴: 환경변수 KOREAN_FONT_PATH → fonts/Pretendard-Regular.ttf → 시스템 글꼴 → 내장 글꼴 순으로 찾는다.
#     (reportlab은 TrueType(.ttf)만 쓸 수 있고 .otf/CFF 글꼴은 못 쓴다)
# ═════════════════════════════════════════════════════════════
# ── 설정값(상수) ─────────────────────────────────────────────
CAMPUS = "울산캠퍼스"                            # 설계서 파일명 기준. 조가 바뀌면 여기만 수정
CLASS_NO = "4반"
MEMBERS = ["강지수", "김다은", "이진우", "이채목", "장정훈"]
OUTPUT_DIR = Path("outputs")                   # 결과 저장 폴더 (설계서 디렉터리 구조의 outputs/)
FONT_ENV_VAR = "KOREAN_FONT_PATH"
FONT_NAME, FONT_NAME_BOLD = "KoreanFont", "KoreanFont-Bold"
CID_FALLBACK_FONT = "HYGothic-Medium"
FONT_CANDIDATES = [
    Path("fonts/Pretendard-Regular.ttf"),   # 저장소에 포함 (SIL OFL)
    Path("fonts/NanumGothic.ttf"),
    Path("/Library/Fonts/NanumGothic.ttf"),
    Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf"),   # macOS
    Path("C:/Windows/Fonts/malgun.ttf"),                          # Windows
    Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),      # Linux
]

# 색 팔레트 — 투자=초록, 보류=주황, 확인 필요=빨강, 기본=남색
NAVY, INK, MUTED = "#0b3d91", "#1a1a1a", "#6b7280"
LINE, PANEL, NOTE_BG = "#d8dee9", "#f3f6fb", "#f5f5f5"
GOOD, HOLD, BAD = "#1f8a4c", "#d97706", "#c0392b"

# 레이아웃
PAGE_MARGIN = 16 * mm
BOTTOM_EXTRA = 4 * mm                          # 바닥글 자리
USABLE_PAGE_HEIGHT = A4[1] - 2 * PAGE_MARGIN - BOTTOM_EXTRA   # 한 쪽의 본문 높이
SUMMARY_MAX_PAGE_RATIO = 0.5                   # 과제 조건: SUMMARY는 1/2 페이지 이내
BODY_SIZE, SMALL_SIZE = 10, 8.5
LINE_RATIO = 1.8                               # 줄 간격 (글자 크기 배수)
PARA_GAP, BULLET_GAP = 7, 3                    # 문단·글머리표 뒤 간격(pt)
SECTION_GAP, HEADING_GAP = 18, 8               # 장 제목 앞·뒤 간격(pt)
# 이 장들은 쪽에 남은 공간이 PAGE_BREAK_ROOM보다 적으면 새 쪽에서 시작한다 (강제로 넘기면 앞 장의 마지막
# 한두 줄만 있는 빈 쪽이 생겼다). 본문 1장은 항상 새 쪽 (1쪽은 제목 밴드·SUMMARY·대시보드만)
PAGE_BREAK_BEFORE = ("경쟁 구도", "투자 판단 결과")
PAGE_BREAK_ROOM = 0.3
# REFERENCE는 제목과 출처 두세 줄이 들어갈 공간만 있으면 이어서 쓴다 (30%면 한계점 뒤 30% 가까이 비고 출처가 다음 쪽으로 밀렸다)
REFERENCE_ROOM = 0.12
BODY_START_ROOM = 0.6   # 본문 1장: 쪽에 남은 공간이 60%보다 적으면(=1쪽이 SUMMARY·대시보드로 찼으면) 새 쪽
LAYOUT_SCALES = (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6)   # MIN_PAGES에 못 미치거나 마지막 쪽이 거의 비면 간격 배율을 키워 다시 조판
# 마지막 쪽이 이보다 적게 차면 다른 배율을 찾는다. 최소 쪽수를 맞추려고 배율을 키우다 출처 몇 줄만 넘어간 순간 멈춰,
# 마지막 쪽이 90% 넘게 빈 7쪽짜리가 나왔다
LAST_PAGE_MIN_FILL = 0.3
DASH_HEIGHT = 168                              # 대시보드 카드·차트 높이(pt)
CARD_RATIO = 0.42                              # 대시보드에서 왼쪽 카드가 차지하는 폭 비율
SCORE_BAR_W, SCORE_BAR_H = 70, 9
MAX_CANDIDATE_ROWS = 8                         # 대시보드 후보별 점수 막대 최대 개수
SCORE_TABLE_FRACS = (0.36, 0.26, 0.14, 0.24)   # 점수표(4칸) 열 폭 비율
TABLE_FRACS = {                                # 표 첫 칸 제목 → 열 폭 비율 (없으면 균등)
    "항목": SCORE_TABLE_FRACS,
    "순서": (0.07, 0.17, 0.12, 0.12, 0.52),     # 후보 평가 과정 표: '사유' 칸을 넓게
    TIMELINE_HEADER: (0.14, 0.72, 0.14),        # 개발 단계 타임라인: '내용' 칸을 넓게
    RUBRIC_POSITION_HEADER: (0.17, 0.25, 0.29, 0.29),   # 점수 칸은 점수 막대가 들어가 점수표만큼 넓게
    SCORE_BASIS_HEADER: (0.18, 0.68, 0.14),     # 항목별 판단 근거: '판단 근거' 칸을 넓게
    COMPARISON_HEADER: (0.2, 0.8),              # 경쟁사 비교 근거
}

# State 안의 company 키 후보 (팀원 State와 다르면 여기만 수정)
SECTOR_KEYS = ("segment", "sector", "세부_분야", "세부분야", "domain")
STAGE_KEYS = ("stage", "투자_단계", "투자단계", "round")

HEADING_PATTERN = re.compile(r"^(#{1,3})\s+(.*)$")
SCORE_CELL_PATTERN = re.compile(r"^(\d)/(\d)$")
CITATION_PATTERN = re.compile(r"\[(\d+)\]")
NOTE_TITLE = "자동 검증 결과"


# ── 파일명 ───────────────────────────────────────────────────
def build_output_filename(campus: str = CAMPUS, class_no: str = CLASS_NO,
                          members: list[str] | None = None) -> str:
    """보고서 PDF 파일명: Agent-Output_{캠퍼스}-{X반}_{이름1+이름2+...}.pdf (제출 zip은 Agent_{캠퍼스}_{X반}_{이름...}.zip)"""
    members = members or MEMBERS
    if not members:
        raise ValueError("조원 이름 목록이 비어 있습니다.")
    return f"Agent-Output_{campus}-{class_no}_{'+'.join(members)}.pdf"


# ── 글꼴 ─────────────────────────────────────────────────────
def _find_font_files() -> list[Path]:
    """글꼴 후보를 우선순위 순서로 돌려준다."""
    candidates = [Path(os.environ[FONT_ENV_VAR])] if os.environ.get(FONT_ENV_VAR) else []
    return [p for p in candidates + FONT_CANDIDATES if p.exists()]


def register_korean_font() -> tuple[str, str]:
    """한글 글꼴을 등록하고 (보통 글꼴 이름, 굵은 글꼴 이름)을 돌려준다."""
    for path in _find_font_files():
        try:
            pdfmetrics.registerFont(TTFont(FONT_NAME, str(path)))
        except (TTFError, OSError) as err:
            logger.warning("[PDF] 글꼴 사용 불가(%s): %s", path, err)
            continue
        bold_stem = path.stem.replace("Regular", "Bold") if "Regular" in path.stem else path.stem + "Bold"
        bold_path, bold_name = path.with_name(bold_stem + path.suffix), FONT_NAME
        if bold_path.exists():  # 굵은 글꼴이 같은 폴더에 있으면 쓰고, 없으면 보통 글꼴로 대신한다
            try:
                pdfmetrics.registerFont(TTFont(FONT_NAME_BOLD, str(bold_path)))
                bold_name = FONT_NAME_BOLD
            except (TTFError, OSError) as err:
                logger.warning("[PDF] 굵은 글꼴 사용 불가(%s): %s", bold_path, err)
        pdfmetrics.registerFontFamily(FONT_NAME, normal=FONT_NAME, bold=bold_name,
                                      italic=FONT_NAME, boldItalic=bold_name)
        logger.info("[PDF] 한글 글꼴 사용: %s", path)
        return FONT_NAME, bold_name

    logger.warning("[PDF] 한글 TrueType 글꼴을 찾지 못해 내장 글꼴(%s)을 사용합니다. "
                   "fonts/Pretendard-Regular.ttf를 넣으면 더 안정적입니다.", CID_FALLBACK_FONT)
    pdfmetrics.registerFont(UnicodeCIDFont(CID_FALLBACK_FONT))
    pdfmetrics.registerFontFamily(CID_FALLBACK_FONT, normal=CID_FALLBACK_FONT, bold=CID_FALLBACK_FONT,
                                  italic=CID_FALLBACK_FONT, boldItalic=CID_FALLBACK_FONT)
    return CID_FALLBACK_FONT, CID_FALLBACK_FONT


# ── 스타일 ───────────────────────────────────────────────────
def _hex(code: str):
    return colors.HexColor(code)


def _inline(text: str) -> str:
    """한 줄 안의 서식을 PDF용으로 바꾼다. (&,<,> 이스케이프 후 **굵게** 처리)"""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


_LAYOUT_SCALE = 1.0   # export_report_pdf가 MIN_PAGES를 맞추려고 바꾸는 간격 배율


def _gap(points: float) -> float:
    return points * _LAYOUT_SCALE


def _styles(font: str, bold: str, size: float = BODY_SIZE, scaled: bool = True) -> dict[str, ParagraphStyle]:
    """본문 크기(size)에 맞춘 문단 스타일 묶음. 줄·문단 간격은 _LAYOUT_SCALE배 (scaled=False면 배율 1)."""
    global _LAYOUT_SCALE
    saved = _LAYOUT_SCALE
    if not scaled:
        _LAYOUT_SCALE = 1.0
    try:
        return _build_styles(font, bold, size)
    finally:
        _LAYOUT_SCALE = saved


def _build_styles(font: str, bold: str, size: float) -> dict[str, ParagraphStyle]:
    base = dict(fontName=font, alignment=TA_LEFT, textColor=_hex(INK))
    lead = size * LINE_RATIO * (1 + (_LAYOUT_SCALE - 1) / 2)   # 줄 간격은 배율의 절반만 키운다
    return {
        "body": ParagraphStyle("body", fontSize=size, leading=lead, spaceAfter=_gap(PARA_GAP), **base),
        "bullet": ParagraphStyle("bullet", fontSize=size, leading=lead, leftIndent=11, bulletIndent=2,
                                 spaceAfter=_gap(BULLET_GAP), **base),
        "sub": ParagraphStyle("sub", fontSize=size - 0.5, leading=lead - 1, leftIndent=24, bulletIndent=14,
                              **{**base, "textColor": _hex(MUTED)}),
        "label": ParagraphStyle("label", fontSize=size, leading=lead, spaceBefore=3,
                                **{**base, "fontName": bold, "textColor": _hex(NAVY)}),
        "h1": ParagraphStyle("h1", fontSize=16, leading=22, spaceAfter=6, **{**base, "fontName": bold}),
        "h2": ParagraphStyle("h2", fontSize=13.5, leading=19, **{**base, "fontName": bold, "textColor": _hex(NAVY)}),
        "h3": ParagraphStyle("h3", fontSize=size + 1, leading=(size + 1) * 1.4, spaceBefore=_gap(6), spaceAfter=_gap(3),
                             keepWithNext=1, **{**base, "fontName": bold}),   # 소제목이 쪽 끝에 홀로 남지 않게
        "cell": ParagraphStyle("cell", fontSize=size - 0.5, leading=(size - 0.5) * 1.5, **base),
    }


# ── 작은 그래픽: 점수 막대 ───────────────────────────────────
def _score_color(score: int) -> str:
    return GOOD if score >= 4 else NAVY if score == 3 else HOLD


def _score_bar(score: int, max_score: int, font: str) -> Drawing:
    """표 안에 들어가는 작은 점수 막대. (예: 4/5 → 80% 채워진 막대와 '4/5' 글자)"""
    drawing = Drawing(SCORE_BAR_W + 26, SCORE_BAR_H + 2)
    drawing.add(Rect(0, 1, SCORE_BAR_W, SCORE_BAR_H, fillColor=_hex(LINE), strokeColor=None))
    drawing.add(Rect(0, 1, SCORE_BAR_W * score / max_score, SCORE_BAR_H,
                     fillColor=_hex(_score_color(score)), strokeColor=None))
    drawing.add(String(SCORE_BAR_W + 5, 2, f"{score}/{max_score}", fontName=font, fontSize=8, fillColor=_hex(INK)))
    return drawing


# ── 마크다운 → PDF 조각 ──────────────────────────────────────
def _table(rows: list[str], styles: dict, width: float, font: str) -> Table:
    """마크다운 표(| a | b |) 줄들을 PDF 표로 바꾼다. 'N/5' 칸은 점수 막대로 바꾼다."""
    parsed = []
    for line in rows:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{3,}:?", c) for c in cells):  # |---|---| 구분선은 건너뜀
            continue
        row = []
        for cell in cells:
            m = SCORE_CELL_PATTERN.match(cell)
            row.append(_score_bar(int(m.group(1)), int(m.group(2)), font) if m
                       else Paragraph(_inline(cell), styles["cell"]))
        parsed.append(row)

    col_count = max(len(r) for r in parsed)
    for r in parsed:  # 칸 수가 모자란 줄은 빈 칸으로 채운다
        r.extend([Paragraph("", styles["cell"])] * (col_count - len(r)))
    first_cell = rows[0].strip().strip("|").split("|")[0].strip()
    fracs = TABLE_FRACS.get(first_cell)
    if fracs is None or len(fracs) != col_count:
        fracs = [1 / col_count] * col_count
    table = Table(parsed, colWidths=[width * f for f in fracs], repeatRows=1)

    style = [
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, _hex(LINE)),
        ("BACKGROUND", (0, 0), (-1, 0), _hex(PANEL)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]
    last_first_cell = rows[-1].strip().strip("|").split("|")[0]
    if "환산 점수" in last_first_cell:  # 합계 행은 강조 (첫 칸만 본다: 사유 칸의 '환산 점수 65점'에 걸리지 않게)
        style.append(("BACKGROUND", (0, len(parsed) - 1), (-1, len(parsed) - 1), _hex(PANEL)))
        style.append(("LINEABOVE", (0, len(parsed) - 1), (-1, len(parsed) - 1), 1, _hex(NAVY)))
    table.setStyle(TableStyle(style))
    return table


def _md_rows(block: list[str]) -> list[list[str]]:
    """마크다운 표 줄들을 칸 목록으로. (|---| 구분선은 뺀다)"""
    rows = [[c.strip() for c in line.strip().strip("|").split("|")] for line in block]
    return [r for r in rows if not all(re.fullmatch(r":?-{3,}:?", c) for c in r)]


def _column_box(title: str, caption: str, cards: list, width: float, background: str, bar: str,
                styles: dict) -> Table:
    """구도 그림의 한 칸: 제목 → 기업 카드들 → 아래 설명."""
    head = Paragraph(f'<font color="{bar}"><b>{_inline(title)}</b></font>', styles["cell"])
    foot = Paragraph(f'<font size="7" color="{MUTED}">{_inline(caption)}</font>', styles["cell"])
    box = Table([[head]] + [[c] for c in cards] + [[foot]], colWidths=[width])
    box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), _hex(background)),
                             ("LINEABOVE", (0, 0), (-1, 0), 2.5, _hex(bar)),
                             ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                             ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return box


CARD_TEXT_CHARS, CARD_PRODUCT_CHARS = 110, 60   # 경쟁 구도 그림 카드에 싣는 설명·제품 글자 수
SENTENCE_END = re.compile(r"(?<=[.!?])\s|(?<=다\.)")


def _short(text: str, limit: int) -> str:
    """첫 문장이 limit 글자 안이면 첫 문장, 아니면 빈 문자열.
    (중간에 잘라 '…'을 붙이면 읽는 사람이 원문을 알 수 없어, 카드에는 짧은 소개만 싣고 긴 글은 싣지 않는다)"""
    text = (text or "").strip()
    first = SENTENCE_END.split(text, maxsplit=1)[0].strip() if text else ""
    return first if len(first) <= limit else ""


def comparison_rows(block: list[str]) -> list[str]:
    """경쟁 구도 표(새 형식)에서 경쟁사별 비교 근거만 뽑아 2칸 표 줄로. 예전 형식이면 빈 목록 (카드에 비교 문장이 있다)."""
    header, *rows = _md_rows(block)
    if len(header) < 6:
        return []
    body = [f"| {r[1]} | {r[5]} |" for r in rows if len(r) >= 6 and r[0] != TARGET_TIER]
    return [f"| {COMPARISON_HEADER} | 비교 근거 (발표 주체, 출처) |", "|---|---|", *body] if body else []


def _landscape(block: list[str], styles: dict, width: float) -> Table:
    """경쟁 구도 그림: [선도 기업(진입 위협)] ← [평가 대상] → [동급 기업(직접 경쟁)]."""
    header, *rows = _md_rows(block)
    with_intro = len(header) >= 6   # 새 형식: 구분|기업|국가|제품|소개|비교 근거 (예전 보고서는 소개 칸이 없다)
    groups = {LEADING_TIER: [], TARGET_TIER: [], PEER_TIER: []}
    for row in rows:
        tier, name, country, product, intro, comparison = (
            (row + [""] * 6)[:6] if with_intro else [*(row + [""] * 5)[:4], "", (row + [""] * 5)[4]])
        groups[tier if tier in groups else PEER_TIER].append((name, country, product, intro, comparison))

    def card(name, country, product, intro, comparison):
        # 경쟁사 설명이 길면(영문 소개문 등) 그림이 한 쪽 남은 공간보다 커져 통째로 다음 쪽으로 넘어가고
        # 앞 쪽이 비었다. 카드에는 짧은 소개만 싣고, 비교 근거는 그림 아래 표에 출처 번호와 함께 싣는다
        product = product if len(product) <= CARD_PRODUCT_CHARS else _short(product, CARD_PRODUCT_CHARS)
        text_line = _short(intro, CARD_TEXT_CHARS) if with_intro else _short(comparison, CARD_TEXT_CHARS)
        meta = " · ".join(v for v in (country, product) if v and v != "–")
        text = f"<b>{_inline(name)}</b>" + (f' <font size="7.5" color="{MUTED}">{_inline(meta)}</font>' if meta else "")
        if text_line and text_line != "–":
            text += f'<br/><font size="7.5">{_inline(text_line)}</font>'
        return Paragraph(text, styles["cell"])

    none = [Paragraph(f'<font color="{MUTED}">해당 기업 없음</font>', styles["cell"])]
    side_w, mid_w, gap_w = width * 0.37, width * 0.2, width * 0.03
    left = _column_box(f"{LEADING_TIER} · 대기업·상장사", "진입 위협 (시장 장악력)",
                       [card(*g) for g in groups[LEADING_TIER]] or none, side_w - 2, PANEL, HOLD, styles)
    right = _column_box(f"{PEER_TIER} · 비상장 스타트업", "직접 경쟁 (비슷한 단계)",
                        [card(*g) for g in groups[PEER_TIER]] or none, side_w - 2, PANEL, NAVY, styles)
    target = groups[TARGET_TIER][0] if groups[TARGET_TIER] else ("", "", "", "", "")
    mid_style = ParagraphStyle("mid", parent=styles["cell"], textColor=colors.white, alignment=1)
    mid = Table([[Paragraph('<font size="7.5">평가 대상</font>', mid_style)],
                 [Paragraph(f'<font size="11"><b>{_inline(target[0])}</b></font>', mid_style)],
                 [Paragraph(f'<font size="7.5">{_inline(target[2])}</font>', mid_style)]], colWidths=[mid_w - 2])
    mid.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), _hex(NAVY)),
                             ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    arrow = ParagraphStyle("arrow", parent=styles["cell"], alignment=1, textColor=_hex(MUTED))
    outer = Table([[left, Paragraph("◀", arrow), mid, Paragraph("▶", arrow), right]],
                  colWidths=[side_w, gap_w, mid_w, gap_w, side_w])
    outer.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                               ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    return outer


def _two_panels(block: list[str], styles: dict, width: float) -> Table:
    """차별점(강점) / 경쟁 리스크 2칸 패널. 표의 각 열을 목록으로 그린다."""
    rows = _md_rows(block)
    titles, body = rows[0], rows[1:]

    def panel(col: int, bar: str, w: float) -> Table:
        items = [r[col] for r in body if len(r) > col and r[col]]
        cells = [[Paragraph(f'<font color="{bar}"><b>{_inline(titles[col])}</b></font>', styles["cell"])]]
        cells += [[Paragraph(_inline(t), styles["bullet"], bulletText="•")] for t in items] or \
                 [[Paragraph(f'<font color="{MUTED}">자료 없음</font>', styles["cell"])]]
        box = Table(cells, colWidths=[w])
        box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), _hex(PANEL)),
                                 ("LINEBEFORE", (0, 0), (0, -1), 3, _hex(bar)),
                                 ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 2),
                                 ("BOTTOMPADDING", (0, 0), (-1, -1), 2), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        return box

    half = (width - 8) / 2
    outer = Table([[panel(0, GOOD, half - 2), "", panel(1, BAD, half - 2)]], colWidths=[half, 8, half])
    outer.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    return outer


# 경쟁 구도 그림·2칸 패널은 한 행짜리 표라 쪽을 넘겨 나뉘지 않는다. 카드가 많고 설명이 길어 한 쪽보다 높아지면
# LayoutError로 PDF 저장이 실패하므로, 그때는 이 값을 켜고 일반 표(쪽 나눔 가능)로 다시 그린다
_PLAIN_FIGURES = False


def content_flowables(lines: list[str], styles: dict, width: float, font: str) -> list:
    """제목을 뺀 내용 줄들을 PDF 조각(표·목록·문단)으로 바꾼다."""
    flow, i = [], 0
    while i < len(lines):
        line = lines[i].rstrip()
        if line.startswith("|"):  # 표는 연속된 '|' 줄을 한 덩어리로 모은다
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            first = _md_rows(block)[0][0] if _md_rows(block) else ""
            if _PLAIN_FIGURES:                 # 저장 재시도: 그림 대신 일반 표
                flow += [_table(block, styles, width, font), Spacer(1, 4)]
            elif first == LANDSCAPE_HEADER:    # 경쟁 구도 표 → 구도 그림 + 비교 근거 표
                flow += [Spacer(1, 2), _landscape(block, styles, width), Spacer(1, 6)]
                evidence = comparison_rows(block)
                if evidence:
                    flow += [_table(evidence, styles, width, font), Spacer(1, 6)]
            elif first == STRENGTH_HEADER:     # 차별점·경쟁 리스크 표 → 2칸 패널
                flow += [_two_panels(block, styles, width), Spacer(1, 6)]
            else:
                flow += [_table(block, styles, width, font), Spacer(1, 4)]
            continue
        if HR_PATTERN.fullmatch(line):  # '---' 구분선은 글자로 찍히지 않게 건너뛴다
            i += 1
            continue
        if re.fullmatch(r"\*\*[^*]+\*\*:?", line.strip()):      # '**핵심 근거**' 같은 소제목 줄
            flow.append(Paragraph(_inline(line.strip()), styles["label"]))
        elif line.startswith("  - "):
            flow.append(Paragraph(_inline(line[4:]), styles["sub"], bulletText="–"))
        elif line.startswith(("- ", "* ")):
            flow.append(Paragraph(_inline(line[2:]), styles["bullet"], bulletText="•"))
        elif line.strip():
            flow.append(Paragraph(_inline(line.strip()), styles["body"]))
        i += 1
    return flow


def parse_blocks(markdown: str) -> list[dict]:
    """마크다운을 제목 기준 덩어리 [{level, title, lines}]로 나눈다. level 0 = 첫 제목 앞 글."""
    blocks = [{"level": 0, "title": "", "lines": []}]
    for line in markdown.splitlines():
        m = HEADING_PATTERN.match(line)
        if m:
            blocks.append({"level": len(m.group(1)), "title": m.group(2).strip(), "lines": []})
        else:
            blocks[-1]["lines"].append(line)
    return blocks


def _heading2(title: str, styles: dict, width: float) -> Table:
    """장 제목: 좌측 색 띠가 있는 한 줄."""
    box = Table([[Paragraph(_inline(title), styles["h2"])]], colWidths=[width])
    box.setStyle(TableStyle([("LINEBEFORE", (0, 0), (0, 0), 3.5, _hex(NAVY)),
                             ("LEFTPADDING", (0, 0), (-1, -1), 7),
                             ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
    return box


def _panel(flowables: list, width: float, background: str, bar: str | None) -> Table:
    """색 배경 박스(SUMMARY, 검증 결과에 사용). bar가 있으면 왼쪽에 색 띠."""
    box = Table([[flowables]], colWidths=[width])
    style = [("BACKGROUND", (0, 0), (-1, -1), _hex(background)),
             ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
             ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]
    if bar:
        style.append(("LINEBEFORE", (0, 0), (0, 0), 3.5, _hex(bar)))
    box.setStyle(TableStyle(style))
    return box


def render_blocks(blocks: list[dict], font: str, bold: str, width: float, has_dashboard: bool) -> list:
    """덩어리별로 디자인을 달리해 PDF 조각 목록을 만든다."""
    normal, small = _styles(font, bold), _styles(font, bold, SMALL_SIZE)
    flow = []
    for block in blocks:
        level, title, lines = block["level"], block["title"], block["lines"]
        upper = title.upper()
        if level == 0:
            flow += content_flowables(lines, normal, width, font)
        elif level == 1:
            if not has_dashboard:  # 대시보드가 제목 밴드를 대신하므로 그때는 건너뛴다
                flow.append(Paragraph(_inline(title), normal["h1"]))
            flow += content_flowables(lines, normal, width, font)
        elif level == 2 and "SUMMARY" in upper:
            # SUMMARY는 간격 배율을 적용하지 않는다 (반 쪽 기준을 지키고 1쪽 대시보드가 밀리지 않게)
            fixed = _styles(font, bold, scaled=False)
            inner = [Paragraph("SUMMARY", fixed["h2"])] + content_flowables(lines, fixed, width - 24, font)
            flow += [Spacer(1, 4), _panel(inner, width, PANEL, NAVY)]
        elif level == 2 and "REFERENCE" in upper:
            heading, gap = _heading2(title, small, width), Spacer(1, _gap(HEADING_GAP))
            heading.keepWithNext = gap.keepWithNext = True  # 제목만 쪽 끝에 홀로 남지 않게
            # 강제로 새 쪽을 열면 한계점 뒤 검증 결과만 있는 반쪽짜리 쪽이 생겨, 공간이 부족할 때만 넘긴다
            flow += [CondPageBreak(USABLE_PAGE_HEIGHT * REFERENCE_ROOM), Spacer(1, _gap(SECTION_GAP)), heading, gap]
            flow += content_flowables(lines, small, width, font)
        elif level == 2:
            body = content_flowables(lines, normal, width, font)
            if "경쟁 구도" in title:
                # 경쟁 구도 그림·패널은 쪽을 나눌 수 없는 큰 덩어리라, 먼저 오면 앞 쪽 남은 공간이 통째로 비었다.
                # 해설 문단을 먼저 두어 남은 공간을 채우고 그림은 그 뒤에 둔다
                # 비교 불가·분야 제외 줄은 비교 근거라 그림 뒤에 둔다
                text = [f for f in body if not isinstance(f, (Table, Spacer))
                        and not getattr(f, "text", "").startswith((EXCLUDED_PREFIX, NOT_COMPARABLE_PREFIX))]
                body = text + [f for f in body if f not in text]
            opener = ([CondPageBreak(USABLE_PAGE_HEIGHT * PAGE_BREAK_ROOM)] if any(k in title for k in PAGE_BREAK_BEFORE)
                      else []) + [Spacer(1, _gap(SECTION_GAP))]
            flow += opener + [KeepTogether([_heading2(title, normal, width), Spacer(1, _gap(HEADING_GAP))] + body[:1])]
            flow += body[1:]
        elif level == 3 and title.startswith(NOTE_TITLE):
            inner = [Paragraph(_inline(title), small["label"])] + content_flowables(lines, small, width - 24, font)
            flow += [Spacer(1, 6), _panel(inner, width, NOTE_BG, MUTED)]
        elif level == 3:
            flow.append(Paragraph(_inline(title), small["h3"] if "REFERENCE" in upper else normal["h3"]))
            flow += content_flowables(lines, small if _is_reference_child(blocks, block) else normal, width, font)
    return flow


def _is_reference_child(blocks: list[dict], block: dict) -> bool:
    """이 소제목(###)이 REFERENCE 장 안에 있는가? (그렇다면 작은 글씨로 보여 준다)"""
    parent = ""
    for b in blocks:
        if b["level"] == 2:
            parent = b["title"].upper()
        if b is block:
            return "REFERENCE" in parent
    return False


# ── 결정 대시보드 (1쪽 위쪽) ─────────────────────────────────
def _first_value(data: dict, keys: tuple[str, ...]) -> str:
    return next((str(data[k]) for k in keys if data.get(k)), "")


def _title_band(company: dict, decision_text: str, report_date: str, styles: dict, font: str,
                bold: str, width: float) -> Table:
    """1쪽 맨 위 남색 제목 밴드."""
    name = company.get(COMPANY_NAME_KEY)
    title = f"{name} 투자 평가 보고서" if name else "투자 평가 보고서"   # 추천 없음이면 부제에 '추천 기업 없음'이 표시된다
    title_style = ParagraphStyle("band_t", fontName=bold, fontSize=17, leading=23, textColor=colors.white)
    sub_style = ParagraphStyle("band_s", fontName=font, fontSize=8.5, leading=12, textColor=_hex("#c9d6f2"))
    meta = " · ".join(v for v in (_first_value(company, SECTOR_KEYS), _first_value(company, STAGE_KEYS),
                                  decision_text, report_date) if v)
    band = Table([[Paragraph(_inline(title), title_style)],
                  [Paragraph(_inline(meta), sub_style)]], colWidths=[width])
    band.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), _hex(NAVY)),
                              ("LEFTPADDING", (0, 0), (-1, -1), 12), ("TOPPADDING", (0, 0), (0, 0), 9),
                              ("BOTTOMPADDING", (0, 1), (0, 1), 9), ("TOPPADDING", (0, 1), (0, 1), 0),
                              ("BOTTOMPADDING", (0, 0), (0, 0), 1)]))
    return band


def _score_card(total: float, decision: str, font: str, bold: str, w: float, h: float = DASH_HEIGHT) -> Drawing:
    """결정 배지 + 큰 환산 점수 + 70점 기준선이 있는 게이지."""
    d = Drawing(w, h)
    color = GOOD if decision == DECISION_INVEST else HOLD
    d.add(Rect(0, h - 30, 74, 24, rx=12, ry=12, fillColor=_hex(color), strokeColor=None))
    d.add(String(37, h - 22, decision, fontName=bold, fontSize=12, fillColor=colors.white, textAnchor="middle"))

    score_text = f"{total:.1f}"
    d.add(String(0, h - 86, score_text, fontName=bold, fontSize=42, fillColor=_hex(NAVY)))
    text_w = pdfmetrics.stringWidth(score_text, bold, 42)
    d.add(String(text_w + 5, h - 86, "/ 100", fontName=font, fontSize=12, fillColor=_hex(MUTED)))

    gauge_w, gauge_y = w - 14, h - 118
    d.add(Rect(0, gauge_y, gauge_w, 10, fillColor=_hex(LINE), strokeColor=None))
    d.add(Rect(0, gauge_y, gauge_w * min(total, 100) / 100, 10, fillColor=_hex(color), strokeColor=None))
    x = gauge_w * DECISION_THRESHOLD / 100  # 투자 기준선
    d.add(Line(x, gauge_y - 4, x, gauge_y + 14, strokeColor=_hex(NAVY), strokeWidth=1.6))
    d.add(String(x, gauge_y + 18, f"기준 {DECISION_THRESHOLD}", fontName=font, fontSize=7.5,
                 fillColor=_hex(NAVY), textAnchor="middle"))
    d.add(String(0, 6, "환산 점수 = Σ(항목 점수 ÷ 5 × 비중)", fontName=font, fontSize=7, fillColor=_hex(MUTED)))
    return d


def _radar(items: dict, font: str, bold: str, w: float, h: float = DASH_HEIGHT) -> Drawing:
    """6개 평가 항목 점수를 한눈에 보는 레이더 차트."""
    names = list(WEIGHTS)
    scores = [items[n]["score"] for n in names]
    cx, cy, radius = w / 2, h / 2 - 2, min(w / 2 - 66, h / 2 - 20)
    angle = lambda i: math.pi / 2 - 2 * math.pi * i / len(names)          # 12시 방향에서 시계 방향
    point = lambda i, r: (cx + r * math.cos(angle(i)), cy + r * math.sin(angle(i)))

    d = Drawing(w, h)
    for ring in range(1, SCORE_MAX + 1):  # 1~5점 눈금 다각형
        pts = [c for i in range(len(names)) for c in point(i, radius * ring / SCORE_MAX)]
        d.add(Polygon(pts, fillColor=None, strokeColor=_hex(LINE), strokeWidth=0.6))
    for i in range(len(names)):  # 축
        x, y = point(i, radius)
        d.add(Line(cx, cy, x, y, strokeColor=_hex(LINE), strokeWidth=0.6))

    data_pts = [c for i, s in enumerate(scores) for c in point(i, radius * s / SCORE_MAX)]
    d.add(Polygon(data_pts, fillColor=_hex(NAVY), fillOpacity=0.22, strokeColor=_hex(NAVY), strokeWidth=1.6))
    for i, s in enumerate(scores):
        x, y = point(i, radius * s / SCORE_MAX)
        d.add(Circle(x, y, 2.3, fillColor=_hex(NAVY), strokeColor=None))

    for i, (name, s) in enumerate(zip(names, scores)):  # 항목 이름과 점수
        x, y = point(i, radius + 9)
        cos, sin = math.cos(angle(i)), math.sin(angle(i))
        anchor = "middle" if abs(cos) < 0.3 else "start" if cos > 0 else "end"
        y += 4 if sin > 0.5 else -10 if sin < -0.5 else -3
        d.add(String(x, y, f"{name} {s}", fontName=font, fontSize=7.5, fillColor=_hex(INK), textAnchor=anchor))
    return d


def candidate_rows(state: dict) -> list[tuple[str, float | None, str]]:
    """평가 순서대로 (후보, 환산 점수, 상태). 상태: 투자 / 보류 / 조건 미충족 / 미평가.

    투자 결정이 나오면 그래프가 멈추므로 뒤 순번 후보는 점수가 없다('미평가').
    """
    rejected = {r.get("company"): r for r in state.get("rejected") or [] if isinstance(r, dict)}
    names = state.get("candidates") or list(rejected)
    selected = (state.get("company") or {}).get(COMPANY_NAME_KEY) if _is_recommend_mode(state) else None
    rows = []
    for name in names:
        if name == selected:
            rows.append((name, recompute_total(state.get("scores") or {}), DECISION_INVEST))
        elif name in rejected:
            total = rejected[name].get("total")
            rows.append((name, total, "보류") if isinstance(total, (int, float)) else (name, None, "조건 미충족"))
        else:
            rows.append((name, None, "미평가"))
    return rows


def _candidate_bars(state: dict, font: str, bold: str, w: float) -> Drawing | None:
    """후보별 환산 점수 막대(평가 순서)와 70점 기준선. 선정 기업은 초록, 보류는 주황, 점수 없는 후보는 글자로."""
    rows = candidate_rows(state)[:MAX_CANDIDATE_ROWS]
    if not any(total is not None for _, total, _ in rows):
        return None
    row_h, label_w, title_h = 20, 90, 14
    bar_w = w - label_w - 40
    top = row_h * len(rows)
    d = Drawing(w, top + 16 + title_h)
    d.add(String(0, top + 18, "후보별 환산 점수 (평가 순서)", fontName=bold, fontSize=8.5, fillColor=_hex(NAVY)))
    for i, (name, total, status) in enumerate(rows):
        y = top - (i + 1) * row_h + 6
        d.add(String(0, y + 2, str(name)[:10], fontName=bold if status == DECISION_INVEST else font,
                     fontSize=8.5, fillColor=_hex(INK)))
        d.add(Rect(label_w, y, bar_w, 10, fillColor=_hex(LINE), strokeColor=None))
        if total is None:
            d.add(String(label_w + 6, y + 2, status, fontName=font, fontSize=8, fillColor=_hex(MUTED)))
            continue
        color = GOOD if status == DECISION_INVEST else HOLD
        d.add(Rect(label_w, y, bar_w * min(total, 100) / 100, 10, fillColor=_hex(color), strokeColor=None))
        d.add(String(label_w + bar_w + 6, y + 2, f"{total:g}", fontName=bold, fontSize=8.5, fillColor=_hex(INK)))
    x = label_w + bar_w * DECISION_THRESHOLD / 100
    d.add(Line(x, 2, x, top + 4, strokeColor=_hex(NAVY), strokeWidth=1.4))
    d.add(String(x, top + 6, f"투자 기준 {DECISION_THRESHOLD}", fontName=font, fontSize=7.5,
                 fillColor=_hex(NAVY), textAnchor="middle"))
    return d


def _badge_strip(state: dict, report: str, styles: dict, width: float, per_company: bool = True) -> Table | None:
    """검증 배지 줄: 수치 검증 / 원문 대조(RAG) / (있으면) 주장 검증(LLM) / 인용 / 정보 부족.

    per_company=False(추천 기업 없음)면 한 기업 기준인 '인용'·'정보 부족' 배지는 뺀다.
    """
    result = state.get("verify_result")
    badges = []  # (제목, 값, 색)
    if result:
        problems = result.get("mismatches", [])
        is_rule = lambda m: not str(m.get("kind", "")).startswith(("claim", "rag"))
        bad_numbers = sum(1 for m in problems if m.get("kind") == "number")
        checked = result.get("checked", 0)
        value = f"{max(checked - bad_numbers, 0)}/{checked} 일치"
        others = sum(1 for m in problems if is_rule(m) and m.get("kind") != "number")
        if others:
            value += f" · 표기·인용 {others}건"
        badges.append(("수치 검증(State)", value, GOOD if not [m for m in problems if is_rule(m)] else BAD))

        rag = result.get("rag") or {}
        if rag.get("checked"):
            rag_bad = sum(1 for m in problems if str(m.get("kind", "")).startswith("rag"))
            found = rag["checked"] - rag.get("not_found", 0)
            badges.append(("원문 대조(RAG)", f"{found}/{rag['checked']} 원문 확인 · 불일치 {rag_bad}",
                           GOOD if not rag_bad else BAD))
        else:
            badges.append(("원문 대조(RAG)", "미실행", MUTED))

        if result.get("judged"):
            claims_bad = sum(1 for m in problems if m.get("kind") == "claim")
            badges.append(("주장 검증(LLM)", f"{result['judged'] - claims_bad}/{result['judged']} 근거 확인",
                           GOOD if not claims_bad else HOLD))
    else:
        badges.append(("수치 검증", "미검증", MUTED))
    if per_company:
        body = report.split("## REFERENCE")[0]
        badges.append(("인용", f"{len(set(CITATION_PATTERN.findall(body)))}건 연결", NAVY))
        lacking = len(find_insufficient_items(state.get("scores") or {}))
        # 채점에서 2점 상한을 받은 항목 수. 한계점의 '세부 정보 부족'과 헷갈리지 않게 '채점'을 붙인다
        badges.append(("정보 부족(채점)", f"{lacking}개 항목" if lacking else "없음", HOLD if lacking else GOOD))

    cells = [[Paragraph(f'<font size="7.5" color="{MUTED}">{title}</font><br/>'
                        f'<font color="{color}"><b>{_inline(value)}</b></font>', styles["cell"])
              for title, value, color in badges]]
    strip = Table(cells, colWidths=[width / len(badges)] * len(badges))
    style = [("BACKGROUND", (0, 0), (-1, -1), _hex(PANEL)),
             ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
             ("LEFTPADDING", (0, 0), (-1, -1), 9)]
    for col, (_, _, color) in enumerate(badges):
        style.append(("LINEBEFORE", (col, 0), (col, 0), 3, _hex(color)))
    strip.setStyle(TableStyle(style))
    return strip


def build_dashboard(state: dict, report: str, report_date: str, font: str, bold: str,
                    width: float) -> tuple[list, list]:
    """1쪽 대시보드를 (제목 밴드, 시각 자료) 두 부분으로 만든다.

    왜 둘로 나누나: 과제 조건상 SUMMARY가 맨 앞이어야 하므로,
    순서를 '제목 밴드 → SUMMARY → 시각 자료(배지·게이지·레이더)'로 끼워 넣기 위해서다.
    필요한 State가 없으면 ([], []) = 대시보드 생략.
    """
    scores, company = state.get("scores") or {}, state.get("company") or {}
    normal = _styles(font, bold)
    total, items = recompute_total(scores), score_items(scores)

    # 모양은 '점수가 있는가'가 아니라 '투자 추천인가'로 고른다. 모든 후보가 보류되면 State에는
    # 마지막 후보의 점수가 남아 있어서, 점수로 고르면 추천 없음 보고서가 그 후보의 보고서처럼 보인다
    if _is_recommend_mode(state) and total is not None and items:
        # 투자 추천: 선정 기업 게이지 + 레이더 → 검증 배지
        # (후보별 점수 막대는 넣지 않는다: 첫 투자 결정에서 평가를 멈춰 뒤 후보는 점수가 없다. 후보 이름은 SUMMARY 결론에 있다)
        decision = scores.get("decision", "미정")
        card_w = width * CARD_RATIO
        row = Table([[_score_card(total, decision, font, bold, card_w - 8),
                      _radar(items, font, bold, width - card_w)]], colWidths=[card_w, width - card_w])
        row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
        header = [_title_band(company, decision, report_date, normal, font, bold, width), Spacer(1, 4)]
        visuals = [Spacer(1, 10), row, Spacer(1, 6), _badge_strip(state, report, normal, width), Spacer(1, 4)]
        return header, visuals

    # 추천 기업 없음: 제목 밴드 + 후보별 점수 막대(모든 후보가 채점됨) + 검증 배지(정보 부족·인용은 뺀다)
    bars = _candidate_bars(state, font, bold, width)
    header = [_title_band({}, "추천 기업 없음", report_date, normal, font, bold, width), Spacer(1, 4)]
    visuals = [Spacer(1, 10), bars, Spacer(1, 4)] if bars else []
    if state.get("verify_result"):
        visuals += [Spacer(1, 6), _badge_strip(state, report, normal, width, per_company=False), Spacer(1, 4)]
    return header, visuals


def _split_at_summary(blocks: list[dict]) -> tuple[list[dict], list[dict]]:
    """덩어리 목록을 'SUMMARY까지'와 '그 뒤'로 나눈다. SUMMARY가 없으면 (전부, [])."""
    for i, block in enumerate(blocks):
        if block["level"] == 2 and "SUMMARY" in block["title"].upper():
            return blocks[: i + 1], blocks[i + 1:]
    return blocks, []


def _flow_height(flowables: list, width: float) -> float:
    """PDF 조각들이 실제로 차지할 세로 높이(pt)를 잰다. (글자 수 추정이 아니라 실제 조판 결과)"""
    return sum(f.wrap(width, A4[1])[1] + f.getSpaceBefore() + f.getSpaceAfter() for f in flowables)


def measure_summary_ratio(report: str) -> float:
    """SUMMARY 박스가 한 쪽 본문 높이의 몇 %를 차지하는지 잰다. (과제 조건: 1/2 페이지 이내)"""
    font, bold = register_korean_font()
    width = A4[0] - 2 * PAGE_MARGIN
    summary_blocks = [b for b in parse_blocks(report)
                      if b["level"] == 2 and "SUMMARY" in b["title"].upper()]
    if not summary_blocks:
        return 0.0
    height = _flow_height(render_blocks(summary_blocks, font, bold, width, True), width)
    return height / USABLE_PAGE_HEIGHT


# ── 쪽 머리글·바닥글 ─────────────────────────────────────────
def _page_decorations(font: str, label: str, counter: dict):
    """쪽마다 호출되는 그리기 함수 2개(첫 쪽용, 이후 쪽용)를 만든다. 쪽 수도 함께 센다."""
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(_hex(LINE))
        canvas.line(PAGE_MARGIN, 11 * mm, A4[0] - PAGE_MARGIN, 11 * mm)
        canvas.setFont(font, 7.5)
        canvas.setFillColor(_hex(MUTED))
        canvas.drawString(PAGE_MARGIN, 7 * mm, label)
        canvas.drawRightString(A4[0] - PAGE_MARGIN, 7 * mm, f"{doc.page}쪽")
        canvas.restoreState()
        counter["n"] += 1

    def later(canvas, doc):
        canvas.saveState()
        canvas.setFont(font, 7.5)
        canvas.setFillColor(_hex(MUTED))
        canvas.drawString(PAGE_MARGIN, A4[1] - 10 * mm, label)
        canvas.setStrokeColor(_hex(NAVY))
        canvas.setLineWidth(1.2)
        canvas.line(PAGE_MARGIN, A4[1] - 11.5 * mm, A4[0] - PAGE_MARGIN, A4[1] - 11.5 * mm)
        canvas.restoreState()
        footer(canvas, doc)

    return footer, later


# ── PDF 저장 ─────────────────────────────────────────────────
class _EndMarker(Flowable):
    """맨 끝에 넣어 마지막 쪽에 남은 높이를 잰다. 높이가 0이라 항상 들어가고, wrap이 받는 높이가 그 쪽에 남은 공간이다."""

    def __init__(self, sink: dict):
        super().__init__()
        self.sink = sink

    def wrap(self, avail_width, avail_height):
        self.sink["left"] = avail_height
        return 0, 0

    def draw(self):
        pass


def choose_layout_scale(trials: list[tuple[float, int, float]]) -> float:
    """조판해 본 (배율, 쪽수, 마지막 쪽 채움 비율) 중에서 고른다.
    ① 최소~최대 쪽이고 마지막 쪽이 LAST_PAGE_MIN_FILL 이상 찬 가장 작은 배율
    ② 없으면 최소~최대 쪽 중 마지막 쪽이 가장 많이 찬 배율
    ③ 최소 쪽에 못 미치면 최대 쪽 이내에서 쪽수가 가장 많고 마지막 쪽이 가장 많이 찬 배율"""
    fit = [t for t in trials if MIN_PAGES <= t[1] <= MAX_PAGES]
    good = [t for t in fit if t[2] >= LAST_PAGE_MIN_FILL]
    if good:
        return min(good)[0]
    if fit:
        return max(fit, key=lambda t: t[2])[0]
    within = [t for t in trials if t[1] <= MAX_PAGES] or trials
    return max(within, key=lambda t: (t[1], t[2]))[0]


def export_report_pdf(report: str, output_dir: Path = OUTPUT_DIR, filename: str | None = None,
                      state: dict | None = None, report_date: str | None = None) -> tuple[Path, int]:
    """보고서를 PDF로 저장하고 (저장 경로, 페이지 수)를 돌려준다. MIN_PAGES~MAX_PAGES를 벗어나면 경고한다.

    state를 함께 주면 1쪽에 '제목 밴드 → SUMMARY → 결정 대시보드' 순서로 그린다. (없으면 본문만)
    SUMMARY가 실제 조판 기준으로 반 페이지를 넘으면 경고한다.
    """
    if not report or not report.strip():
        raise ValueError("빈 보고서는 PDF로 저장할 수 없습니다.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / (filename or build_output_filename())
    report_date = report_date or date.today().isoformat()

    font, bold = register_korean_font()
    width = A4[0] - 2 * PAGE_MARGIN
    state = state or {}

    # 순서: 제목 밴드 → SUMMARY → 시각 자료(대시보드) → 1장~ → REFERENCE
    has_result = any(state.get(k) for k in ("scores", "rejected", "candidates"))   # 평가 결과가 있으면 대시보드

    def make_story() -> list:  # 조판에 한 번 쓴 조각은 다시 쓸 수 없어, 재시도 때 새로 만든다
        header, visuals = (build_dashboard(state, report, report_date, font, bold, width)
                           if has_result else ([], []))
        until_summary, after_summary = _split_at_summary(parse_blocks(report))
        has_band = bool(header)
        rest = render_blocks(after_summary, font, bold, width, has_band)
        if rest and not isinstance(rest[0], PageBreak):
            # 1쪽은 제목 밴드·SUMMARY·대시보드만. 대시보드 일부가 2쪽으로 넘어가 있으면 또 넘기지 않는다
            # (강제로 넘기면 대시보드 조각만 있는 빈 쪽이 생겼다)
            rest = [CondPageBreak(USABLE_PAGE_HEIGHT * BODY_START_ROOM)] + rest
        return header + render_blocks(until_summary, font, bold, width, has_band) + visuals + rest

    ratio = measure_summary_ratio(report)  # SUMMARY가 반 페이지를 넘는지 실제 높이로 확인
    if ratio > SUMMARY_MAX_PAGE_RATIO:
        logger.warning("[PDF] SUMMARY가 한 쪽의 %.0f%%를 차지해 반 페이지 기준을 넘었습니다.", ratio * 100)
    else:
        logger.info("[PDF] SUMMARY 높이: 한 쪽의 %.0f%%", ratio * 100)

    # 추천 기업이 없으면 마지막으로 평가한 기업 이름을 바닥글에 넣지 않는다
    company_name = (state.get("company") or {}).get(COMPANY_NAME_KEY, "") if _is_recommend_mode(state) else ""
    label = " · ".join(v for v in ("AI 스타트업 투자 평가", company_name) if v)
    def build() -> dict:
        counter = {"n": 0, "left": 0.0}
        first, later = _page_decorations(font, label, counter)
        doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=PAGE_MARGIN, rightMargin=PAGE_MARGIN,
                                topMargin=PAGE_MARGIN, bottomMargin=PAGE_MARGIN + BOTTOM_EXTRA, title=path.stem)
        doc.build(make_story() + [_EndMarker(counter)], onFirstPage=first, onLaterPages=later)
        counter["fill"] = 1 - counter["left"] / USABLE_PAGE_HEIGHT   # 마지막 쪽이 찬 비율
        return counter

    global _PLAIN_FIGURES, _LAYOUT_SCALE

    def build_once() -> dict:
        global _PLAIN_FIGURES
        try:
            return build()
        except LayoutError as e:
            logger.warning("[PDF] 경쟁 구도 그림이 한 쪽을 넘어 일반 표로 다시 저장합니다: %s", str(e)[:120])
            _PLAIN_FIGURES = True
            try:
                return build()
            finally:
                _PLAIN_FIGURES = False

    # 간격 배율을 차례로 키워 조판해 보고, 최소~최대 쪽이면서 마지막 쪽이 충분히 찬 배율을 고른다 (choose_layout_scale)
    try:
        trials = []
        for scale in LAYOUT_SCALES:
            _LAYOUT_SCALE = scale
            counter = build_once()
            trials.append((scale, counter["n"], counter["fill"]))
            if counter["n"] > MAX_PAGES or (counter["n"] >= MIN_PAGES and counter["fill"] >= LAST_PAGE_MIN_FILL):
                break   # 넘쳤거나 조건을 채웠으면 더 키우지 않는다
        used = choose_layout_scale(trials)
        if used != trials[-1][0]:
            _LAYOUT_SCALE = used
            counter = build_once()   # 고른 배율로 다시 저장 (파일은 마지막으로 조판한 것이 남는다)
    finally:
        _LAYOUT_SCALE = 1.0
    logger.info("[PDF] 간격 배율 %.2f로 조판 (%d쪽, 마지막 쪽 %.0f%% 참). 시도: %s", used, counter["n"], counter["fill"] * 100,
                ", ".join(f"x{t[0]:g}={t[1]}쪽/{t[2]:.0%}" for t in trials))

    pages = counter["n"]
    if pages < MIN_PAGES:
        logger.warning("[PDF] %d장으로 최소 %d장에 못 미칩니다 (간격 배율 최대 %.2f).", pages, MIN_PAGES, used)
    if pages > MAX_PAGES:
        logger.warning("[PDF] %d장으로 과제 기준(%d장 이내)을 넘었습니다. 본문을 줄이세요.", pages, MAX_PAGES)
    else:
        logger.info("[PDF] 저장 완료: %s (%d장)", path, pages)
    return path, pages

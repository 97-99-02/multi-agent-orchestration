"""웹 검색 공통 도우미 (Tavily). 검색 결과를 state.Source 형식 출처로 바꾸는 함수도 둔다.

2차 출처(블로그·뉴스레터·포털 재전송 등)는 그대로 쓰지 않고 원본을 찾아 내용을 확인한 뒤 원본으로 바꾼다
(resolve_secondary). 블로그가 옮겨 적은 수치는 원본과 시점·표기가 달라질 수 있어서다.
"""
import html as html_lib
import logging
import re
import threading
import time
import urllib.request
from datetime import date, timezone
from email.utils import parsedate_to_datetime
from functools import lru_cache
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo

from langchain_tavily import TavilySearch

logger = logging.getLogger(__name__)


def search(query: str, max_results: int = 8, topic: str = "news") -> list[dict]:
    """Tavily 검색 결과 리스트. 각 항목: title, url, content, published_date.
    2차 출처는 원본으로 바꾸거나(확인된 경우) 뺀다 (resolve_secondary)."""
    return resolve_secondary(_tavily(query, max_results, topic))


# 429(분당 요청 한도)일 때 기다렸다 다시 검색하는 간격(초). 분석 노드 4개가 동시에 검색하다 개발용 키 한도에 걸려
# 기업 하나가 근거 없이 채점되고 다음 후보 탐색까지 실패한 적이 있다. 한도는 분 단위라 조금 기다리면 풀린다
TAVILY_RETRY_WAITS = (5, 15, 30)


def _tavily(query: str, max_results: int = 8, topic: str = "news", retry: bool = True) -> list[dict]:
    """Tavily 검색 원본 결과 (2차 출처 처리 전). retry=False면 429에서 기다리지 않고 바로 실패한다."""
    for wait in (*(TAVILY_RETRY_WAITS if retry else ()), None):
        response = TavilySearch(max_results=max_results, topic=topic).invoke({"query": query})
        if wait is not None and isinstance(response, dict) and "429" in str(response.get("error", "")):
            logger.warning("[웹] Tavily 요청 한도(429) → %d초 뒤 다시 검색: %s", wait, query)
            time.sleep(wait)
            continue
        break
    # 결과가 0건이면 langchain-tavily가 ToolException을 문자열로 돌려준다(handle_tool_error=True).
    # 작은 스타트업의 좁은 검색어에서 자주 생기며, 오류가 아니라 '결과 없음'으로 처리한다
    if isinstance(response, str):
        return []
    # 사용량 초과(Error 432) 등은 예외가 아니라 {"error": ...}로 돌아온다. 빈 결과로 넘기면 모든 에이전트가
    # 근거 없이 계속 진행하므로 여기서 멈춘다
    if isinstance(response, dict) and response.get("error"):
        raise RuntimeError(f"Tavily 검색 실패: {response['error']}")
    return response.get("results", [])


def search_many(queries: list[str], max_results: int = 8, topic: str = "news") -> list[dict]:
    """여러 검색어 결과를 url 기준으로 중복 제거해 합친다."""
    seen, merged = set(), []
    for q in queries:
        for r in search(q, max_results, topic):
            if r["url"] not in seen:
                seen.add(r["url"])
                merged.append(r)
    return merged


def format_results(results: list[dict], limit: int = 1200) -> str:
    """LLM 프롬프트에 넣을 형태. [번호]로 결과를 구분해 근거 url을 고를 수 있게 한다.
    게시일을 함께 넣는다. 날짜가 없으면 LLM이 "가장 최근 기사를 따른다"는 규칙을 지킬 수 없어,
    2026년 시리즈 B 기사 본문에 나온 '2024년 5월 시리즈 A' 날짜를 시리즈 B 시기로 뽑은 적이 있다."""
    return "\n\n".join(
        f"[{i}] {r['title']} ({source_date(r) or '날짜 미상'})\nURL: {r['url']}\n{r['content'][:limit]}"
        for i, r in enumerate(results)
    )


# 기사 URL에 들어 있는 날짜: /2024/02/18/ 형식, 또는 20240104083305처럼 날짜로 시작하는 10자리 이상 기사 번호
_URL_DATE_PATH = re.compile(r"/(20\d{2})/(\d{2})/(\d{2})(?:/|$)")
_URL_DATE_ID = re.compile(r"[=/](20\d{2})(\d{2})(\d{2})\d{2,}(?!\d)")


def date_from_url(url: str) -> str:
    """URL에서 게시일(YYYY-MM-DD)을 찾는다. 달력에 없는 날짜나 미래 날짜는 기사 번호로 보고 버린다."""
    for pattern in (_URL_DATE_PATH, _URL_DATE_ID):
        for y, m, d in pattern.findall(url):
            try:
                found = date(int(y), int(m), int(d))
            except ValueError:
                continue
            if found <= date.today():
                return found.isoformat()
    return ""


# 기사 페이지의 게시일 메타 태그 (article:published_time, JSON-LD datePublished, 다음 뉴스의 regdate 등).
# 매일경제TV는 article:published로 끝나는 이름을 쓴다
_META_DATE = re.compile(r"""(?:article:published(?:_time)?|datePublished|pubdate|publish-date|dateCreated|regdate)["']?\s*"""
                        r"""(?:content|:)\s*=?\s*["'](20\d{2})[-./]?(\d{2})[-./]?(\d{2})""", re.I)
# 메타 태그가 없는 언론사는 본문 머리에 '입력 : 2026.06.03', '입력시간 | 2026.07.04'처럼 적는다 (이데일리TV)
_TEXT_DATE = re.compile(r"입력(?:시간|일)?\s*[:|]?\s*(20\d{2})[-.](\d{1,2})[-.](\d{1,2})")
# 매일경제는 'Mozilla/5.0'만 보내면 403으로 막는다
_BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")


@lru_cache(maxsize=512)
def date_from_page(url: str) -> str:
    """기사 페이지를 읽어 게시일 메타 태그(없으면 본문의 '입력' 날짜)에서 날짜를 찾는다. 못 읽거나 없으면 빈 문자열.
    Tavily 게시일도 URL 날짜도 없어 보고서 REFERENCE에 '날짜 미상'이 많이 남았다. 메타 태그를 확인한
    기사는 Tavily 게시일과 모두 일치해 같은 기준으로 쓴다. 문서용 예약 도메인(example.org 등)은 요청하지 않는다."""
    host = urlparse(url).hostname or ""
    if not url.startswith(("http://", "https://")) or host.split(".")[-2:-1] == ["example"]:
        return ""
    try:
        request = urllib.request.Request(url, headers={"User-Agent": _BROWSER_UA})
        with urllib.request.urlopen(request, timeout=4) as response:
            html = response.read(300_000).decode("utf-8", "ignore")
    except Exception:  # noqa: BLE001 - 날짜를 못 읽어도 출처 기록은 계속한다
        return ""
    return date_in_page(html)


def date_in_page(html: str) -> str:
    """페이지 HTML에서 게시일을 찾는다. 메타 태그를 먼저 보고, 없을 때만 본문의 '입력' 날짜를 쓴다."""
    for pattern in (_META_DATE, _TEXT_DATE):
        for y, m, d in pattern.findall(html):
            try:
                found = date(int(y), int(m), int(d))
            except ValueError:
                continue
            if found <= date.today():
                return found.isoformat()
    return ""


def source_date(r: dict, read_page: bool = False) -> str:
    """검색 결과의 게시일. Tavily 게시일(뉴스 검색) → URL 속 날짜 → (read_page=True면) 기사 페이지 메타 태그 순.
    페이지 읽기는 시간이 들어, 출처로 기록할 때(to_source)만 한다. 모두 없으면 빈 문자열."""
    try:
        published = parsedate_to_datetime(r["published_date"])
        if published.tzinfo is None:  # '-0000' 표기는 시간대 없는 UTC로 돌아온다
            published = published.replace(tzinfo=timezone.utc)
        day = published.astimezone(ZoneInfo("Asia/Seoul")).date()  # UTC로 자르면 한국 오전 기사가 전날로 찍힌다
        if day <= date.today():  # 미래 날짜는 게시일로 쓰지 않는다
            return day.isoformat()
    except (KeyError, TypeError, ValueError):
        pass  # 일반 웹 검색은 게시일을 주지 않는다
    return date_from_url(r.get("url", "")) or (date_from_page(r.get("url", "")) if read_page else "")


def to_source(r: dict, company: str, node: str) -> dict:
    """Tavily 결과 하나를 Source 형식으로. 언론사는 제목 끝 ' - 언론사' 에서 추출한다."""
    title, _, publisher = r["title"].rpartition(" - ")
    return {
        "company": company,
        "node": node,
        "kind": "web",
        "title": (title or r["title"]).strip(),
        "publisher": publisher.strip() if title else "",
        "date": source_date(r, read_page=True),
        "url": r["url"],
        "snippet": r["content"][:800],
    }


# ═════════════════════════════════════════════════════════════
# 2차 출처 → 원본 확인
# ═════════════════════════════════════════════════════════════
# 블로그·뉴스레터·위키 등: 남의 기사를 옮겨 적거나 요약한 글. 원본을 확인하지 못하면 근거로 쓰지 않는다
# (블로그가 1년 전 투자 기사 수치를 최신처럼 옮겨 보고서 수치가 서로 어긋난 적이 있다)
SECONDARY_HOSTS = ("tistory.com", "blog.naver.com", "post.naver.com", "cafe.naver.com", "contents.premium.naver.com",
                   "brunch.co.kr", "velog.io", "medium.com", "blogspot.com", "wordpress.com", "cafe.daum.net",
                   "namu.wiki", "wikipedia.org", "demoday.co.kr",
                   # SNS 게시물: 기사·공시를 옮겨 적은 글이다 (인스타그램 게시물이 실적 점수의 유일한 출처가 된 적이 있다)
                   "instagram.com", "facebook.com", "linkedin.com", "x.com", "twitter.com", "threads.net",
                   "youtube.com")
# 포털 재전송: 언론사 기사를 그대로 옮긴 페이지. 원본 기사로 바꾸되, 원본을 못 찾아도 내용은 언론사 기사라 그대로 둔다
PORTAL_HOSTS = ("v.daum.net", "news.daum.net", "news.nate.com", "n.news.naver.com", "news.naver.com", "news.zum.com",
                "msn.com")
# 원본으로 볼 수 없는 링크 (공유·광고·검색·SNS)
_NOT_ORIGINAL = ("facebook.com", "twitter.com", "x.com", "instagram.com", "youtube.com", "kakao.com", "google.com", "google.co.kr",
                 "apple.com", "naver.com", "daum.net", "nate.com", "tistory.com", "linkedin.com")
_ANCHOR = re.compile(r"""<a\b[^>]*?href=["']([^"']+)["'][^>]*>(.*?)</a>""", re.S | re.I)
_ORIGINAL_TEXT = re.compile(r"^\s*(?:기사\s*)?원문(?:\s*보기)?\s*$|원문\s*기사|기사\s*원문|^\s*출처\b|^\s*source\b", re.I)
_SOURCE_CONTEXT = re.compile(r"(?:출처|원문|source|via)\s*[:：]?\s*$", re.I)   # 링크 바로 앞에 '출처:'가 있는 경우
_FACT = re.compile(r"\d[\d,.]*\s*(?:조|억|만|천)?\s*(?:원|달러|%|배|명|개|나노|nm|TOPS|W)|\d{4}년|\d{2,}[\d,.]*")
_WORD = re.compile(r"[가-힣A-Za-z][가-힣A-Za-z0-9]{1,}")
MAX_ORIGINAL_CANDIDATES = 2   # 2차 출처 하나에서 따라가 볼 원본 링크 수
# 원문 링크가 없는 포털 재전송의 원본을 찾으려고 더 하는 제목 검색 수 (실행 한 번 기준). 검색 호출이 늘어
# 검색 API 분당 한도(429)에 걸린 적이 있어 상한을 두고, 넘으면 포털 기사를 그대로 둔다
TITLE_SEARCH_LIMIT = 8
_title_searches = 0
_title_lock = threading.Lock()
EXCERPT_CHARS = 1500          # 원본에서 옮길 본문 길이 (Tavily content와 비슷한 분량)


def _host_in(url: str, hosts: tuple) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == h or host.endswith("." + h) for h in hosts)


def is_secondary(url: str) -> bool:
    return _host_in(url, SECONDARY_HOSTS)


def is_portal(url: str) -> bool:
    return _host_in(url, PORTAL_HOSTS)


@lru_cache(maxsize=256)
def fetch_page(url: str) -> tuple[str, str]:
    """(최종 url, HTML). 못 읽으면 ("", ""). 한국 언론사는 EUC-KR 페이지가 있어 charset을 따른다."""
    host = urlparse(url).hostname or ""
    if not url.startswith(("http://", "https://")) or host.split(".")[-2:-1] == ["example"]:
        return "", ""
    try:
        request = urllib.request.Request(url, headers={"User-Agent": _BROWSER_UA})
        with urllib.request.urlopen(request, timeout=6) as response:
            raw = response.read(800_000)
            final = response.geturl()
            charset = response.headers.get_content_charset()
    except Exception:  # noqa: BLE001 - 원본을 못 읽으면 확인 실패로 처리한다
        return "", ""
    if not charset:
        m = re.search(rb"""charset=["']?([\w-]+)""", raw[:4000])
        charset = m.group(1).decode() if m else "utf-8"
    try:
        return final, raw.decode(charset, "ignore")
    except LookupError:
        return final, raw.decode("utf-8", "ignore")


def page_text(html: str) -> str:
    """HTML에서 본문 글자만 (스크립트·스타일 제거, 태그 → 공백)."""
    html = re.sub(r"(?is)<(script|style|noscript|template)\b.*?</\1>", " ", html)
    return re.sub(r"\s+", " ", html_lib.unescape(re.sub(r"(?s)<[^>]+>", " ", html))).strip()


def page_title(html: str) -> str:
    m = (re.search(r"""<meta[^>]+property=["']og:title["'][^>]+content=["']([^"']+)""", html, re.I)
         or re.search(r"(?is)<title[^>]*>(.*?)</title>", html))
    return html_lib.unescape(m.group(1)).strip() if m else ""


def original_links(url: str, html: str) -> list[str]:
    """2차 출처 페이지에서 원본 기사로 보이는 링크. '원문', '출처'라고 적힌 링크만 따른다 (관련 기사 목록은 따르지 않는다)."""
    host = urlparse(url).hostname
    found = []
    for m in _ANCHOR.finditer(html):
        href = html_lib.unescape(urljoin(url, m.group(1).strip()))
        text = re.sub(r"(?s)<[^>]+>", "", m.group(2))
        # 링크 바로 앞 글자. 앞 링크의 글자('원문')가 섞이지 않게 마지막 </a> 뒤만 본다
        before = re.split(r"(?i)</a>", html[max(0, m.start() - 80):m.start()])[-1]
        before = re.sub(r"(?s)<[^>]+>", " ", before)
        if (not href.startswith(("http://", "https://")) or urlparse(href).hostname == host
                or _host_in(href, _NOT_ORIGINAL) or is_secondary(href) or is_portal(href)):
            continue
        if _ORIGINAL_TEXT.search(text) or _SOURCE_CONTEXT.search(before):
            found.append(href)
    return list(dict.fromkeys(found))


def _norm_fact(s: str) -> str:
    return re.sub(r"[\s,]", "", s)


def confirms(secondary: str, original: str) -> bool:
    """원본 본문이 2차 출처 내용을 뒷받침하는가. 수치가 2개 이상이면 절반 이상이 원본에 있어야 하고,
    수치가 적으면 단어의 40% 이상이 원본에 있어야 한다 (제목만 같고 내용이 다른 페이지를 거른다)."""
    target = _norm_fact(original)
    facts = {_norm_fact(f) for f in _FACT.findall(secondary)} - {""}
    if len(facts) >= 2:
        return sum(f in target for f in facts) * 2 >= len(facts)
    words = set(_WORD.findall(secondary))
    return bool(words) and sum(w in original for w in words) / len(words) >= 0.4


def excerpt(secondary: str, original: str, limit: int = EXCERPT_CHARS) -> str:
    """원본 본문에서 2차 출처의 수치·단어가 나온 부분을 앞뒤로 잘라 이어 붙인다 (없으면 본문 앞부분)."""
    spans = []
    for f in sorted({f.strip() for f in _FACT.findall(secondary)} | set(_WORD.findall(secondary)[:20]), key=len, reverse=True):
        pos = original.find(f)
        if pos >= 0 and len(f) >= 2:
            spans.append((max(0, pos - 150), pos + len(f) + 150))
    if not spans:
        return original[:limit]
    spans.sort()
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return " [...] ".join(original[s:e].strip() for s, e in merged)[:limit]


def _original_by_title(title: str) -> list[str]:
    """원문 링크가 없는 포털 재전송: 같은 제목 기사를 검색해 원본 언론사 기사 후보를 찾는다."""
    global _title_searches
    core = re.sub(r"\s*[-|:]\s*[^-|:]+$", "", title).strip() or title   # 제목 끝 ' - 언론사' 제거
    if len(core) < 8:
        return []
    with _title_lock:   # 분석 노드가 병렬로 검색하므로 상한 계산을 잠근다
        if _title_searches >= TITLE_SEARCH_LIMIT:
            return []
        _title_searches += 1
    try:   # 보조 검색이라 429에서 기다리지 않는다 (본 검색의 한도를 쓰지 않게)
        results = _tavily(core, max_results=5, topic="general", retry=False)
    except Exception:  # noqa: BLE001 - 검색 실패는 원본 미확인으로 처리한다
        return []
    words = set(_WORD.findall(core))
    out = []
    for r in results:
        if is_secondary(r["url"]) or is_portal(r["url"]):
            continue
        if words and sum(w in r.get("title", "") for w in words) / len(words) >= 0.6:
            out.append(r["url"])
    return out


@lru_cache(maxsize=256)
def find_original(url: str, title: str, content: str) -> dict | None:
    """2차 출처의 원본 기사를 찾아 내용을 확인한다. 확인되면 원본으로 만든 검색 결과, 아니면 None."""
    final, html = fetch_page(url)
    candidates = original_links(final or url, html) if html else []
    if not candidates and is_portal(url):
        candidates = _original_by_title(page_title(html) if html else title)
    for link in candidates[:MAX_ORIGINAL_CANDIDATES]:
        o_url, o_html = fetch_page(link)
        if not o_html:
            continue
        text = page_text(o_html)
        if confirms(content, text):
            return {"title": page_title(o_html) or title, "url": o_url or link,
                    "content": excerpt(content, text), "via": url}
    return None


def resolve_secondary(results: list[dict]) -> list[dict]:
    """2차 출처를 원본으로 바꾼다. 원본을 확인하지 못한 블로그류는 빼고, 포털 재전송은 그대로 둔다."""
    out = []
    for r in results:
        url = r.get("url", "")
        if not (is_secondary(url) or is_portal(url)):
            out.append(r)
            continue
        original = find_original(url, r.get("title", ""), r.get("content", ""))
        if original:
            logger.info("[웹] 2차 출처 → 원본 확인: %s → %s", url, original["url"])
            out.append(original)
        elif is_portal(url):
            out.append(r)
        else:
            logger.info("[웹] 2차 출처의 원본을 확인하지 못해 제외: %s", url)
    return out

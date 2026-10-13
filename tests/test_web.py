"""웹 검색 공통 도우미 테스트 (API 키·네트워크 없이 실행): uv run --with pytest python -m pytest tests/test_web.py -q"""
from agents.web import date_from_url, date_in_page


def test_page_date_from_meta_tags():
    # 실제 언론사 페이지에서 날짜가 들어 있던 모양 (매일경제·뉴스1 / 매일경제TV / 다음 뉴스)
    assert date_in_page('<meta property="article:published_time" content="2026-06-03T10:00:00+09:00">') == "2026-06-03"
    assert date_in_page("<meta property='article:published'  content='2026-06-03 09:12'>") == "2026-06-03"
    assert date_in_page('{"title":"기사", regdate:"2025-05-15T09:00:00"}') == "2025-05-15"


def test_page_date_from_body_text_only_without_meta():
    assert date_in_page("<span>입력시간 | 2026.07.04 09:00</span>") == "2026-07-04"
    assert date_in_page("<span>입력 : 2026.06.03 11:20</span>") == "2026-06-03"
    # 메타 태그가 있으면 본문 날짜보다 먼저 쓴다
    html = '<span>입력 2025.01.01</span><meta property="article:published_time" content="2026-06-03">'
    assert date_in_page(html) == "2026-06-03"


def test_page_date_rejects_future_and_invalid():
    assert date_in_page('<meta property="article:published_time" content="2999-01-01">') == ""
    assert date_in_page("<span>입력 2026.13.40</span>") == ""
    assert date_in_page("<p>2026년 6월 3일 투자 유치</p>") == ""   # '입력'·메타 태그가 없는 본문 날짜는 쓰지 않는다


def test_url_date():
    assert date_from_url("https://wowtale.net/2026/05/30/259381") == "2026-05-30"
    assert date_from_url("https://www.etnews.com/20260601000245") == "2026-06-01"
    assert date_from_url("https://www.mk.co.kr/news/business/12064949") == ""


# ---------- 2차 출처 → 원본 확인 (네트워크는 가짜 페이지로 대신한다) ----------
import pytest  # noqa: E402

from agents import web  # noqa: E402

BLOG = "https://someone.tistory.com/546"
PORTAL = "https://news.nate.com/view/20260525n02042"
ORIGINAL = "https://www.news1.kr/industry/6173053"
CLAIM = "엑시나는 2000억 원 규모 시리즈 B를 마무리했고 기업가치는 8500억 원이다."


@pytest.fixture
def pages(monkeypatch):
    """url → HTML 가짜 페이지. find_original·fetch_page 캐시는 테스트마다 비운다."""
    site = {}
    monkeypatch.setattr(web, "fetch_page", lambda url: (url, site.get(url, "")))
    monkeypatch.setattr(web, "_tavily", lambda *a, **k: [])
    web.find_original.cache_clear()
    yield site
    web.find_original.cache_clear()


def test_secondary_and_portal_hosts():
    assert web.is_secondary("https://demoday.tistory.com/1") and web.is_secondary("https://m.blog.naver.com/x/1")
    assert web.is_portal("https://v.daum.net/v/abc") and web.is_portal(PORTAL)
    assert not web.is_secondary(ORIGINAL) and not web.is_portal(ORIGINAL)


def test_original_links_follow_only_source_links():
    html = (f'<a href="{ORIGINAL}">원문</a> <a href="https://www.news1.kr/other/1">다른 기사 제목</a>'
            '<a href="https://twitter.com/news1">원문</a> 출처: <a href="https://www.etnews.com/2026">전자신문</a>')
    assert web.original_links(PORTAL, html) == [ORIGINAL, "https://www.etnews.com/2026"]


def test_confirms_needs_matching_facts():
    assert web.confirms(CLAIM, "엑시나가 약 2,000억 원 시리즈 B 투자를 유치했다. 기업가치 8500억 원으로 평가됐다.")
    assert not web.confirms(CLAIM, "엑시나가 1500억 원 투자를 추진한다. 기업가치는 8000억 원 수준.")


def test_blog_is_replaced_by_confirmed_original(pages):
    pages[BLOG] = f'<p>{CLAIM}</p> 출처: <a href="{ORIGINAL}">뉴스1</a>'
    pages[ORIGINAL] = ('<meta property="og:title" content="엑시나 2000억 시리즈 B"><p>메뉴</p>'
                       '<p>엑시나는 2000억 원 규모 시리즈 B 투자를 마무리했다. 기업가치는 8500억 원이다.</p>')
    out = web.resolve_secondary([{"url": BLOG, "title": "블로그 분석", "content": CLAIM}])
    assert len(out) == 1 and out[0]["url"] == ORIGINAL and out[0]["via"] == BLOG
    assert out[0]["title"] == "엑시나 2000억 시리즈 B" and "8500억 원" in out[0]["content"]


def test_blog_without_confirmed_original_is_dropped(pages):
    pages[BLOG] = f'<p>{CLAIM}</p> 출처: <a href="{ORIGINAL}">뉴스1</a>'
    pages[ORIGINAL] = "<p>엑시나는 1500억 원 투자를 추진 중이다.</p>"     # 원본 내용이 다르다
    assert web.resolve_secondary([{"url": BLOG, "title": "블로그", "content": CLAIM}]) == []


def test_portal_kept_when_original_not_found_and_normal_results_untouched(pages):
    plain = {"url": ORIGINAL, "title": "기사", "content": CLAIM}
    portal = {"url": PORTAL, "title": "기사 - 뉴스1", "content": CLAIM}
    assert web.resolve_secondary([plain, portal]) == [plain, portal]


def test_sns_posts_are_secondary(pages):
    post = "https://www.instagram.com/p/DdDUVfWnxBl"
    assert web.is_secondary(post) and web.is_secondary("https://www.linkedin.com/posts/abc")
    assert not web.is_secondary("https://dx.com/a")                 # 'x.com'으로 끝나는 다른 도메인은 아니다
    pages[post] = f"<p>{CLAIM}</p>"                                  # 원본 링크가 없는 게시물은 뺀다
    assert web.resolve_secondary([{"url": post, "title": "[공시분석]", "content": CLAIM}]) == []

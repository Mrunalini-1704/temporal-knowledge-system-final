"""
Page parsing: clean markdown, classify pages by authority, extract rate tables and text chunks.

Everything here is pure standard library and deterministic.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

# --------------------------------------------------------------------------------------
# Text normalisation
# --------------------------------------------------------------------------------------

MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"])}
MONTH_ABBR = {k[:3]: v for k, v in MONTHS.items()}

_IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)\s]*)(?:\s+\"[^\"]*\")?\)")
_WS_RE = re.compile(r"[ \t   ]+")


def normalize_text(md: str) -> str:
    """Fix mojibake/dashes/escaped newlines that the crawler left in the markdown."""
    s = md.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\r\n", "\n")
    s = re.sub(r"(?<=\d)�(?=\d)", "-", s)       # "2023�26" -> "2023-26"
    s = re.sub(r"(?<=\w)�(?=s\b)", "'", s)       # "Domino�s" -> "Domino's"
    s = s.replace("�", " ")
    for a, b in (("–", "-"), ("—", "-"), ("‑", "-"), ("’", "'"), ("‘", "'"),
                 ("“", '"'), ("”", '"'), (" ", " "), (" ", " "), ("​", "")):
        s = s.replace(a, b)
    s = s.replace("\\*", "*").replace("\\#", "#").replace("\\$", "$").replace("\\_", "_")
    return s


def strip_inline(line: str, keep_urls: bool = False) -> str:
    """Drop images, turn links into their anchor text (optionally "text (url)")."""
    line = _IMG_RE.sub(" ", line)

    def _link(m):
        text, url = m.group(1).strip(), m.group(2).strip()
        if keep_urls and url.startswith("http") and len(url) < 90 and "?" not in url and "#" not in url:
            if not text or text.lower() in url.lower():
                return url
            return f"{text} ({url})"
        return text
    line = _LINK_RE.sub(_link, line)
    line = line.replace("**", "").replace("__", "")
    return _WS_RE.sub(" ", line).strip()


# --------------------------------------------------------------------------------------
# Page classification
# --------------------------------------------------------------------------------------

LANDING_PATHS = ("/switch-and-earn", "/earn-more", "/mortgage-refi", "/homebuying-guidance", "/checking",
                 "/offer-expired", "/money-makeover", "/not-for-sale", "/thankuse", "/kccc",
                 "/21-day-sprint-to-savings-challenge", "/sprint-to-savings-challenge", "/covid-19")


def url_path(url: str) -> str:
    p = urlparse(url)
    return (p.path.rstrip("/") or "/").lower()


def canonical_url(url: str) -> str:
    p = urlparse(url)
    path = p.path.rstrip("/") or ""
    return f"https://{p.netloc.lower()}{path}"


def page_kind(url: str) -> str:
    path = url_path(url)
    if path in ("/rates", "/business-rates"):
        return "rates"
    if path.startswith("/hubfs") or re.search(r"\.(pdf|xlsx?|docx?)$", path):
        return "file"
    if path.startswith("/calculators") or path == "/business-calculators":
        return "calculator"
    if path.startswith("/newsroom/tag") or path.startswith("/newsroom/author") or path in ("/newsroom", "/resources",
                                                                                            "/resources/dupaco-news"):
        return "listing"
    if path.startswith("/newsroom"):
        return "news"
    if path.startswith("/business/resources") or path.startswith("/resources/"):
        return "bizres" if path.startswith("/business/resources") else "listing"
    if path.startswith("/learn") or path.startswith("/dupacodaily") or re.match(r"^/20\d\d/", path):
        return "blog"
    if path.endswith("-lp") or path in LANDING_PATHS:
        return "landing"
    return "product"


# How much we trust each kind of page for "what is true now".
AUTHORITY = {"rates": 1.0, "product": 0.85, "landing": 0.45, "news": 0.55, "blog": 0.4, "bizres": 0.35,
             "calculator": 0.25, "file": 0.3, "listing": 0.1}


def skip_url(url: str) -> bool:
    """Crawler junk: tracking-modal duplicates and query-string variants."""
    q = urlparse(url).query
    return "campaignId=" in q or "pageUrl=" in q or len(url) > 300


# --------------------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------------------

_LONG_DATE_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\.?\s+(\d{1,2}),\s*(20\d\d)\b")
_ABBR_DATE_RE = re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+(\d{1,2}),\s*(20\d\d)\b")
_NUM_DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(20\d\d|\d\d)\b")


def parse_long_date(s: str) -> Optional[str]:
    m = _LONG_DATE_RE.search(s)
    if m:
        return f"{int(m.group(3)):04d}-{MONTHS[m.group(1).lower()]:02d}-{int(m.group(2)):02d}"
    m = _ABBR_DATE_RE.search(s)
    if m:
        return f"{int(m.group(3)):04d}-{MONTH_ABBR[m.group(1).lower()[:3]]:02d}-{int(m.group(2)):02d}"
    return None


def parse_num_date(s: str) -> Optional[str]:
    m = _NUM_DATE_RE.search(s)
    if not m:
        return None
    y = int(m.group(3))
    y = y + 2000 if y < 100 else y
    return f"{y:04d}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"


def article_date(lines: list[str], kind: str) -> Optional[str]:
    """Publication date of a news/blog article: first long-form date near the top."""
    if kind not in ("news", "blog", "bizres"):
        return None
    for ln in lines[:25]:
        if len(ln) < 60:
            d = parse_long_date(ln)
            if d:
                return d
    return None


# --------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------

_FOOTNOTE_RE = re.compile(r"(%|\bAPR|\bAPY|\))\s?[1-9¹²³⁴]\b")
_PCT_RE = re.compile(r"\d+(?:\.\d+)?\s*%")
VALUE_HEADER_RE = re.compile(r"rate|apr|apy|yield|points|payment|min\.?|minimum|fee|price|cost|amount financed", re.I)


def clean_cell(c: str) -> str:
    c = strip_inline(c)
    c = c.replace("\\", "").replace("*", "").strip()
    c = _FOOTNOTE_RE.sub(r"\1", c)
    c = re.sub(r"(\d)\s*%\s*([1-9])$", r"\1%", c)      # "20.5% 1" -> "20.5%"
    c = re.sub(r"(?<=[A-Za-z])[1-9]$", "", c) if re.search(r"APR[1-9]$|APY[1-9]$", c) else c
    return _WS_RE.sub(" ", c).strip()


@dataclass
class Row:
    cells: list[str]
    labels: list[str]                     # descriptive cells (product, tier, term)
    values: list[tuple[str, str]]         # (header, value) for numeric/rate cells
    label_text: str = ""


@dataclass
class Table:
    url: str
    kind: str
    title: str                            # line right above the table
    section: str                          # nearest section heading above it
    header: list[str]
    rows: list[Row]
    effective: Optional[str] = None       # "Rates are effective as of ..." (ISO)
    effective_raw: Optional[str] = None
    index: int = 0                        # order on the page
    notes: str = ""                       # footnote text right after the table
    detail_links: list[str] = field(default_factory=list)   # "See details" product pages

    @property
    def doc(self) -> str:
        return " ".join([self.section, self.title, " ".join(self.header)])


_SECTION_RE = re.compile(r"^(#{1,6}\s+.+|-\s*\[[^\]]+\]\([^)]*\)\s*$|- [A-Z][^.]{2,60}$)")


def _is_title_candidate(ln: str) -> bool:
    if re.fullmatch(r"(\s*\[[^\]]*\]\([^)]*\))+\s*", ln):     # a row of buttons/links
        return False
    s = strip_inline(ln)
    if not s or len(s) > 80 or s.startswith("|"):
        return False
    if s.startswith(("_", "Rates are", "Rates effective", "*")) or s.endswith((".", ":")) and len(s) > 40:
        return False
    if re.match(r"^\d+[A-Z]|^[\d.,$%\s-]+$", s):      # footnote ("1Checking Rates: ...") or a bare number
        return False
    return True


def _is_value_cell(header: str, cell: str) -> bool:
    if not cell or cell in ("—", "-", "N/A", "None"):
        return bool(VALUE_HEADER_RE.search(header)) and not re.search(r"tier|balance|term|type|program", header, re.I)
    if re.search(r"tier|balance|term|type|program|certificate$|loan amount|product|card", header, re.I) \
            and not re.search(r"rate|apr|apy|yield", header, re.I):
        return False
    if VALUE_HEADER_RE.search(header) and re.search(r"\d", cell):
        return True
    return bool(re.fullmatch(r"[\d.,$%\s\-APRY]+", cell)) and "%" in cell


def parse_tables(url: str, kind: str, lines: list[str]) -> list[Table]:
    tables: list[Table] = []
    i, n = 0, len(lines)
    section = ""
    while i < n:
        ln = lines[i]
        st = ln.strip()
        if _SECTION_RE.match(st) and not st.startswith("|"):
            section = strip_inline(st.lstrip("#- ").strip())
        if st.startswith("|") and i + 1 < n and re.match(r"^\|\s*:?-{3}", lines[i + 1].strip()):
            header = [clean_cell(c) for c in st.strip("|").split("|")]
            # title: closest short line above the table
            title, k = "", i - 1
            above = []
            while k >= 0 and len(above) < 4 and i - k < 30:
                s = lines[k].strip()
                if s and not s.startswith("|"):
                    above.append(s)
                k -= 1
            for s in above:
                if _is_title_candidate(s):
                    title = strip_inline(s.lstrip("#- "))
                    break
            j = i + 2
            rows: list[Row] = []
            while j < n and lines[j].strip().startswith("|"):
                cells = [clean_cell(c) for c in lines[j].strip().strip("|").split("|")]
                if any(cells):
                    labels, values = [], []
                    for ci, c in enumerate(cells):
                        h = header[ci] if ci < len(header) else ""
                        if _is_value_cell(h, c):
                            # bare numbers in a rate column ("7.29") are percentages
                            if re.fullmatch(r"\d{1,2}\.\d{1,3}", c) and re.search(r"rate|apr|apy|yield", h, re.I):
                                c = c + "%"
                            values.append((h, c))
                        elif c:
                            labels.append(c)
                    rows.append(Row(cells=cells, labels=labels, values=values, label_text=" ".join(labels)))
                j += 1
            eff_raw, eff, notes, detail_links = None, None, [], []
            for k in range(j, min(n, j + 40)):
                s = lines[k].strip()
                if s.startswith("|") or len(notes) >= 6:
                    break
                if not s:
                    continue
                for m in _LINK_RE.finditer(s):
                    u = m.group(2)
                    if u.startswith("http") and "dupaco.com/" in u and "join." not in u and "shine." not in u \
                            and "homeloans." not in u \
                            and "#" not in u.split("dupaco.com", 1)[1][:2]:
                        detail_links.append(canonical_url(u))
                notes.append(strip_inline(s))
                if eff is None and re.search(r"effective", s, re.I):
                    d = parse_num_date(s) or parse_long_date(s)
                    if d:
                        eff, eff_raw = d, re.search(r"\d{1,2}/\d{1,2}/\d{2,4}|[A-Z][a-z]+\.? \d{1,2}, 20\d\d", s).group(0)
            if rows:
                tables.append(Table(url=url, kind=kind, title=title, section=section, header=header, rows=rows,
                                    effective=eff, effective_raw=eff_raw, index=len(tables),
                                    notes=" ".join(notes)[:1500], detail_links=detail_links[:3]))
            i = j
            continue
        i += 1
    return tables


# --------------------------------------------------------------------------------------
# Chunks for text retrieval
# --------------------------------------------------------------------------------------

STOP_SECTION_RE = re.compile(r"^#{1,6}\s*(Handpicked for You|Related (Articles|Posts|Resources)|You might also like|"
                             r"More from|Recent Posts|Keep Reading)\b", re.I)
JUNK_LINE_RE = re.compile(r"^(Third Party Plugins|Messenger Launcher|Messenger|✕|Continue Cancel|Cancel|Loading.*|"
                          r"\d+ min read|Read More|Keep reading|Share|Go|Previous|Next|Skip to .*)$", re.I)


@dataclass
class Chunk:
    url: str
    kind: str
    title: str
    heading: str
    text: str            # display text (links rendered as "text (url)")
    pos: int             # position on page (0 = top)
    date: Optional[str] = None


def make_chunks(url: str, kind: str, title: str, lines: list[str], date: Optional[str]) -> list[Chunk]:
    chunks: list[Chunk] = []
    heading = ""
    buf: list[str] = []

    def flush():
        if buf:
            txt = "\n".join(buf).strip()
            if len(re.sub(r"\W", "", txt)) >= 12:
                chunks.append(Chunk(url, kind, title, heading, txt, len(chunks), date))
            buf.clear()

    for raw in lines:
        st = raw.strip()
        if STOP_SECTION_RE.match(st):
            break
        if st.startswith("|"):
            continue  # tables are handled separately (and as their own chunks below)
        if st.startswith("#"):
            flush()
            heading = strip_inline(st.lstrip("#").strip())
            continue
        s = strip_inline(st, keep_urls=True)
        plain = strip_inline(st)
        if plain.endswith("...") and len(plain) < 300:      # teaser of another article
            continue
        if len(_LINK_RE.findall(st)) >= 2 and not _LINK_RE.sub("", _IMG_RE.sub("", st)).strip(" |-*·,"):
            continue                                        # a row of tag/nav links
        if re.fullmatch(r"[\w ,/'()-]{1,45}\*", plain):
            continue                                        # a form field label ("Phone number*")
        if not s or JUNK_LINE_RE.match(plain) or len(plain) < 3:
            if not s and sum(len(b) for b in buf) > 450:
                flush()
            continue
        buf.append(s)
        if sum(len(b) for b in buf) > 700:
            flush()
    flush()
    return chunks


def table_chunks(tables: list[Table], title: str, date: Optional[str]) -> list[Chunk]:
    out = []
    for t in tables:
        rows = []
        for r in t.rows[:25]:
            rows.append(" | ".join(c for c in r.cells if c))
        head = " / ".join(x for x in (t.section, t.title) if x)
        txt = (head + ":\n" if head else "") + "\n".join(rows)
        out.append(Chunk(t.url, t.kind, title, head, txt, 50 + t.index, date))
    return out


# --------------------------------------------------------------------------------------
# Whole page
# --------------------------------------------------------------------------------------

@dataclass
class PageRec:
    url: str
    title: str
    kind: str
    authority: float
    date: Optional[str]
    lines: list[str]
    tables: list[Table] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    text: str = ""
    links: list[tuple[str, str]] = field(default_factory=list)   # (anchor text, url)
    carried_from: Optional[str] = None    # snapshot id, when the page was missing from a later crawl


def build_page(url: str, title: str, markdown: str, boiler: set[str]) -> PageRec:
    kind = page_kind(url)
    md = normalize_text(markdown)
    all_lines = md.split("\n")
    # Nav/footer lines repeated across the site are dropped for text, but tables are parsed from the
    # full page because product names in the nav menu double as table titles.
    lines = [ln for ln in all_lines if ln.strip() not in boiler or ln.strip().startswith("|")]
    date = article_date(lines, kind)
    tables = parse_tables(url, kind, all_lines)
    t = re.sub(r"\s*\|\s*Dupaco.*$", "", normalize_text(title or "")).strip()
    chunks = make_chunks(url, kind, t, lines, date) + table_chunks(tables, t, date)
    links = [(strip_inline(m.group(1)), m.group(2)) for m in _LINK_RE.finditer(md) if m.group(2).startswith("http")]
    text = "\n".join(strip_inline(ln) for ln in lines)
    return PageRec(url=url, title=t, kind=kind, authority=AUTHORITY[kind], date=date, lines=lines, tables=tables,
                   chunks=chunks, text=text, links=links)

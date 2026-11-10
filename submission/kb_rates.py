"""
Rate lookups over the parsed rate tables of one snapshot.

The question is turned into qualifiers (term, model year, balance/loan amount, jumbo, Roth/Traditional,
purchase/refinance, business, intro...). Tables are ranked by how well they name the product, weighted by
page authority (official rate sheet > product page > landing page > everything else). Rows are then
filtered by the qualifiers so we quote the right tier and nothing else.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

from kb_parse import Table, canonical_url
from kb_text import GENERIC, NUM_WORDS, STOP, content_tokens, expand, stem, tokens

RATE_WORDS_RE = re.compile(r"\b(rates?|aprs?|apys?|yields?|dividends?|interest|percent|earns?|earning|pays?|paying)\b|%",
                           re.I)

# Products a member may ask about that the site never prices: if the question names one of these and
# the best table doesn't, we must not answer with a neighbouring product's rate.
UNLISTED = {"construction", "adjustable", "arm", "land", "lot", "fha", "va", "usda", "sba", "reverse", "bridge",
            "balloon", "interest-only", "student", "commercial", "jumbo", "agricultural", "farm", "solar",
            "condo", "investment", "rental", "signature", "secured", "share-secured", "pledge", "payday",
            "overdraft", "line", "business", "christmas", "youth", "kid", "teen", "trust", "estate", "used",
            "new", "lease", "leasing", "equipment", "tractor", "aircraft", "plane", "ira", "roth", "traditional",
            "money", "market", "health", "hsa", "launchpad", "bright", "holiday", "high", "achieve",
            "variable", "relief", "refinance", "purchase", "intro", "introductory", "transfer", "cash", "advance",
            "platinum", "reward", "heloc", "equity", "improvement", "boat", "rv", "atv", "snowmobile", "jet",
            "motorcycle", "personal", "auto", "home", "certificate", "checking", "saving", "share", "card", "visa",
            "mortgage", "special", "regular", "you", "name"}

PRODUCT_NOUNS = {"certificate", "saving", "checking", "money", "market", "ira", "roth", "traditional", "auto",
                 "motorcycle", "home", "mortgage", "heloc", "equity", "improvement", "personal", "boat", "rv",
                 "atv", "snowmobile", "jet", "card", "visa", "platinum", "reward", "launchpad", "bright",
                 "holiday", "health", "high", "construction", "student", "land", "business", "share"}

UNLISTED = {stem(w) for w in UNLISTED}
PRODUCT_NOUNS = {stem(w) for w in PRODUCT_NOUNS}
# qualifiers that name a product the rate sheet never prices
HARD_UNLISTED = {stem(w) for w in ("construction", "adjustable", "arm", "land", "lot", "fha", "va", "usda", "sba",
                                   "reverse", "bridge", "balloon", "student", "agricultural", "farm", "solar",
                                   "jumbo", "lease", "leasing", "equipment", "aircraft", "signature", "secured",
                                   "overdraft", "christmas", "youth")}
_INVESTOR, _MONEY, _MARKET, _CARD, _VISA = stem("investor"), stem("money"), stem("market"), stem("card"), "visa"
_PLAT, _REWARD = stem("platinum"), stem("rewards")

SHORT_COL = [(re.compile(r"dividend", re.I), "dividend rate"), (re.compile(r"apy|yield", re.I), "APY"),
             (re.compile(r"\bapr\b|annual percentage rate", re.I), "APR"), (re.compile(r"^rate$", re.I), "rate"),
             (re.compile(r"min\.? to open", re.I), "min. to open"), (re.compile(r"min\.? to earn", re.I),
                                                                      "min. to earn"),
             (re.compile(r"points", re.I), "points"), (re.compile(r"payment", re.I), "payment example")]


def short_col(h: str) -> str:
    for rx, name in SHORT_COL:
        if rx.search(h):
            return name
    return h.lower()


def _money(s: str) -> Optional[float]:
    m = re.search(r"\$?\s*([\d,]+(?:\.\d+)?)\s*(k|m)?\b", s, re.I)
    if not m:
        return None
    try:
        v = float(m.group(1).replace(",", ""))
    except ValueError:
        return None
    if m.group(2):
        v *= 1000 if m.group(2).lower() == "k" else 1_000_000
    return v


def _pct(s: str) -> Optional[float]:
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", s)
    return float(m.group(1)) if m else None


# --------------------------------------------------------------------------------------
# Row / table features
# --------------------------------------------------------------------------------------

@dataclass
class RowFeat:
    table: "TableFeat"
    idx: int
    labels: list[str]
    values: list[tuple[str, str]]
    term: Optional[int] = None
    years: Optional[tuple[int, int]] = None
    amount: Optional[tuple[float, float]] = None
    flags: set = field(default_factory=set)
    primary: Optional[float] = None
    text: str = ""

    @property
    def signature(self) -> tuple:
        return tuple(v for _, v in self.values if "%" in v)


@dataclass
class TableFeat:
    table: Table
    page_title: str
    authority: float
    toks: set
    flags: set
    rows: list[RowFeat]
    name: str


def _row_features(tf: TableFeat, i: int, labels: list[str], values: list[tuple[str, str]]) -> RowFeat:
    lt = " ".join(labels)
    low = lt.lower()
    r = RowFeat(tf, i, labels, values, text=lt)
    m = re.search(r"(\d{1,3})\s*-?\s*months?\b|\((\d{1,3})\s*mo\)", low)
    if m:
        r.term = int(m.group(1) or m.group(2))
    else:
        m = re.search(r"(\d{1,2})\s*-?\s*years?\b", low)
        if m:
            r.term = int(m.group(1)) * 12
    m = re.search(r"(?:model year\s*)?((?:19|20)\d\d)\s*-\s*(\d{2,4})(?:\s*model year)?", low)
    if m and "model" in low:
        a = int(m.group(1))
        b = int(m.group(2))
        if b < 100:
            b += (a // 100) * 100
        r.years = (min(a, b), max(a, b))
    else:
        m = re.search(r"((?:19|20)\d\d)\s*model year\s*\(and older\)", low)
        if m:
            r.years = (1900, int(m.group(1)))
    # balance / loan-amount tier
    for lab in labels:
        ll = lab.lower()
        if "$" not in lab and not re.search(r"\d{1,3},\d{3}", lab):
            continue
        m = re.search(r"\$?\s*([\d,]+(?:\.\d+)?)\s*-\s*\$?\s*([\d,]+(?:\.\d+)?)", lab)
        if m:
            r.amount = (float(m.group(1).replace(",", "")), float(m.group(2).replace(",", "")))
            break
        m = re.search(r"\$\s*([\d,]+(?:\.\d+)?)\s*\+", lab)
        if m:
            r.amount = (float(m.group(1).replace(",", "")), math.inf)
            break
        m = re.search(r"\$\s*([\d,]+(?:\.\d+)?)\s*or less", ll)
        if m:
            r.amount = (0.0, float(m.group(1).replace(",", "")))
            break
    for f, rx in (("jumbo", r"\bjumbo\b"), ("special", r"\bspecial\b"), ("intro", r"\bintro"),
                  ("relief", r"\brelief\b"), ("roth", r"\broth\b"), ("traditional", r"\btraditional\b"),
                  ("purchase", r"\bpurchase\b"), ("refinance", r"\brefinance\b"), ("business", r"\bbusiness\b")):
        if re.search(rx, low):
            r.flags.add(f)
    for pref in ("APY", "APR", "rate", "dividend rate"):
        for h, v in values:
            if short_col(h) == pref and _pct(v) is not None:
                r.primary = _pct(v)
                break
        if r.primary is not None:
            break
    return r


def _norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", re.sub(r"\b(rates?|accounts?)\b", "", name.lower()))


def _clean_title(t: str) -> str:
    t = re.sub(r"\b(Open my .*|See details|Learn more|Explore)\b.*$", "", t).strip()
    return t


def build_table_feats(pages: dict) -> list[TableFeat]:
    feats: list[TableFeat] = []
    for url, p in pages.items():
        if p.kind in ("listing", "news", "blog", "bizres"):
            continue
        for t in p.tables:
            if not any("%" in v for r in t.rows for _, v in r.values):
                continue
            title = _clean_title(t.title)
            if not title and t.rows and t.rows[0].labels:
                title = t.rows[0].labels[0]
            doc = " ".join([t.section, title, " ".join(t.header)] + [r.label_text for r in t.rows])
            toks = set(expand(tokens(doc)))
            if stem("conventional") in toks or {stem("home"), "loan"} <= toks:
                toks.add(stem("mortgage"))
            if "visa" in toks:
                toks.update({stem("credit"), stem("card")})
            if stem("certificate") in toks:
                toks.update({"cd", "cds", stem("term")})
            flags = set()
            hdr_title = (t.section + " " + title + " " + " ".join(t.header)).lower()
            if "business" in hdr_title or url.endswith("/business-rates") or "/business/" in url:
                flags.add("business")
            if re.search(r"\bira", hdr_title) or all(re.search(r"roth|traditional", r.label_text, re.I) for r in t.rows):
                flags.add("ira")
            if "refinance" in hdr_title and "equity" not in hdr_title:
                flags.add("refinance")
            if "equity" in hdr_title and "refinance" in hdr_title or "cash out" in hdr_title:
                flags.add("equity_refi")
            if p.kind == "rates":
                auth = 1.0
            elif p.kind == "product":
                auth = 0.85
            elif p.kind == "landing":
                auth = 0.45
            else:
                auth = 0.2
            if p.carried_from:
                auth *= 0.7         # missing from the latest crawl: never quoted, used for conflicts only
            tf = TableFeat(t, p.title, auth, toks, flags, [], title or p.title)
            tf.rows = [_row_features(tf, i, r.labels, r.values) for i, r in enumerate(t.rows)]
            feats.append(tf)
    return feats


# --------------------------------------------------------------------------------------
# Question features
# --------------------------------------------------------------------------------------

@dataclass
class QFeat:
    text: str
    toks: list[str]
    prod: list[str]
    terms: set
    model_year: Optional[int]
    amounts: list[float]
    flags: set
    superlative: Optional[str]
    want_cols: set
    raw_prod: list = field(default_factory=list)   # product words as typed (before synonym expansion)


# words that describe the asker or the situation rather than the product
_FILLER = {stem(w) for w in """
balance between borrow borrowing deposit deposits put want looking need get buy buying new used model loan
vehicle car truck suv automobile highest lowest top best minimum maximum on with over under year month day
dollar dollars amount term terms qualified qualify typical people member members someone
quote quoted rn tier tiers level levels fixed variable option options product products type types kind kinds
open opening start starting financing finance finances say tell send email cost costs give number
list show all every their them its your our today going please various different available currently
change changed changes increase increased decrease decreased gone went up down since raised lowered cut
previous previously before old used differ different still
""".split()}


_MONTH_NAMES = r"(january|february|march|april|may|june|july|august|september|october|november|december)"


def parse_question(q: str) -> QFeat:
    ql = q.lower().replace("–", "-")
    ql = re.sub(r"\bas of\b.*?\b20\d\d\b", " ", ql)
    ql = re.sub(_MONTH_NAMES + r"\s+(\d{1,2},\s*)?20\d\d", " ", ql)
    for w, n in sorted(NUM_WORDS.items(), key=lambda x: -len(x[0])):
        ql = re.sub(rf"\b{w}(?=[\s-]*(month|year|yr|mo)\b)", str(n), ql)
    terms = set()
    for m in re.finditer(r"\b(\d{1,3})\s*-?\s*(months?|mos?|mo\.)\b", ql):
        terms.add(int(m.group(1)))
    for m in re.finditer(r"\b(\d{1,2})\s*-?\s*(years?|yrs?)\b(?!\s*old)", ql):
        if not re.search(r"(19|20)\d\d\s*-?\s*$", ql[:m.start()]):
            terms.add(int(m.group(1)) * 12)
    if re.search(r"\b(a|one)\s+year\b", ql) and not terms:
        terms.add(12)
    model_year = None
    m = re.search(r"\b((?:19|20)\d\d)\b", ql)
    if m and re.search(r"model|vehicle|car|auto|truck|suv|motorcycle|atv|snowmobile|boat|jet|year", ql):
        model_year = int(m.group(1))
    amounts = []
    for m in re.finditer(r"\$\s*([\d,]+(?:\.\d+)?)\s*(k|thousand|million|m)?\b|\b([\d,]+(?:\.\d+)?)\s*(k|thousand)\b"
                         r"|\b(\d{1,3}(?:,\d{3})+)(?!\s*(%|percent))", ql):
        num = m.group(1) or m.group(3) or m.group(5)
        unit = (m.group(2) or m.group(4) or "").lower()
        try:
            v = float(num.replace(",", ""))
        except ValueError:
            continue
        if unit in ("k", "thousand"):
            v *= 1000
        elif unit in ("m", "million"):
            v *= 1_000_000
        amounts.append(v)
    toks = expand(content_tokens(ql))
    flags = set()
    for f, rx in (("jumbo", r"\bjumbo\b"), ("special", r"\bspecial|promo"), ("intro", r"\bintro|balance transfer"),
                  ("relief", r"\brelief\b"), ("roth", r"\broth\b"), ("traditional", r"\btraditional\b"),
                  ("purchase", r"\bpurchas|\bbuy"), ("refinance", r"\brefi"), ("business", r"\bbusiness|commercial"),
                  ("ira", r"\bira\b|\biras\b|roth|traditional|retirement"), ("personal", r"\bpersonal\b"),
                  ("equity", r"\bequity\b|heloc"), ("list", r"\b(all|every|list|options|types|kinds)\b|what .* offer")):
        if re.search(rx, ql):
            flags.add(f)
    sup = None
    if re.search(r"\b(highest|best|top|max(imum)?|most)\b", ql) and not re.search(r"best (way|time)", ql):
        sup = "max"
    elif re.search(r"\b(lowest|cheapest|min(imum)? rate|least)\b", ql):
        sup = "min"
    want = set()
    if re.search(r"\bpayment", ql):
        want.add("payment example")
    if re.search(r"\bpoints?\b", ql):
        want.add("points")
    if re.search(r"\bminimum|\bmin\b|to open|to earn|deposit", ql):
        want.update({"min. to open", "min. to earn"})
    prod = [w for w in toks if w not in GENERIC and not w.isdigit() and w not in STOP]
    raw_prod = [w for w in content_tokens(ql) if w not in GENERIC and not w.isdigit() and w not in STOP
                and w not in _FILLER]
    return QFeat(q, toks, prod, terms, model_year, amounts, flags, sup, want, raw_prod)


def is_rate_question(q: str) -> bool:
    return bool(RATE_WORDS_RE.search(q)) or bool(
        re.search(r"\bminimum\b.{0,40}\b(open|deposit|balance|earn)\b|\bmin(imum)? to (open|earn)\b", q, re.I))


# --------------------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------------------

@dataclass
class RateMatch:
    tables: list[tuple[TableFeat, list[RowFeat]]]
    score: float
    refused_reason: Optional[str] = None


class RateIndex:
    def __init__(self, pages: dict):
        self.tables = build_table_feats(pages)
        n = len(self.tables) or 1
        df: dict[str, int] = {}
        for t in self.tables:
            for w in t.toks:
                df[w] = df.get(w, 0) + 1
        self.idf = {w: math.log(1 + n / c) for w, c in df.items()}
        self.vocab = set(df)
        # words that name products somewhere on the site (product page titles and URLs, table text)
        self.product_vocab = set(self.vocab) | UNLISTED | PRODUCT_NOUNS
        for url, p in pages.items():
            if p.kind in ("product", "rates"):
                self.product_vocab.update(tokens(p.title + " " + url.split("dupaco.com", 1)[-1].replace("-", " ")))
        self.product_vocab -= _FILLER | GENERIC | STOP

    # -- table score ------------------------------------------------------------------
    def _table_score(self, q: QFeat, tf: TableFeat) -> float:
        prod = set(q.prod)
        s = sum(self.idf.get(w, 0.0) for w in prod if w in tf.toks)
        # unmatched product nouns from the question count against the table
        s -= 0.5 * sum(self.idf.get(w, 1.0) for w in prod if w in PRODUCT_NOUNS and w not in tf.toks)
        f = q.flags
        if "ira" in tf.flags and "ira" not in f:
            s -= 3.0
        if "ira" in f and "ira" not in tf.flags:
            s -= 2.0
        if "business" in tf.flags and "business" not in f:
            s -= 2.5
        if "business" in f and "business" not in tf.flags:
            s -= 2.0
        if "refinance" in tf.flags and "refinance" not in f:
            s -= 1.5
        if "equity_refi" in tf.flags and not ("refinance" in f and "equity" in f):
            s -= 2.5
        rows = tf.rows
        if q.terms:
            if not any(r.term for r in rows):
                s -= 2.5          # the member named a term; this product has none
            elif not any(r.term in q.terms for r in rows):
                s -= 4.0
            else:
                s += 1.0
        if "jumbo" in f and not any("jumbo" in r.flags for r in rows):
            s -= 1.5
        if q.model_year and any(r.years for r in rows):
            s += 0.5
        return s * (0.6 + 0.4 * tf.authority)

    # -- row filtering ------------------------------------------------------------------
    @staticmethod
    def _filter_rows(q: QFeat, tf: TableFeat) -> tuple[list[RowFeat], Optional[str]]:
        rows = list(tf.rows)
        f = q.flags

        def narrow(pred, required: bool, why: str):
            nonlocal rows
            keep = [r for r in rows if pred(r)]
            if keep:
                rows = keep
                return None
            return why if required else None

        if q.terms and any(r.term for r in rows):
            why = narrow(lambda r: r.term in q.terms, True, "term not offered")
            if why:
                return [], why
        if q.model_year and any(r.years for r in rows):
            why = narrow(lambda r: r.years and r.years[0] <= q.model_year <= r.years[1], True,
                         "model year not listed")
            if why:
                return [], why
        if q.amounts and any(r.amount for r in rows):
            amts = [a for a in q.amounts if a >= 1]
            if amts:
                keep = [r for r in rows if r.amount and all(r.amount[0] <= a <= r.amount[1] + 0.999 for a in amts)]
                if keep:
                    # "$X or less" tiers nest: pick the tightest one
                    tight = min(r.amount[1] - r.amount[0] for r in keep)
                    rows = [r for r in keep if r.amount[1] - r.amount[0] == tight]
        for flag in ("jumbo", "intro", "relief", "roth", "traditional"):
            has = [r for r in rows if flag in r.flags]
            if flag in f:
                if has:
                    rows = has
                elif flag in ("jumbo", "relief", "intro"):
                    return [], f"no {flag} version listed"
            elif has and len(has) < len(rows) and flag not in ("roth", "traditional"):
                rows = [r for r in rows if flag not in r.flags]   # variants only when asked for
        if any("refinance" in r.flags for r in rows) and any("purchase" in r.flags for r in rows):
            want = "refinance" if "refinance" in f else "purchase"
            rows = [r for r in rows if want in r.flags]
        if "special" in f and any("special" in r.flags for r in rows):
            rows = [r for r in rows if "special" in r.flags]
        if q.superlative and rows and any(r.primary is not None for r in rows):
            vals = [r.primary for r in rows if r.primary is not None]
            best = max(vals) if q.superlative == "max" else min(vals)
            rows = [r for r in rows if r.primary == best][:1]
        return rows, None

    def match(self, q: QFeat) -> RateMatch:
        if not self.tables:
            return RateMatch([], 0.0, "no rate tables")
        # Only the official rate sheets and product pages are quoted. Landing pages, calculators and PDFs
        # are where stale and expired-promotion rates live; they are used for conflict detection only.
        raw = [(self._table_score(q, tf), i, tf) for i, tf in enumerate(self.tables) if tf.authority >= 0.85]
        if not raw:
            return RateMatch([], 0.0, "no rate tables")
        best_s = max(x[0] for x in raw)
        if best_s <= 0.5:
            return RateMatch([], best_s, "no matching product")
        scored = sorted(((sc, -tf.authority, i, tf) for sc, i, tf in raw), key=lambda x: (-round(x[0], 1), x[1], x[2]))
        # The same product shown on several pages: the most authoritative copy wins, even if a less
        # authoritative page happens to match one more word of the question.
        top_name = _norm_name(scored[0][3].name)
        same = [x for x in scored if _norm_name(x[3].name) == top_name and x[0] >= 0.75 * best_s]
        if same:
            lead = min(same, key=lambda x: (x[1], -x[0], x[2]))
            scored.remove(lead)
            scored.insert(0, lead)
        best = scored[0][3]
        # most of what the member named must be on the table, or we are answering a different product
        named = [w for w in q.raw_prod if not w.isdigit() and w in self.product_vocab]
        if named:
            covered = sum(1 for w in named if w in best.toks or any(s in best.toks for s in expand([w])[1:]))
            if covered / len(named) < 0.5:
                return RateMatch([], best_s, "product not on the rate sheet")
        missing = [w for w in set(q.prod) if w in UNLISTED and w not in best.toks
                   and not (w in (_MONEY, _MARKET) and _INVESTOR in best.toks)]
        if any(w in HARD_UNLISTED for w in missing):
            return RateMatch([], best_s, "product not on the rate sheet: " + ", ".join(sorted(missing)))
        chosen: list[tuple[TableFeat, list[RowFeat]]] = []
        seen_names = set()
        listing = not q.terms and not q.amounts and not q.model_year and not q.superlative
        card_list = (_CARD in q.prod or _VISA in q.prod) and not ({_PLAT, _REWARD} & set(q.prod)) \
            and "business" not in q.flags and "personal" not in q.flags
        for s, _, _, tf in scored:
            if s < best_s * 0.8 and not (card_list and _VISA in tf.toks and s > 0):
                continue
            if chosen and not listing and not card_list:
                break
            key = re.sub(r"[^a-z0-9]", "", tf.name.lower())
            if key in seen_names:
                continue
            # never mix an official table with a lower-authority copy of the same product
            if chosen and tf.authority < chosen[0][0].authority and not card_list:
                continue
            if card_list and _VISA not in tf.toks:
                continue
            rows, why = self._filter_rows(q, tf)
            if not rows:
                if not chosen and why:
                    return RateMatch([], s, why)
                continue
            seen_names.add(key)
            chosen.append((tf, rows))
            if len(chosen) >= (4 if card_list else 3):
                break
        if not chosen:
            return RateMatch([], best_s, "no matching rows")
        return RateMatch(chosen, best_s)


# --------------------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------------------

def row_label(tf: TableFeat, r: RowFeat) -> str:
    labels = [l for l in r.labels if l not in ("-", "—")]
    name = tf.name
    if labels:
        m = re.match(r"(purchase|refinance):\s*", labels[0], re.I)
        if m and m.group(1).lower() in name.lower():
            labels[0] = labels[0][m.end():]
    base = labels[0] if labels else name
    rest = labels[1:]
    if name and not set(tokens(name)) - {"rate", "account", "loan"} <= set(tokens(" ".join(labels))):
        if labels and not re.search(r"[a-z]{3}", labels[0].lower().replace("month", "").replace("year", "")):
            base, rest = name, labels
        elif labels:
            base, rest = f"{name} - {labels[0]}", labels[1:]
    return base + (f" ({', '.join(rest)})" if rest else "")


def row_values(r: RowFeat, want: set) -> str:
    parts = []
    for h, v in r.values:
        col = short_col(h)
        if col in ("points", "payment example", "min. to open", "min. to earn") and col not in want:
            continue
        if v in ("-", "—", "N/A", "None", ""):
            continue
        if col in ("APR", "APY") and col in v:
            parts.append(v)
        elif col == "rate":
            parts.append(f"{v} interest rate")
        else:
            parts.append(f"{v} {col}")
    return ", ".join(parts)


def format_match(m: RateMatch, q: QFeat) -> str:
    out = []
    for tf, rows in m.tables:
        lines = [f"{row_label(tf, r)}: {row_values(r, q.want_cols)}" for r in rows[:14]]
        eff = f" (rates effective {tf.table.effective_raw})" if tf.table.effective_raw else ""
        out.append("; ".join(lines) + eff + ".")
    note = _footnote_for(m, q)
    if note:
        out.append(note)
    return " ".join(out)


def _footnote_for(m: RateMatch, q: QFeat) -> Optional[str]:
    """If the question asks about a condition (credit score, LTV, closing costs, Mini-Max...), quote the
    footnote sentence under the table that covers it."""
    tf = m.tables[0][0]
    extra = [w for w in set(q.raw_prod) | {w for w in content_tokens(q.text) if w not in GENERIC and w not in STOP}
             if w not in tf.toks and not w.isdigit() and len(w) > 2 and w not in _FILLER]
    if not extra or not tf.table.notes:
        return None
    best, best_n = None, 0.0
    for s in re.split(r"(?<=[.;])\s+", tf.table.notes):
        st = set(tokens(s))
        n = sum(1 for w in extra if w in st)
        n += 0.5 * bool(n and re.search(r"\d", s))       # conditions usually come with a number
        if n > best_n and len(s) < 400:
            best, best_n = re.sub(r"^[^A-Za-z]+", "", s.strip()), n
    if best and best_n >= max(1, (len(extra) + 1) // 2):
        return best
    return None


def sources_for(m: RateMatch, index: RateIndex) -> list[str]:
    """Cite the table's page plus every other official page that shows the same numbers."""
    urls: list[str] = []
    for tf, rows in m.tables:
        urls.append(tf.table.url)
        for u in tf.table.detail_links:
            urls.append(u)
        sig = {r.signature for r in rows}
        for other in index.tables:
            if other is tf or other.authority < 0.85:
                continue
            if re.sub(r"[^a-z]", "", other.name.lower()) != re.sub(r"[^a-z]", "", tf.name.lower()):
                continue
            if sig <= {r.signature for r in other.rows}:
                urls.append(other.table.url)
    seen, out = set(), []
    for u in urls:
        k = canonical_url(u)
        if k not in seen:
            seen.add(k)
            out.append(u)
    return out[:5]

"""
A knowledge system that knows what is true *now* on a credit-union website crawled several times.

    ks = build()
    ks.ingest(snapshot)            # oldest -> newest
    ks.answer(question, as_of)     # -> Answer
    ks.conflicts()                 # -> list[Conflict]

Design (details in NOTES.md):
  * Every snapshot is kept whole. `as_of` picks the last snapshot taken on or before that date; no date
    means the newest snapshot.
  * Rate questions are answered from parsed rate tables, preferring the official rate sheet over product
    pages, landing pages, calculators, PDFs and articles, and filtered to the exact tier asked for.
  * Everything else goes through BM25 over cleaned page chunks, weighted by page authority and by article
    publish date (the newest press release wins), plus a few structured extractors (branch addresses and
    hours, routing number, links).
  * When nothing on the site supports an answer we say so instead of guessing.
Standard library only; no API keys.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
for _d in [_HERE, *_HERE.parents]:
    if (_d / "harness" / "contract.py").exists():
        sys.path.insert(0, str(_d / "harness"))
        break

from contract import Answer, Conflict, Snapshot, Source  # noqa: E402

from kb_conflicts import find_conflicts  # noqa: E402
from kb_parse import PageRec, build_page, canonical_url, normalize_text, skip_url, url_path  # noqa: E402
from kb_rates import RateIndex, format_match, is_rate_question, parse_question, sources_for  # noqa: E402
from kb_text import BM25, content_tokens, expand, stem, tokens  # noqa: E402

REFUSAL = ("I'm sorry, I don't know: the Dupaco website doesn't list {what}. "
           "Please contact Dupaco directly (call, visit a branch or use the Contact Us page) for that information.")

KIND_WEIGHT = {"product": 1.0, "rates": 0.9, "news": 0.85, "landing": 0.6, "blog": 0.6, "bizres": 0.5,
               "file": 0.45, "calculator": 0.35, "listing": 0.15}


class SnapState:
    def __init__(self, snapshot_id: str, taken_on: str, pages: dict[str, PageRec]):
        self.id = snapshot_id
        self.date = taken_on
        self.pages = pages
        self.rates = RateIndex(pages)
        # Articles re-published under the same title (yearly award releases, annual recaps...): only the
        # newest one describes the current state; the older ones are superseded.
        newest: dict[str, str] = {}
        for p in pages.values():
            if p.date and p.kind in ("news", "blog", "bizres"):
                k = re.sub(r"[^a-z0-9]", "", p.title.lower())
                newest[k] = max(newest.get(k, ""), p.date)
        self.superseded = {u for u, p in pages.items() if p.date and p.kind in ("news", "blog", "bizres")
                           and newest.get(re.sub(r"[^a-z0-9]", "", p.title.lower()), "") > p.date}
        self.carried = {u for u, p in pages.items() if p.carried_from}
        self.chunks = []
        docs = []
        for p in pages.values():
            if p.kind == "listing":
                continue
            for c in p.chunks:
                if _is_data_dump(c.text):
                    continue
                self.chunks.append(c)
                docs.append(expand(content_tokens(c.title + " " + c.heading + " " + c.text)))
        self.bm25 = BM25(docs)


def _is_data_dump(text: str) -> bool:
    nums = len(re.findall(r"\d+", text))
    words = len(re.findall(r"[A-Za-z]{3,}", text))
    return nums > 30 and nums > words


def _boilerplate(snapshot: Snapshot) -> set[str]:
    c = Counter()
    for p in snapshot.pages:
        c.update({ln.strip() for ln in normalize_text(p.markdown).split("\n") if ln.strip()})
    return {ln for ln, k in c.items() if k >= 25}


def _year_frac(iso: Optional[str]) -> Optional[float]:
    if not iso:
        return None
    y, m, d = (int(x) for x in iso.split("-"))
    return y + (m - 1) / 12 + (d - 1) / 365


class KnowledgeSystem:
    def __init__(self):
        self.snaps: list[SnapState] = []
        self._conflicts: Optional[list[Conflict]] = None
        self._cache: dict = {}

    # ---------------------------------------------------------------------------------- ingest
    def ingest(self, snapshot: Snapshot) -> None:
        boiler = _boilerplate(snapshot)
        pages: dict[str, PageRec] = {}
        for p in snapshot.pages:
            if skip_url(p.url):
                continue
            rec = build_page(p.url, p.title, p.markdown, boiler)
            key = canonical_url(p.url) + ("?" + p.url.split("?", 1)[1] if "?" in p.url else "")
            if key in pages and len(pages[key].text) >= len(rec.text):
                continue
            pages[key] = rec
        # A page missing from this crawl is not proof it was taken down (crawls are incomplete), so the
        # last known copy is kept as a low-priority fallback.
        if self.snaps and self.snaps[-1].date < snapshot.taken_on:
            prev = self.snaps[-1]
            for key, rec in prev.pages.items():
                if key not in pages and rec.kind not in ("listing",):
                    pages[key] = replace(rec, carried_from=rec.carried_from or prev.id)
        self.snaps.append(SnapState(snapshot.snapshot_id, snapshot.taken_on, pages))
        self.snaps.sort(key=lambda s: s.date)
        self._conflicts = None
        self._cache.clear()

    # ---------------------------------------------------------------------------------- answer
    def _snap_for(self, as_of: Optional[str]) -> SnapState:
        if not as_of:
            return self.snaps[-1]
        eligible = [s for s in self.snaps if s.date <= as_of]
        return eligible[-1] if eligible else self.snaps[0]   # before the first crawl: oldest we have

    def answer(self, question: str, as_of: Optional[str] = None) -> Answer:
        if not self.snaps:
            return Answer(text="I don't know yet: no website snapshots have been loaded.", confident=False)
        question = str(question or "")
        as_of = _iso_date(as_of) or _date_in_question(question, self.snaps[-1].date)
        key = (question.strip().lower(), as_of)
        if key not in self._cache:
            snap = self._snap_for(as_of)
            try:
                a = self._answer(question, snap)
            except Exception:                       # never crash a member conversation
                try:
                    a = self._text_answer(question, snap)
                except Exception:
                    a = Answer(text=REFUSAL.format(what="an answer to that question"), confident=False)
            self._cache[key] = a
        return self._cache[key]

    def _answer(self, q: str, snap: SnapState) -> Answer:
        ql = q.lower()
        for handler in (self._routing, self._url_question, self._branch_address, self._hours, self._eligibility,
                        self._product_list):
            a = handler(q, ql, snap)
            if a is not None:
                return a
        if is_rate_question(q):
            a = self._rate_answer(q, snap)
            if a is not None:
                return a
        return self._text_answer(q, snap)

    def _product_list(self, q: str, ql: str, snap: SnapState) -> Optional[Answer]:
        """'What kinds of savings accounts / credit cards / loans do you offer?' -> the products on the sheet."""
        if is_rate_question(q) or not re.search(r"\b(what|which)\b.*\b(types?|kinds?|sorts?)\s+of\b|"
                                                r"\b(what|which)\b.*\bdo(es)? (you|dupaco) (offer|have)\b|\blist\b", ql):
            return None
        noun = next((n for n in ("certificate", "cd", "savings", "checking", "credit card", "card", "ira",
                                 "money market", "loan") if re.search(rf"\b{n}s?\b", ql)), None)
        if not noun:
            return None
        key = {"cd": "certificate", "credit card": "visa", "card": "visa"}.get(noun, noun)
        biz = "business" in ql
        names: list[str] = []
        urls: list[str] = []
        for tf in snap.rates.tables:
            if tf.authority < 0.85:
                continue
            t = tf.table
            hay = " ".join([tf.name, " ".join(t.header), " ".join(r.label_text for r in t.rows)]).lower()
            if key == "certificate":
                hay += " " + t.section.lower()
            is_biz = "business" in hay or "/business" in t.url
            if key == "loan" and not re.search(r"\bloans?\b|mortgage|heloc|equity", hay):
                continue
            if key != "loan" and key not in hay:
                continue
            if is_biz and not biz and key != "visa" or biz and not is_biz:
                continue
            if "ira" in tf.flags and key not in ("ira",):
                continue
            if key == "savings" and re.search(r"certificate|checking", hay):
                continue
            name = tf.name if not re.fullmatch(r"(rates?|special|regular)", tf.name, re.I) else \
                f"{tf.name} {t.section}".strip()
            name = re.sub(r"\s+(account\s+)?rates?$", "", name, flags=re.I).strip()
            if not name or re.fullmatch(r"(savings?|checking)( account)?", name, re.I):
                continue
            if name.lower() not in (n.lower() for n in names):
                names.append(name)
                urls.append(t.url)
        if len(names) < 2:
            return None
        return Answer(text=f"Dupaco's current {noun} options on its rate sheets: " + "; ".join(names) +
                           ". Ask about any of them for the current rate.",
                      sources=[Source(url=u, snapshot_id=snap.id) for u in dict.fromkeys(urls)])

    # -- rates -----------------------------------------------------------------------------
    def _rate_answer(self, q: str, snap: SnapState) -> Optional[Answer]:
        qf = parse_question(q)
        m = snap.rates.match(qf)
        if not m.tables:
            if m.refused_reason and m.refused_reason.startswith("no ") and m.refused_reason.endswith("listed") \
                    and not _asks_specific_rate(q):
                return None     # e.g. "What is the Mortgage Rate Relief program?": describe it from text
            if not (m.refused_reason or "").startswith("product not on the rate sheet:") \
                    and m.refused_reason not in ("term not offered", "model year not listed"):
                a = self._text_rate_answer(q, qf, snap)
                if a is not None:
                    return a
            if m.refused_reason and m.refused_reason != "no matching product" or _asks_specific_rate(q):
                what = "a rate for that product or term" if m.refused_reason != "term not offered" \
                    else "a rate for that term"
                return Answer(text=REFUSAL.format(what=what), confident=False,
                              conflict_note=f"no supporting rate table ({m.refused_reason})")
            return None
        text = format_match(m, qf)
        if CHANGE_RE.search(q):
            text += self._history_note(qf, m, snap)
        urls = sources_for(m, snap.rates)
        srcs = [Source(url=u, snapshot_id=snap.id) for u in urls]
        if srcs:
            tf, rows = m.tables[0]
            srcs[0].quote = " | ".join(rows[0].labels + [v for _, v in rows[0].values])[:200]
        note = self._stale_note(m, snap)
        return Answer(text=text, sources=srcs, confident=True, conflict_note=note)

    def _text_rate_answer(self, q: str, qf, snap: SnapState) -> Optional[Answer]:
        """A rate stated in a sentence on an official page ("6-month certificate APY: 3.50%")."""
        ranked = self._rank_chunks(q, snap, kinds={"product", "rates"})[:25]
        if not ranked:
            return None
        top = ranked[0][0]
        best = None
        for s, i in ranked:
            c = snap.chunks[i]
            if c.pos >= 50:
                continue    # table chunks were already handled by the table engine
            for st in split_sentences(c.text):
                # a percentage that is a rate ("APY: 3.50%", "5.39% APR"), not "20% down payment"
                if len(st) > 400 or not re.search(
                        r"\b(rate|apr|apy|yield|dividend|interest)\b[^.%\d]{0,40}\d+(\.\d+)?\s*%|"
                        r"\d+(\.\d+)?\s*%\s*(\(?\s*)?(apr|apy|annual percentage|interest|dividend|fixed|variable)",
                        st, re.I):
                    continue
                if qf.terms and not any(re.search(rf"\b{t}[\s-]*(month|mo)\b|\b{t // 12}[\s-]*year", st, re.I)
                                        for t in qf.terms):
                    continue
                cov = _coverage(q, st + " " + c.title + " " + c.heading, snap.bm25.idf)
                if cov < 0.6:
                    continue
                if re.search(r"offer ended|\bexpired?\b|no longer (available|offered)|\bended\b|limited time",
                             st, re.I):
                    continue    # a promotion that is over is not today's rate
                sc = cov + 0.5 * s / top - (0.3 if canonical_url(c.url) in snap.carried else 0.0)
                if best is None or sc > best[0]:
                    best = (sc, st, c)
        if not best:
            return None
        _, st, c = best
        return Answer(text=st, sources=[Source(url=c.url, snapshot_id=snap.id, quote=st[:200])])

    def _history_note(self, qf, m, snap: SnapState) -> str:
        """For 'has it changed?' questions: the earlier values of the same rows on older snapshots."""
        tf, rows = m.tables[0]
        if len(rows) > 2:
            return ""
        cur = [v for r in rows for _, v in r.values if "%" in v]
        notes = []
        for old in self.snaps:
            if old.date >= snap.date:
                break
            om = old.rates.match(qf)
            if not om.tables:
                continue
            ovals = [v for r in om.tables[0][1] for _, v in r.values if "%" in v]
            if not ovals:
                continue
            if notes and notes[-1][1] == ovals:
                notes[-1] = (old.date, ovals)        # same old value, seen on a later snapshot
            elif ovals != cur:
                notes.append((old.date, ovals))
            else:
                notes.clear()                        # already at today's value back then
        if not notes:
            return f" It has not changed across the snapshots on file (since {self.snaps[0].date})."
        d, vals = notes[-1]
        return (f" This has changed: the site still showed {', '.join(vals)} as of {d}; "
                f"the current figure above replaces it.")

    @staticmethod
    def _stale_note(m, snap: SnapState) -> Optional[str]:
        tf, rows = m.tables[0]
        mine = {v for r in rows for _, v in r.values if "%" in v}
        stale = []
        for other in snap.rates.tables:
            if other.authority >= tf.authority or other is tf:
                continue
            if re.sub(r"[^a-z]", "", other.name.lower()) != re.sub(r"[^a-z]", "", tf.name.lower()):
                continue
            for r in other.rows:
                if any(r.term == x.term and r.term for x in rows) or len(rows) == len(other.rows) == 1:
                    vals = {v for _, v in r.values if "%" in v} - mine
                    if vals:
                        stale.append(f"{url_path(other.table.url)} shows {', '.join(sorted(vals))}")
        if stale:
            return ("Other pages disagree (" + "; ".join(stale[:3]) + "); the official rate sheet "
                    f"({url_path(tf.table.url)}) is treated as authoritative.")
        return None

    # -- structured extractors ---------------------------------------------------------------
    def _routing(self, q: str, ql: str, snap: SnapState) -> Optional[Answer]:
        if "routing" not in ql and "aba" not in ql.split():
            return None
        found: Counter = Counter()
        where: dict[str, list[str]] = {}
        for url, p in snap.pages.items():
            for m in re.finditer(r"routing (?:number|#|no\.?)?[^0-9]{0,20}(\d{9})\b", p.text, re.I):
                found[m.group(1)] += KIND_WEIGHT.get(p.kind, 0.3)
                where.setdefault(m.group(1), []).append(p.url)
        if not found:
            return Answer(text=REFUSAL.format(what="a routing number"), confident=False)
        num = found.most_common(1)[0][0]
        urls = sorted(set(where[num]), key=lambda u: (snap.pages[canonical_url(u)].kind != "product"
                                                     if canonical_url(u) in snap.pages else True, len(u)))
        return Answer(text=f"Dupaco's routing number is {num}.",
                      sources=[Source(url=u, snapshot_id=snap.id, quote=f"Routing Number: {num}") for u in urls[:5]])

    def _url_question(self, q: str, ql: str, snap: SnapState) -> Optional[Answer]:
        if not re.search(r"\b(link|url|web ?address|web ?page|website|site address)\b", ql):
            return None
        core = re.sub(r"\b(link|url|website|web|page|address|site|use|see|dupaco'?s?|what|which|should|i)\b", " ", ql)
        # A rate page for a named product: point at that product's page (where the rate table lives).
        if re.search(r"\brates?\b|\bapy\b|\bapr\b", ql):
            m = snap.rates.match(parse_question(core))
            if m.tables:
                tf = m.tables[0][0]
                prod = tf.table.detail_links[0] if tf.table.detail_links else tf.table.url
                prod = prod.rstrip("/")
                txt = f"You can see them at {prod}#rates"
                if url_path(prod) != "/rates":
                    txt += " (the full rate sheet is at https://www.dupaco.com/rates)"
                return Answer(text=txt + ".", sources=[Source(url=prod + "#rates", snapshot_id=snap.id),
                                                       Source(url=tf.table.url, snapshot_id=snap.id)])
        qt = set(expand(content_tokens(core)))
        page_score: dict[str, float] = {}
        for s, i in self._rank_chunks(core, snap, kinds={"product", "rates"})[:60]:
            u = snap.chunks[i].url
            page_score[u] = max(page_score.get(u, 0.0), s)
        if not page_score:
            return None
        top = max(page_score.values())
        best, best_s = None, -1.0
        for u, s in page_score.items():
            p = snap.pages.get(canonical_url(u))
            if p is None or "?" in u:
                continue
            path_t = set(tokens(url_path(u).replace("/", " ").replace("-", " ")))
            title_t = set(expand(content_tokens(p.title)))
            sc = s / top + 0.6 * len(qt & path_t) + 0.3 * len(qt & title_t) - 0.08 * url_path(u).count("/")
            if sc > best_s:
                best, best_s = p, sc
        if not best:
            return None
        url = best.url.rstrip("/")
        return Answer(text=f"You can find it at {url}.", sources=[Source(url=best.url, snapshot_id=snap.id)])

    def _locations(self, snap: SnapState) -> list[tuple[str, str, str]]:
        """(name, kind, address) from the branch locator on the contact page."""
        out = []
        for url, p in snap.pages.items():
            if url_path(url) != "/contact":
                continue
            lines = [ln.strip() for ln in p.lines if ln.strip()]
            for i, ln in enumerate(lines):
                if ln.startswith("### ") and i + 2 < len(lines):
                    kind, addr = lines[i + 1], lines[i + 2]
                    if re.search(r"\d", addr) and re.search(r",\s*[A-Z]{2},\s*\d{5}", addr):
                        out.append((ln[4:].strip(), kind, addr))
        return out

    def _branch_address(self, q: str, ql: str, snap: SnapState) -> Optional[Answer]:
        if not re.search(r"\baddress|where is|located|location of|directions", ql):
            return None
        if not re.search(r"\bbranch|office|atm|location", ql):
            return None
        name_part = re.sub(r"\b(what|what's|whats|is|are|the|address|addresses|of|for|where|located|location|dupaco|"
                           r"dupaco's|branch|branches|office|atm|atms|your|can|you|tell|me|give|please|find|exact|"
                           r"street|avenue|ave|st|rd|road|in|at|a|an|how|do|i|get|to|directions|there|one)\b",
                           " ", ql.replace("?", " "))
        words = [w for w in re.findall(r"[a-z0-9\-]+", name_part) if len(w) > 2]
        want = [stem(w) for w in words]
        if not want:
            return None
        wants_atm = bool(re.search(r"\batms?\b", ql))
        place = " ".join(w.capitalize() for w in words)
        best, best_s = None, 0
        for name, kind, addr in self._locations(snap):
            if ("ATM" in kind) != wants_atm:
                continue
            at = set(tokens(addr + " " + name))
            s = sum(1 for w in want if w in at)
            if s > best_s:
                best, best_s = (name, kind, addr), s
        if best and best_s >= max(1, (len(want) + 1) // 2):
            name, kind, addr = best
            return Answer(text=f"The {place} {'ATM' if wants_atm else 'branch'} is at {addr}.",
                          sources=[Source(url="https://www.dupaco.com/contact", snapshot_id=snap.id, quote=addr)])
        # Not in the locator list: look for "<place> branch ... located at <street address>" in articles.
        street = r"\d{2,6}\s+(?:[NSEW]\.?\s+)?[A-Z0-9][\w.]*(?:\s+[A-Z0-9][\w.]*){0,4}\s+(?:St|Street|Ave|Avenue|Rd|Road|Dr|" \
                 r"Drive|Hwy|Highway|Blvd|Boulevard|Ln|Lane|Way|Pkwy|Parkway|Ct|Court|Pl|Place|Plaza)\b\.?(?:\s*\d+)?"
        found = None
        for c in snap.chunks:
            low = c.text.lower()
            if not all(w in low for w in words):
                continue
            for st in split_sentences(c.text):
                m = re.search(street, st)
                if m and all(w in st.lower() for w in words) and re.search(r"branch|office|location|learning lab", st, re.I):
                    if found is None or (c.date or "9999") > (found[2] or "9999"):
                        found = (st, c, c.date)
        if found:
            st, c, _ = found
            return Answer(text=st, sources=[Source(url=c.url, snapshot_id=snap.id, quote=st[:200])])
        return Answer(text=REFUSAL.format(what=f"an address for a '{place}' {'ATM' if wants_atm else 'branch'}"),
                      confident=False)

    def _hour_blocks(self, snap: SnapState) -> list[dict]:
        """Branch schedules published on the site: {'url','date','where','lobby':[...], 'drive':[...]}"""
        out = []
        for url, p in snap.pages.items():
            if "Lobby" not in p.text and "Drive-up" not in p.text:
                continue
            lines = [strip for strip in (ln.strip().replace("**", "") for ln in p.lines) if strip]
            for i, ln in enumerate(lines):
                if not re.match(r"^Lobby\b", ln):
                    continue
                block = {"url": p.url, "date": p.date, "lobby": [], "drive": [], "where": ""}
                ctx = " ".join(lines[max(0, i - 3):i])
                block["where"] = ctx
                cur = "lobby"
                for ln2 in [ln[5:].strip()] + lines[i + 1:i + 14]:
                    if re.match(r"^Drive[- ]?(up|thru|through)", ln2, re.I):
                        cur = "drive"
                        ln2 = re.sub(r"^Drive[- ]?(up|thru|through)\s*", "", ln2, flags=re.I)
                    if re.match(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*", ln2) and ":" in ln2:
                        block[cur].append(ln2)
                    elif ln2 and not re.match(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)", ln2) and block["drive"]:
                        break
                if block["lobby"] or block["drive"]:
                    out.append(block)
        return out

    def _hours(self, q: str, ql: str, snap: SnapState) -> Optional[Answer]:
        if not re.search(r"\bhours?\b|\bopen\b.*\b(saturday|sunday|weekend|weekday|until|late|today|tonight)|"
                         r"what time|\bwhen (are|is|do|does)\b.*\b(open|close)\b", ql):
            return None
        if re.search(r"\bopen (an?|my|the|your) [a-z ]*(account|loan|card|certificate|membership)|\bonline\b|"
                     r"\b(still|permanently) (open|clos)|\bclos(ed|ing|ure)\b.*\b(branch|location)\b.*\b(20\d\d|why)\b",
                     ql):
            return None
        if re.search(r"\b(call|phone|contact center|member service|online|chat)\b", ql):
            return None
        blocks = self._hour_blocks(snap)
        if not blocks:
            return None
        drive = bool(re.search(r"drive|drive-up|drive-thru|drive thru", ql))
        # specific branch named?
        stop = set(tokens("what are the your dupaco lobby drive up thru hours branch branches for at of is do all "
                          "keep same open general typical time does on saturday sunday weekend can you give me "
                          "tell please located street avenue"))
        want = [w for w in content_tokens(ql) if w not in stop and len(w) > 2]
        if want:
            hits = [b for b in blocks if all(w in tokens(b["where"] + " " + snap.pages[canonical_url(b["url"])].title
                                                         + " " + snap.pages[canonical_url(b["url"])].text[:1500])
                                             for w in want)]
            if not hits:
                if not re.search(r"\bhours?\b|what time", ql):
                    return None     # "when does the X branch close?" may be about a closure, not hours
                return Answer(text=REFUSAL.format(what=f"hours for a '{' '.join(want)}' branch"), confident=False)
            blocks = hits
        key = "drive" if drive else "lobby"
        sched = Counter()
        first_url: dict[tuple, str] = {}
        for b in sorted(blocks, key=lambda b: b["date"] or "", reverse=True):
            s = tuple(b[key])
            if s:
                sched[s] += 1
                first_url.setdefault(s, b["url"])
        if not sched:
            return None
        ordered = [s for s, _ in sched.most_common()]
        label = "Drive-up" if drive else "Lobby"
        if len(ordered) == 1 or want:
            text = f"{label} hours: " + "; ".join(ordered[0]) + "."
        else:
            alts = " Other branches use: " + " | ".join("; ".join(s) for s in ordered[1:4]) + "."
            text = (f"{label} hours vary by branch. The most common schedule is: " + "; ".join(ordered[0]) + "."
                    + alts + " Which branch would you like hours for?")
        srcs = [Source(url=first_url[s], snapshot_id=snap.id) for s in ordered[:3]]
        srcs.append(Source(url="https://www.dupaco.com/contact", snapshot_id=snap.id))   # live branch locator
        return Answer(text=text, sources=srcs)

    def _eligibility(self, q: str, ql: str, snap: SnapState) -> Optional[Answer]:
        """Membership eligibility: the official 'who can join' list plus what the charter area covers."""
        if not re.search(r"\beligib|who can (join|become)|can (i|we|my [a-z]+) (join|become a member)|"
                         r"qualif\w* (for|to) (join|membership|become)|requirements? (to|for) (join|membership)|"
                         r"(cost|how much|deposit|need|minimum)\b.{0,25}\b(to join|to become a member|membership)|"
                         r"membership requirement|requirements? for (membership|joining)|"
                         r"how (do|can) i (join|become a (dupaco )?member|apply (for|to) (membership|join))|"
                         r"(open|apply for) (a )?membership", ql):
            return None
        rules = None
        for c in snap.chunks:
            if c.kind == "product" and re.search(r"eligib", c.heading, re.I) and "join" in c.url:
                rules = c
                break
        if rules is None:
            return None
        lines = [ln.strip(" -") for ln in rules.text.split("\n") if ln.strip(" -")]
        text = (rules.heading.rstrip(".") + ": " + "; ".join(lines[:4])).replace("..", ".")
        srcs = [Source(url=rules.url, snapshot_id=snap.id, quote=lines[0][:200] if lines else None)]
        area = None
        for c in snap.chunks:
            m = re.search(r"[^.]*serves (residents|members) in [^.]*counties[^.]*\.", c.text)
            if m and (area is None or (c.date or "") > (area[1] or "")):
                area = (m.group(0).strip(), c.date, c.url)
        if area:
            text += " For reference: " + area[0]
            srcs.append(Source(url=area[2], snapshot_id=snap.id, quote=area[0][:200]))
        page = snap.pages.get(canonical_url(rules.url))
        join = next((u for _, u in (page.links if page else []) if re.match(r"https?://join\.", u)), None)
        if join:
            text += f" You can apply online at {join}"
        return Answer(text=text, sources=srcs)

    # -- free text ---------------------------------------------------------------------------
    def _rank_chunks(self, q: str, snap: SnapState, kinds: Optional[set] = None) -> list[tuple[float, int]]:
        qt = expand(content_tokens(q))
        raw = snap.bm25.scores(qt) if qt else {}
        latest = _year_frac(snap.date) or 2026.0
        qset = set(qt)
        ranked = []
        for i, s in raw.items():
            c = snap.chunks[i]
            if kinds and c.kind not in kinds:
                continue
            w = KIND_WEIGHT.get(c.kind, 0.4)
            if c.date and canonical_url(c.url) in snap.superseded:
                w *= 0.35
            if canonical_url(c.url) in snap.carried:
                w *= 0.5
            yf = _year_frac(c.date)
            rec = 1.0 if yf is None else max(0.4, 1.0 - 0.1 * (latest - yf))
            ttl = len(qset & set(expand(content_tokens(c.title)))) / max(1, len(qset))
            cov = len(qset & set(expand(content_tokens(c.heading + " " + c.text)))) / max(1, len(qset))
            ranked.append((s * w * rec * (0.4 + 0.6 * cov) * (1 + 0.3 * ttl), i))
        ranked.sort(reverse=True)
        return ranked

    def _text_answer(self, q: str, snap: SnapState) -> Answer:
        ranked = self._rank_chunks(q, snap)
        if not ranked:
            return Answer(text=REFUSAL.format(what="an answer to that question"), confident=False)
        qset = set(expand(content_tokens(q)))
        if re.search(r"^\s*(how many|what is the number of|what's the number of)\b|\bnumber of\b|\btotal number\b",
                     q, re.I):
            a = self._count_answer(q, snap, ranked, qset)
            if a is not None:
                return a
        if PHONE_INTENT_RE.search(q):
            # the number itself answers "phone/call"; rank on what the number is for
            rest = " ".join(w for w in re.findall(r"[A-Za-z']+", q) if stem(w.lower()) not in _PHONE_WORDS)
            ranked2 = self._rank_chunks(rest, snap) if content_tokens(rest) else ranked
            with_phone = [(s, i) for s, i in ranked2 if PHONE_RE.search(snap.chunks[i].text)]
            ranked = with_phone or ranked
            qset = set(expand(content_tokens(rest))) or qset
        # rate tables are for rate questions; descriptive text answers "what is / how do I"
        ranked = sorted(((s * (0.6 if snap.chunks[i].pos >= 50 else 1.0), i) for s, i in ranked[:60]), reverse=True)
        top_s = ranked[0][0]
        best = None
        for s, i in ranked[:6]:
            if s < 0.5 * top_s:
                break
            c = snap.chunks[i]
            text, wsc = _best_window(c.text, qset, q, snap.bm25.idf)
            total = wsc + 5.0 * s / top_s
            if best is None or total > best[0]:
                best = (total, text, c)
        _, text, c = best
        if _coverage(q, text + " " + c.title + " " + c.heading, snap.bm25.idf) < 0.4:
            return Answer(text=REFUSAL.format(what="that information"), confident=False,
                          conflict_note=f"closest passage ({c.url}) does not answer the question")
        srcs = [Source(url=c.url, snapshot_id=snap.id, quote=text[:200])]
        for s, i in ranked[1:4]:
            u = snap.chunks[i].url
            if s >= 0.6 * top_s and u not in {x.url for x in srcs}:
                srcs.append(Source(url=u, snapshot_id=snap.id))
        return Answer(text=text, sources=srcs, confident=True)

    def _ordinal_answer(self, q: str, snap: SnapState, qset: set) -> Optional[Answer]:
        """'How many times has X happened': the newest '<ordinal> time' statement about it."""
        rx = re.compile(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|\d+(st|nd|rd|th))"
                        r"\s+(consecutive\s+)?(time|year)\b", re.I)
        best = None
        for c in snap.chunks:
            if not rx.search(c.text):
                continue
            for st in split_sentences(c.text):
                if not rx.search(st):
                    continue
                cov = len(qset & set(expand(content_tokens(st + " " + c.title)))) / max(1, len(qset))
                key = (round(cov, 2), c.date or "", len(st) < 300)
                if cov >= 0.5 and (best is None or key > best[0]):
                    best = (key, st, c)
        if not best:
            return None
        _, st, c = best
        return Answer(text=st, sources=[Source(url=c.url, snapshot_id=snap.id, quote=st[:200])],
                      conflict_note="Older releases give lower counts; the most recent release is used.")

    def _count_answer(self, q: str, snap: SnapState, ranked, qset) -> Optional[Answer]:
        """'How many X ...': find '<number> ... X' statements; the newest dated statement wins."""
        m = re.search(r"(?:how many|number of)\s+([a-z' -]+)", q.lower())
        if not m:
            return None
        noun_toks = [w for w in content_tokens(m.group(1))][:3]
        noun = next((w for w in noun_toks if w not in ("dupaco",)), None)
        if not noun:
            return None
        if noun == stem("times"):
            return self._ordinal_answer(q, snap, qset - {noun})
        root = noun[:-1] if noun.endswith("y") and len(noun) > 4 else noun
        num = r"(\d{1,3}(?:,\d{3})+|\d+)\+?"
        # "22 branch office locations" / "membership has grown to more than 181,000"
        pat = re.compile(r"\b" + num + r"\s+(?:[A-Za-z'\-]+\s+)?(" + re.escape(root) + r"[a-z]*)\b"
                         r"|\b(" + re.escape(root) + r"[a-z]*)\b(?:\s+[A-Za-z'\-]+){0,5}?\s+(?:more than|over|nearly|"
                         r"about|of|to|totals?|exceed\w*)\s+\$?" + num + r"\b(?!\s*(?:%|percent))", re.I)
        score_of = {i: s for s, i in ranked}
        cands = []
        for i, c in enumerate(snap.chunks):
            s = score_of.get(i, 0.0)
            if root not in c.text.lower():
                continue
            for st in split_sentences(c.text):
                mm = pat.search(st)
                if not mm:
                    continue
                n = mm.group(1) or mm.group(4)
                if re.fullmatch(r"(19|20)\d\d", n) or re.search(r"\d+ [A-Z][a-z]+ (St|Ave|Rd|Dr|Hwy|Blvd)|"
                                                                r", [A-Z]{2},? \d{5}", st):
                    continue
                if mm.group(1):
                    # "150,000 member transactions": the noun only modifies another noun
                    nxt = re.match(r"\s+([a-z]+)", st[mm.end():])
                    if nxt and nxt.group(1) not in _COUNT_CONT and stem(nxt.group(1)) not in qset:
                        continue
                cov = len(qset & (set(expand(content_tokens(st))) | {noun})) / max(1, len(qset))
                yf = _year_frac(c.date) or (2025.0 if c.kind != "rates" else 2026.5)
                official = c.kind in ("news", "product", "rates")   # press releases state org facts
                cands.append((cov, official, yf, s, st, c))
        if not cands:
            return None
        best_cov = max(x[0] for x in cands)
        pool = [x for x in cands if x[0] >= best_cov - 0.15]
        pool.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
        cov, _, yf, s, st, c = pool[0]
        text = st if len(st) < 450 else st[:450]
        return Answer(text=text, sources=[Source(url=c.url, snapshot_id=snap.id, quote=st[:200])],
                      conflict_note="Counts differ across older articles; the most recently published one is used.")

    # ---------------------------------------------------------------------------------- conflicts
    def conflicts(self) -> list[Conflict]:
        if self._conflicts is None:
            self._conflicts = find_conflicts(self.snaps)
        return list(self._conflicts)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
           "november", "december"]


def _iso_date(as_of) -> Optional[str]:
    """'2026-05-31', '2026-05-31T10:00:00Z' or a date object -> '2026-05-31'; anything else -> None."""
    if not as_of:
        return None
    m = re.match(r"\s*(\d{4})-(\d{1,2})-(\d{1,2})", str(as_of))
    return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}" if m else None


def _date_in_question(q: str, latest: str) -> Optional[str]:
    """'as of May 2026' / 'in June 2026' in the wording itself -> end of that month (past dates only)."""
    m = re.search(r"\b(?:as of|in|during|back in|on)\s+(" + "|".join(_MONTHS) + r")\s+(?:(\d{1,2}),?\s+)?(20\d\d)\b",
                  q, re.I)
    if not m:
        return None
    mon = _MONTHS.index(m.group(1).lower()) + 1
    year = int(m.group(3))
    day = int(m.group(2)) if m.group(2) else 31 if mon in (1, 3, 5, 7, 8, 10, 12) else 30 if mon != 2 else 28
    iso = f"{year:04d}-{mon:02d}-{day:02d}"
    return iso if iso < latest else None


_DATE_PHRASE_RE = re.compile(r"\b(as of|in|during|on|back in|since)?\s*(january|february|march|april|may|june|july|"
                              r"august|september|october|november|december)\b(\s+\d{1,2},?)?(\s+20\d\d)?|\b20\d\d\b|"
                              r"\b(current(ly)?|today|right now|now|latest)\b", re.I)
# credit-union words that qualify rather than name a product ("share certificate" = certificate)
_SOFT = {stem(w) for w in ("share", "account", "accounts", "credit", "union", "dupaco", "member", "members")}
_NUM_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
              "eight": "8", "nine": "9", "ten": "10", "twelve": "12", "eighteen": "18", "twenty": "20",
              "thirty": "30", "sixty": "60"}


def _coverage(q: str, text: str, idf: dict) -> float:
    """Share of the question's meaningful words (idf-weighted) that the answer text actually contains."""
    q = _DATE_PHRASE_RE.sub(" ", q)
    q = re.sub(r"\b(" + "|".join(_NUM_WORDS) + r")\b", lambda m: _NUM_WORDS[m.group(1).lower()], q, flags=re.I)
    qt = [w for w in set(content_tokens(q)) if not w.isdigit() and w not in _SOFT]
    if not qt:
        return 1.0
    tt = set(expand(content_tokens(text)))
    tot = sum(idf.get(w, 2.0) for w in qt)
    hit = sum(idf.get(w, 2.0) for w in qt if w in tt or any(s in tt for s in expand([w])[1:]))
    return hit / tot if tot else 1.0


def _asks_specific_rate(q: str) -> bool:
    if re.search(r"\bprogram\b", q, re.I):
        return False
    return bool(re.search(r"\b(what|what's|whats|how much)\b.*\b(rate|apr|apy)\b", q, re.I))


_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\-])|\n+")
CHANGE_RE = re.compile(r"\b(chang(e|ed|es)|increase[ds]?|decrease[ds]?|go(ne)? (up|down)|went (up|down)|"
                       r"rais(e|ed)|lower(ed)?|cut|previous(ly)?|used to be|before|old rate|differ(ent)?|"
                       r"since (may|june|july|august|spring|summer))\b", re.I)
PHONE_INTENT_RE = re.compile(r"\b(phone number|telephone|number to (call|reach)|contact number|fax|"
                             r"(who|what number) (do|should|can) i call|call (dupaco|you|the|for))\b", re.I)
_PHONE_WORDS = {stem(w) for w in ("phone", "number", "call", "telephone", "contact", "reach")}
# words that may follow the counted noun without turning it into a modifier ("22 branch office locations")
_COUNT_CONT = set("""office offices location locations site sites nationwide across in throughout and or
with from at of for on as around worldwide statewide countywide""".split())
_DATE_RE = re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|"
                      r"December)\.?\s+\d{1,2}(,\s*20\d\d)?|\b\d{1,2}/\d{1,2}/\d{2,4}\b")


def _is_question(s: str) -> bool:
    return bool(re.search(r"\?\s*(\(https?://[^)]*\))?\s*$", s))


_ABBR_END = re.compile(r"\b(Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept?|Oct|Nov|Dec|St|Ave|Rd|Dr|Blvd|Ln|Hwy|Mr|Mrs|Ms|No|"
                       r"Inc|Co|Corp|Jr|Sr|vs|etc|U\.S|a\.m|p\.m|[A-Z])\.$")
PHONE_RE = re.compile(r"\b(?:1[.\-\s])?\(?\d{3}\)?[.\-\s]\d{3}[.\-]\d{4}\b")


def split_sentences(text: str) -> list[str]:
    parts = [s for s in _SENT_RE.split(text) if s and s.strip(" -")]
    out: list[str] = []
    for p in parts:
        # re-join "Oct." / "St." splits, and keep a bare "Yes!"/"No." with the sentence that explains it
        if out and (_ABBR_END.search(out[-1]) or len(out[-1]) < 12 and not out[-1].endswith(":")):
            out[-1] = out[-1] + " " + p.strip()
        else:
            out.append(p.strip(" -"))
    return out


def _best_window(text: str, qset: set, q: str, idf: dict, limit: int = 480) -> tuple[str, float]:
    """The 1-3 consecutive sentences that cover the question best (FAQ answers follow their question)."""
    sents = split_sentences(text)
    if not sents:
        return text[:limit], 0.0
    toks = [set(expand(content_tokens(s))) for s in sents]
    when = bool(re.search(r"\bwhen\b|what date|still open|\bclos(e|ing|ed)\b|what happened", q, re.I))
    count = bool(re.search(r"\b(how many|what time|number of|how long|what year)\b", q, re.I))
    num = bool(re.search(r"\b(how much|cost|fee|price|amount|up to|what is|what's|what are)\b", q, re.I))
    subject = "dupaco" in q.lower()
    phone = bool(PHONE_INTENT_RE.search(q))
    how = bool(re.search(r"\bhow (can|do|should|would) i\b|\bwhere (can|do) i\b", q, re.I))
    best, best_sc = (0, 1), -1e9
    for i in range(len(sents)):
        for size in (1, 2, 3):
            if i + size > len(sents):
                break
            seg = sents[i:i + size]
            union = set().union(*toks[i:i + size])
            sc = sum(idf.get(w, 1.0) for w in qset & union)
            joined = " ".join(seg)
            if when and _DATE_RE.search(joined):
                sc += 4.0
            elif when and re.search(r"\b(19|20)\d\d\b", joined):
                sc += 2.5
            if phone and PHONE_RE.search(joined):
                sc += 5.0
            if len(re.sub(r"\(https?://[^)]*\)|https?://\S+", "", joined).strip()) < 30:
                sc -= 4.0      # a label or button, not an answer
            if re.match(r"(I'm|I am|I'd|I want|I have|I would)\b", seg[0]):
                sc -= 1.5      # a form option or testimonial
            if joined.rstrip().endswith(":") and i + size < len(sents):
                sc -= 2.0      # the answer is in what follows
            if count and re.search(r"\d", joined) or num and re.search(r"\$\d|\d%|\d,\d{3}", joined):
                sc += 2.0
            if subject and "dupaco" in joined.lower():
                sc += 0.5
            if how and re.search(r"\d{3}-\d{3}-\d{4}|https?://", joined):
                sc += 2.5      # something the member can act on (phone number, link)
            if _is_question(seg[-1]):
                sc -= 3.0      # a FAQ question without its answer
            if seg[0][:1] in "\"'":
                sc -= 1.0      # quotes from people rarely state the facts
            if re.fullmatch(r"[^.]{0,60}\(https?://[^)]*\)", joined):
                sc -= 2.0      # a bare button/link
            sc -= 0.004 * len(joined) + 0.8 * (size - 1) + 0.12 * i
            if sc > best_sc:
                best, best_sc = (i, size), sc
    i, size = best
    out = " ".join(sents[i:i + size])
    if len(out) > limit:
        out = out[:limit].rsplit(" ", 1)[0] + "..."
    return out, best_sc


def build() -> KnowledgeSystem:
    return KnowledgeSystem()

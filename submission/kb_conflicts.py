"""
Contradiction finder.

Three kinds of contradiction are reported, one entry per fact:
  1. a rate on the official rate sheet that changed between snapshots (older value = stale);
  2. a secondary page (landing page, calculator, PDF, product page) in the newest snapshot that still shows
     a different value than the official rate sheet for the same product/tier/term;
  3. text facts restated across press releases with different values (branch count, Forbes recognition),
     where the most recently published statement wins.
Rows are keyed by a canonical product name plus their qualifiers (term, balance tier, model-year rank,
jumbo/special/Roth/Traditional...), so different tiers are never reported as conflicts.
"""
from __future__ import annotations

import re
from collections import OrderedDict

from contract import Conflict, Source
from kb_parse import url_path
from kb_rates import RowFeat, TableFeat, short_col

_PRODUCTS = [  # (regex over section/title/header/first label, canonical name) - first match wins
    (r"achieve", None),                       # promo money market that never appears on the rate sheet
    (r"health savings|\bhsa\b", "Health Savings Account"),
    (r"high-yield|high yield", "High-Yield Savings"),
    (r"investor|money market", "Investor's Choice Money Market"),
    (r"bright", "Bright Savings"),
    (r"you[- ]name[- ]it", "You Name It Savings"),
    (r"holiday", "Holiday Club Savings"),
    (r"launchpad", "Launchpad Savings"),
    (r"share savings|savings account rates", "Share Savings"),
    (r"1st rate|first rate", "1st Rate Checking"),
    (r"free checking", "Free Checking"),
    (r"auto|motorcycle", "Auto & Motorcycle Loan"),
    (r"business platinum", "Business Platinum Visa"),
    (r"business rewards", "Business Rewards Visa"),
    (r"platinum", "Platinum Visa"),
    (r"rewards visa|rewards credit", "Rewards Visa"),
    (r"heloc|line of credit", "HELOC"),
    (r"home improvement", "Home Improvement Loan"),
    (r"personal loan", "Personal Loan"),
    (r"boat|\brv\b", "Boat & RV Loan"),
    (r"atv|snowmobile|jet ski", "ATV, Snowmobile & Jet Ski Loan"),
]


def product_of(tf: TableFeat) -> str | None:
    t = tf.table
    first = t.rows[0].label_text if t.rows else ""
    hay = " ".join([t.section, tf.name, " ".join(t.header), first]).lower()
    labels = " ".join(r.label_text for r in t.rows).lower()
    biz = "business" in hay or "/business" in t.url
    if "ira" in tf.flags:
        kind = "Special IRA" if "special" in labels else "Variable IRA" if "$" in labels else "Regular IRA"
        return ("Business " if biz else "") + kind
    if re.search(r"certificate|term-share|\bspecial\b|^regular\b", hay) and not re.search(r"loan|visa", hay):
        kind = "Special Certificate" if "special" in labels else "Regular Certificate"
        return ("Business " if biz else "") + kind
    if re.search(r"purchase:|refinance:|conventional", labels):
        if re.search(r"equity|cash out", hay):
            return "Home Equity Refinance"
        return "Mortgage Refinance" if "refinance:" in labels and "purchase:" not in labels else "Mortgage Purchase"
    if re.search(r"home equity loan", hay) or re.search(r"\(\d+ mo\)|^\d+ years$", labels):
        return "Home Equity Loan"
    for rx, name in _PRODUCTS:
        if re.search(rx, hay):
            if name is None:
                return None
            if biz and not name.startswith("Business"):
                return "Business " + name
            return name
    return None


def _norm_val(v: str) -> str:
    m = re.match(r"\s*(\d+(?:\.\d+)?)\s*%\s*-\s*(\d+(?:\.\d+)?)\s*%", v)
    if m:
        return f"{m.group(1)}% - {m.group(2)}%"
    m = re.search(r"\d+(?:\.\d+)?\s*%", v)
    return re.sub(r"\s+", "", m.group(0)) if m else v.strip()


def _row_key(prod: str, tf: TableFeat, r: RowFeat) -> tuple:
    year_rank = None
    if r.years:
        ranges = sorted({x.years for x in tf.rows if x.years}, key=lambda y: -y[1])
        year_rank = ranges.index(r.years)
    amt = (round(r.amount[0]), round(r.amount[1]) if r.amount[1] != float("inf") else -1) if r.amount else None
    flags = tuple(sorted(r.flags & {"jumbo", "special", "intro", "relief", "roth", "traditional"}))
    words = None
    if not (r.term or amt or year_rank is not None or flags):
        words = re.sub(r"[^a-z]", "", r.text.lower())[:30]
    return prod, r.term, amt, year_rank, flags, words


def _values(r: RowFeat) -> list[str]:
    """Rate-like values that identify the row (APY/APR first, mortgages also their interest rate)."""
    out = []
    for h, v in r.values:
        col = short_col(h)
        if col in ("APY", "APR", "rate") and "%" in v:
            out.append(_norm_val(v))
    if not out:
        out = [_norm_val(v) for h, v in r.values if "%" in v and short_col(h) not in ("points",)][:1]
    return out


def _label(prod: str, tf: TableFeat, r: RowFeat) -> str:
    lab = r.text or tf.name
    lab = re.sub(r"^(purchase|refinance):\s*", "", lab, flags=re.I)
    if prod.lower() in lab.lower():
        return lab
    return f"{prod} - {lab}" if lab else prod


def _official(tf: TableFeat) -> bool:
    return tf.table.kind == "rates" or (tf.table.kind == "product" and "/business/credit-cards" in tf.table.url)


def find_conflicts(snaps) -> list[Conflict]:
    if not snaps:
        return []
    latest = snaps[-1]
    facts: "OrderedDict[tuple, dict]" = OrderedDict()

    # 1. official values per snapshot
    for si, snap in enumerate(snaps):
        for tf in snap.rates.tables:
            if not _official(tf):
                continue
            prod = product_of(tf)
            if not prod:
                continue
            for r in tf.rows:
                vals = _values(r)
                if not vals:
                    continue
                key = _row_key(prod, tf, r)
                f = facts.setdefault(key, {"hist": [], "label": _label(prod, tf, r), "stale": [], "prod": prod})
                if f["hist"] and f["hist"][-1][0] == si:
                    continue      # same key twice on one snapshot (e.g. mirrored table): keep the first
                f["hist"].append((si, vals, tf.table.url, tf.table.effective_raw))
                f["label"] = _label(prod, tf, r)

    # 2. secondary pages in the newest snapshot that disagree with the official sheet
    n_last = len(snaps) - 1
    for tf in latest.rates.tables:
        if _official(tf):
            continue
        prod = product_of(tf)
        if not prod:
            continue
        for r in tf.rows:
            vals = _values(r)
            key = _row_key(prod, tf, r)
            f = facts.get(key)
            if not vals or not f or not f["hist"] or f["hist"][-1][0] != n_last:
                continue
            cur = f["hist"][-1][1]
            if vals[0] not in cur:
                f["stale"].append((vals, tf.table.url))

    out: list[Conflict] = []
    seen_values: set = set()
    # personal products first, so a business or IRA mirror of the same change is the one dropped
    order = sorted(facts.items(), key=lambda kv: (kv[0][0].startswith("Business"), "IRA" in kv[0][0]))
    for key, f in order:
        hist = f["hist"]
        if not hist or hist[-1][0] != n_last:
            continue        # product no longer on the sheet
        cur_vals = hist[-1][1]
        primary_i = 0
        changed = [h for h in hist if h[1][primary_i] != cur_vals[primary_i]]
        if not changed and not f["stale"]:
            continue
        values: list[str] = []
        for h in hist:
            for v in h[1]:
                if v not in values:
                    values.append(v)
        for vals, _ in f["stale"]:
            for v in vals:
                if v not in values:
                    values.append(v)
        # drop current non-primary values that never changed (they are not part of the dispute)
        sig = tuple(values)
        if sig in seen_values:
            continue        # same dispute already reported for a mirrored product (IRA/business copy)
        seen_values.add(sig)
        reasons = []
        srcs = []
        if changed:
            first_old = changed[-1]
            reasons.append(f"the official rate sheet changed from {first_old[1][0]} (snapshot "
                           f"{snaps[first_old[0]].date}) to {cur_vals[0]} (snapshot {snaps[n_last].date}"
                           + (f", rates effective {hist[-1][3]}" if hist[-1][3] else "") + ")")
            srcs.append(Source(url=first_old[2], snapshot_id=snaps[first_old[0]].id,
                               quote=f"{f['label']}: {first_old[1][0]}"))
        for vals, url in f["stale"][:3]:
            reasons.append(f"{url_path(url)} still shows {vals[0]}, but it is not the official rate sheet")
            srcs.append(Source(url=url, snapshot_id=latest.id, quote=f"{f['label']}: {vals[0]}"))
        srcs.append(Source(url=hist[-1][2], snapshot_id=latest.id, quote=f"{f['label']}: {cur_vals[0]}"))
        reason = "; ".join(reasons) + f". The newest official rate sheet ({url_path(hist[-1][2])}) is authoritative."
        out.append(Conflict(topic=f"{f['label']} ({'APY' if f['prod'] in _DEPOSITS else 'APR/rate'})",
                            values=values, sources=srcs, resolution=" / ".join(cur_vals), reason=reason))

    out.extend(_ira_mismatches(latest))
    out.extend(_text_conflicts(latest))
    out.extend(_sentence_rate_conflicts(snaps, {k[0] for k in facts}))
    return out


_NOUNS = ("certificate", "savings", "checking", "money market", "ira", "auto", "mortgage", "heloc", "personal loan",
          "credit card", "visa", "boat", "rv", "home equity")


def _sentence_rate_conflicts(snaps, table_products: set) -> list[Conflict]:
    """Rates written as sentences ("6-month certificate APY: 3.50%") rather than tables. Only used for
    products the rate tables don't cover, so table-based facts are never double-reported."""
    covered = " ".join(table_products).lower()
    rx = re.compile(r"(\d{1,3})[\s-]*(month|year)s?\b[^.%\n]{0,60}?\b(apy|apr|rate)\b[^0-9\n]{0,20}"
                    r"(\d{1,2}(?:\.\d+)?)\s*%", re.I)
    seen: "OrderedDict[tuple, list]" = OrderedDict()
    for si, snap in enumerate(snaps):
        for url, p in snap.pages.items():
            if p.kind not in ("product", "rates", "landing") or p.carried_from:
                continue
            for ln in p.text.split("\n"):
                low = ln.lower()
                noun = next((n for n in _NOUNS if n in low), None)
                if not noun or noun.split()[0][:5] in covered:
                    continue
                for m in rx.finditer(ln):
                    months = int(m.group(1)) * (12 if m.group(2).lower() == "year" else 1)
                    key = (noun, months, m.group(3).upper())
                    expired = bool(re.search(r"ended|expired|no longer", low))
                    seen.setdefault(key, []).append((si, f"{m.group(4)}%", url, expired, ln.strip()[:160]))
    out = []
    last = len(snaps) - 1
    for (noun, months, metric), obs in seen.items():
        vals = []
        for o in obs:
            if o[1] not in vals:
                vals.append(o[1])
        if len(vals) < 2:
            continue
        live = [o for o in obs if o[0] == last and not o[3]] or [o for o in obs if not o[3]] or obs
        res = live[-1]
        out.append(Conflict(
            topic=f"{months}-month {noun} {metric}", values=vals,
            sources=[Source(url=o[2], snapshot_id=snaps[o[0]].id, quote=o[4]) for o in obs[-4:]],
            resolution=res[1],
            reason=f"{res[1]} is on the newest snapshot's current page; other values are older or from an "
                   f"offer marked as ended."))
    return out


_DEPOSITS = {"Health Savings Account", "High-Yield Savings", "Investor's Choice Money Market", "Bright Savings",
             "You Name It Savings", "Holiday Club Savings", "Launchpad Savings", "Share Savings",
             "1st Rate Checking", "Free Checking", "Special Certificate", "Regular Certificate", "Special IRA",
             "Regular IRA", "Variable IRA"}


def _ira_mismatches(snap) -> list[Conflict]:
    """Roth and Traditional versions of the same IRA certificate must pay the same; flag rows that don't."""
    out = []
    certs = {}
    for tf in snap.rates.tables:
        if tf.table.kind == "rates" and product_of(tf) in ("Special Certificate", "Regular Certificate"):
            for r in tf.rows:
                certs[(r.term, "jumbo" in r.flags)] = _values(r)[0] if _values(r) else None
    for tf in snap.rates.tables:
        if tf.table.kind != "rates" or "ira" not in tf.flags:
            continue
        pairs = {}
        for r in tf.rows:
            k = (r.term, "jumbo" in r.flags, r.amount)
            pairs.setdefault(k, {})["roth" if "roth" in r.flags else "traditional"] = r
        for (term, jumbo, amt), d in pairs.items():
            if len(d) < 2:
                continue
            a, b = _values(d["roth"]), _values(d["traditional"])
            if not a or not b or a[0] == b[0]:
                continue
            ref = certs.get((term, jumbo))
            res = ref if ref in (a[0], b[0]) else a[0]
            out.append(Conflict(
                topic=f"{term}-month {'jumbo ' if jumbo else ''}special IRA certificate (Roth vs Traditional APY)",
                values=[a[0], b[0]],
                sources=[Source(url=tf.table.url, snapshot_id=snap.id, quote=f"{d['roth'].text}: {a[0]}"),
                         Source(url=tf.table.url, snapshot_id=snap.id, quote=f"{d['traditional'].text}: {b[0]}")],
                resolution=res,
                reason="The Roth and Traditional versions of the same IRA certificate are listed with different "
                       f"rates on the same rate sheet; {res} matches the regular certificate rate for that term."))
    return out


_ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8}


def _text_conflicts(snap) -> list[Conflict]:
    out = []
    # branch count restated in press-release boilerplate
    counts = {}
    for url, p in snap.pages.items():
        if not p.date:
            continue
        for m in re.finditer(r"\b(\d{1,2}) branch(?: office)? locations\b", p.text):
            n = int(m.group(1))
            if n not in counts or counts[n][0] < p.date:
                counts[n] = (p.date, p.url)
    if len(counts) >= 2:
        newest = max(counts.items(), key=lambda kv: kv[1][0])
        vals = [f"{n} branch office locations" for n, _ in sorted(counts.items(), key=lambda kv: kv[1][0])]
        out.append(Conflict(
            topic="Number of Dupaco branch locations (branches)", values=vals,
            sources=[Source(url=u, snapshot_id=snap.id, quote=f"{n} branch office locations")
                     for n, (d, u) in sorted(counts.items(), key=lambda kv: kv[1][0])][-4:],
            resolution=f"{newest[0]} branch office locations",
            reason=f"Press releases from different years state different branch counts; the most recent one "
                   f"({newest[1][0]}) says {newest[0]}."))
    # Forbes recognition count
    seen = {}
    for url, p in snap.pages.items():
        if not p.date:
            continue
        for m in re.finditer(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth) time Forbes "
                             r"(?:has )?(?:recognized|named|ranked) Dupaco", p.text):
            w = m.group(1)
            if w not in seen or seen[w][0] < p.date:
                seen[w] = (p.date, p.url)
    if len(seen) >= 2:
        order = sorted(seen.items(), key=lambda kv: kv[1][0])
        best = order[-1][0]
        out.append(Conflict(
            topic="Forbes recognition of Dupaco (how many times)", values=[f"{w} time" for w, _ in order],
            sources=[Source(url=u, snapshot_id=snap.id, quote=f"{w} time Forbes recognized Dupaco")
                     for w, (d, u) in order],
            resolution=f"{best} time",
            reason=f"Older releases count fewer Forbes awards; the newest ({order[-1][1][0]}) says it was the "
                   f"{best} time."))
    return out

"""
Tokenisation, synonyms and a small BM25 index (standard library only).
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

STOP = set("""
a an the and or of to in on for at by with from as is are was were be been being it its this that these those
i me my we our you your he she they them their what which who whom whose when where why how do does did doing
can could would should will shall may might must have has had having not no yes if then than so such there here
about into over under up down out off again further once all any both each few more most other some same own
just very too also only please tell give get got know like want need let let's im i'm id i'd ive i've dupaco dupacos
much many current currently now today right offer offers offering provide available still s t
receive received receiving
""".split())

# question words that carry no product meaning for rate lookups
GENERIC = set("""
rate rates interest apr apy yield dividend annual percentage percent account accounts loan loans current
term terms month months year years mo yr option options price pricing cost charge charges pay earn earns
fixed standard regular best good great much what latest
""".split())

SYNONYMS = {
    "cd": ["certificate"], "cds": ["certificate"], "certificates": ["certificate"],
    "car": ["auto"], "cars": ["auto"], "vehicle": ["auto"], "vehicles": ["auto"], "truck": ["auto"],
    "suv": ["auto"], "automobile": ["auto"], "motorcycle": ["motorcycle", "auto"],
    "mortgage": ["home", "mortgage"], "mortgages": ["home", "mortgage"], "house": ["home"],
    "heloc": ["heloc", "equity", "line"], "camper": ["rv"], "motorhome": ["rv"],
    "jetski": ["jet", "ski"], "snowmobiles": ["snowmobile"],
    "hsa": ["health", "saving"], "first": ["1st", "first"], "roth": ["roth", "ira"], "traditional": ["traditional"],
    "ira": ["ira"], "iras": ["ira"], "retirement": ["ira", "retirement"],
    "creditcard": ["credit", "card"], "visa": ["visa", "card"], "cards": ["card"],
    "remodel": ["improvement"], "renovation": ["improvement"],
    "refi": ["refinance"], "refinancing": ["refinance"], "refinanced": ["refinance"],
    "savings": ["saving"], "jumbo": ["jumbo"], "promo": ["special"], "promotional": ["special"],
    "share": ["share"], "kids": ["youth"], "child": ["youth"], "teen": ["youth"],
}

NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
             "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
             "sixteen": 16, "seventeen": 17, "eighteen": 18, "twenty": 20, "twenty-four": 24, "thirty": 30,
             "thirty-six": 36, "forty-eight": 48, "sixty": 60, "seventy-two": 72, "eighty-four": 84}

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


_SUFFIXES = [("ibility", "ib"), ("ational", "ate"), ("ization", "ize"), ("ations", ""), ("itions", ""),
             ("ation", ""), ("ition", ""), ("izing", ""), ("ized", ""), ("izes", ""), ("ize", ""), ("ments", ""),
             ("ment", ""), ("ness", ""), ("ings", ""), ("ing", ""), ("ible", "ib"), ("ions", ""), ("ion", ""),
             ("ies", "y"), ("ied", "y"), ("ees", "ee"), ("ers", ""), ("ors", ""), ("er", ""), ("or", ""),
             ("ure", ""), ("ed", ""), ("es", ""), ("ly", ""), ("s", ""), ("e", "")]
_NO_STEM = {"business", "jumbo", "visa", "rewards", "apr", "apy", "ira", "roth", "heloc", "atv", "rv", "hours",
            "savings", "checking", "member", "members", "rate", "rates", "dupaco", "shine", "service", "services",
            "foundation", "routing", "interest"}
_FIXED = {"savings": "saving", "rewards": "reward", "hours": "hour", "members": "member", "rates": "rate",
          "services": "service", "checking": "checking"}


def stem(w: str) -> str:
    if w in _FIXED:
        return _FIXED[w]
    if len(w) <= 3 or w.isdigit() or w in _NO_STEM or not w.isalpha():
        return w
    for suf, rep in _SUFFIXES:
        if w.endswith(suf):
            base = w[: -len(suf)] + rep
            min_len = 3 if suf in ("s", "es", "ed", "e") else 4
            if len(base) >= min_len and not (suf == "s" and w.endswith(("ss", "us", "is"))):
                return base
    return w


def tokens(text: str) -> list[str]:
    t = re.sub(r"(?<=[a-z])\$(?=[a-z])", "s", text.lower().replace("&", " and ").replace("'s", ""))
    return [stem(w) for w in _TOKEN_RE.findall(t)]


def content_tokens(text: str) -> list[str]:
    return [w for w in tokens(text) if w not in STOP and len(w) > 1 or w.isdigit()]


def expand(toks: list[str]) -> list[str]:
    out = []
    for w in toks:
        out.append(w)
        for s in _SYN.get(w, []):
            if s != w:
                out.append(s)
    return out


# keep every word list in stemmed form so lookups line up with tokens()
STOP = STOP | {stem(w) for w in STOP}
GENERIC = GENERIC | {stem(w) for w in GENERIC}
SYNONYMS.update({"locator": ["location", "find"], "atms": ["atm"], "eligible": ["eligible", "join"],
                 "join": ["join", "member"], "wallet": ["wallet", "pay"], "phone": ["phone", "call"],
                 "telephone": ["phone", "call"], "close": ["close", "closure"], "closing": ["close", "closure"],
                 "update": ["update", "change"], "change": ["change", "update"], "fee": ["fee", "charge"],
                 "cost": ["cost", "fee"], "replace": ["replace", "replacement"], "lost": ["lost", "stolen"]})
_SYN = {stem(k): [stem(x) for x in v] for k, v in SYNONYMS.items()}


class BM25:
    """Plain BM25 over a list of token lists."""

    def __init__(self, docs: list[list[str]], k1: float = 1.3, b: float = 0.6):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) or 1 for d in docs]
        self.avg = (sum(self.len) / len(self.len)) if docs else 1.0
        df = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        self.post = defaultdict(list)
        for i, c in enumerate(self.tf):
            for w in c:
                self.post[w].append(i)

    def scores(self, q: list[str]) -> dict[int, float]:
        out: dict[int, float] = defaultdict(float)
        for w in set(q):
            idf = self.idf.get(w)
            if not idf:
                continue
            for i in self.post[w]:
                f = self.tf[i][w]
                out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return out

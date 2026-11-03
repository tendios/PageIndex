"""The numbering a document writes its sections with, read from a line: which family (decimal «2.3», letter «B.1»,
roman «IV», ordinal «Vigésima primera» / «Vint-i-unena», a keyword and its number «Cláusula 12», «Annex III») and
the position it names (a bare «14 GARANTÍAS», with no separator, only where ``bare`` allows it: a summary
table's row or a flagged heading), so the sequence of a document can be followed: «2.3» after «2.2», «2.3.1» under «2.3»,
«Tretzena» after «Dotzena»."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional


def _plain(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


# ── ordinals in Spanish and Catalan, feminine and masculine, as clauses are named («Primera. Objeto») ──
_UNITS_ES = {"primer": 1, "segon": 2, "primera": 1, "primero": 1, "segunda": 2, "segundo": 2, "tercera": 3, "tercero": 3,
             "tercer": 3, "cuarta": 4, "cuarto": 4, "quinta": 5, "quinto": 5, "sexta": 6, "sexto": 6,
             "septima": 7, "septimo": 7, "setima": 7, "octava": 8, "octavo": 8, "novena": 9, "noveno": 9}
_TENS_ES = {"decima": 10, "decimo": 10, "vigesima": 20, "vigesimo": 20, "trigesima": 30, "trigesimo": 30,
            "cuadragesima": 40, "cuadragesimo": 40, "quincuagesima": 50, "quincuagesimo": 50}
_TEENS_ES = {"undecima": 11, "undecimo": 11, "duodecima": 12, "duodecimo": 12}
_CA = {"primera": 1, "primer": 1, "segona": 2, "segon": 2, "tercera": 3, "tercer": 3, "quarta": 4, "quart": 4,
       "cinquena": 5, "cinque": 5, "sisena": 6, "sise": 6, "setena": 7, "sete": 7, "vuitena": 8, "vuite": 8,
       "novena": 9, "nove": 9, "desena": 10, "dese": 10, "onzena": 11, "onze": 11, "dotzena": 12, "dotze": 12,
       "tretzena": 13, "tretze": 13, "catorzena": 14, "catorze": 14, "quinzena": 15, "quinze": 15,
       "setzena": 16, "setze": 16, "dissetena": 17, "disset": 17, "divuitena": 18, "divuite": 18,
       "dinovena": 19, "dinove": 19, "vintena": 20, "vinte": 20, "trentena": 30, "trente": 30,
       "quarantena": 40, "quarante": 40, "cinquantena": 50}
_CA_TENS = {"vint": 20, "trenta": 30, "quaranta": 40, "cinquanta": 50}
_CA_UNIT_SUFFIX = {"unena": 1, "une": 1, "dosena": 2, "dose": 2, "tresena": 3, "trese": 3, "quatrena": 4,
                   "quatre": 4, "cinquena": 5, "sisena": 6, "setena": 7, "vuitena": 8, "novena": 9}

_ORDINAL_HEAD = re.compile(r"^(?P<word>[a-zà-ÿ]+(?:(?:\s+|-)[a-zà-ÿ]+){0,3}?)\s*(?:[.:–)]|-(?![a-zà-ÿ]))\s*(?=\S)", re.I)


def ordinal_value(words: str) -> Optional[int]:
    """«Vigésima primera» → 21, «décimo tercera» → 13, «Vint-i-setena» → 27, «Onzena» → 11; None otherwise."""
    w = _plain(words).replace("-i-", " ").replace("-", " ").split()
    if not w or len(w) > 3:
        return None
    if len(w) == 1:
        word = w[0]
        for table in (_UNITS_ES, _TEENS_ES, _TENS_ES, _CA):
            if word in table:
                return table[word]
        for tens, value in (("decimo", 10), ("decima", 10), ("vigesimo", 20), ("vigesima", 20),
                            ("trigesimo", 30), ("trigesima", 30)):  # «decimotercera», «vigesimoprimera»
            if word.startswith(tens) and word[len(tens):] in _UNITS_ES:
                return value + _UNITS_ES[word[len(tens):]]
        return None
    if len(w) == 2:
        tens = _TENS_ES.get(w[0])
        if tens and w[1] in _UNITS_ES:
            return tens + _UNITS_ES[w[1]]
        tens = _CA_TENS.get(w[0])
        if tens and w[1] in _CA_UNIT_SUFFIX:
            return tens + _CA_UNIT_SUFFIX[w[1]]
    return None


ROMAN = re.compile(r"^(?=[IVXLC])(C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")


def roman_value(text: str) -> Optional[int]:
    if not text or not ROMAN.match(text):
        return None
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
    total, prev = 0, 0
    for ch in reversed(text):
        v = values[ch]
        total = total - v if v < prev else total + v
        prev = max(prev, v)
    return total


KEYWORDS = {
    "chapter": ("capitulo", "capitol", "titulo", "titol", "parte", "part", "seccion", "seccio", "libro", "llibre"),
    "clause": ("clausula", "article", "articulo", "apartado", "apartat", "disposicion", "disposicio"),
    "annex": ("anexo", "annex", "annexo", "apendice", "apendix"),
}
_KEYWORD = re.compile(
    r"^(?P<word>cap[ií]tulo|cap[ií]tol|t[ií]tulo|t[ií]tol|parte|part|secci[oó]n?|llibre|libro|cl[aàá]usula|"
    r"article|art[ií]culo|apartado|apartat|disposici[oó]n?|annex[oa]?|anexo|ap[eèé]ndi(?:ce|x))\b\.?\s*"
    r"(?P<num>[0-9]{1,3}(?:\.[0-9]{1,2}){0,2}|[IVXLC]{1,6}(?:\.[0-9]{1,2})?|[A-Z](?![a-z])|"
    r"(?:primer[ao]?|segund[ao]|segona|tercer[ao]?|cuart[ao]|quarta|quint[ao]|cinquena|sext[ao]|sisena|s[eé]ptim[ao]|"
    r"setena|octav[ao]|vuitena|noven[ao]|d[eé]cim[ao]|desena|[uú]nic[ao]?)(?=[\s.:,–-]))?(?P<rest>.*)$", re.I)
_DECIMAL = re.compile(r"^(?P<num>\d{1,2}(?:\.\d{1,2}){0,5})(?:\.?(?P<letter>[A-Z])(?![A-Za-z]))?(?P<sep>\s*[.\-–)º°]*)\s*(?P<rest>.*)$")
_LETTER = re.compile(r"^(?P<num>[A-Z](?:\.?\d{1,2}){0,2})(?P<sep>\s*[.\-–)]+|(?<=\d)\s+(?=[A-ZÀ-Ú]{2}))\s*(?P<rest>.*)$")
_ROMAN = re.compile(r"^(?P<num>[IVXLC]{1,6})(?P<sep>\s*[.\-–)]+)\s*(?P<rest>.*)$")


# Ranks of the keywords that group a document, from the widest: «Libro» over «Título» over «Capítulo» over «Sección».
KEYWORD_RANK = (("libro", "llibre"), ("titulo", "titol", "parte", "part"), ("capitulo", "capitol"), ("seccion", "seccio"))


def keyword_rank(raw: str) -> int:
    word = _plain(raw.split()[0]) if raw else ""
    return next((k for k, words in enumerate(KEYWORD_RANK) if any(word.startswith(w) for w in words)), 2)


@dataclass(frozen=True)
class Marker:
    family: str                  # decimal | letter | roman | ordinal | chapter | clause | annex
    path: tuple                  # (2, 3) for «2.3»; (2,) for «B»; (2, 1) for «B.1»/«B1»; (12,) for «Cláusula 12»
    raw: str
    rest: str                    # the text after the marker: the title, when there is one

    @property
    def depth(self) -> int:
        return len(self.path)

    def parent(self) -> Optional[tuple]:
        return self.path[:-1] if len(self.path) > 1 else None


def _letter_value(ch: str) -> int:
    return ord(ch.upper()) - ord("A") + 1


def _dotted(num: str) -> tuple:
    return tuple(int(x) for x in num.split(".") if x)


def parse(text: str, bare: bool = False) -> Optional[Marker]:
    """The numbering a line opens with, or None. A bare number with no title text («2024», «9 años») is no marker
    unless a separator follows it; amounts («1.000,00») and dates are not either."""
    text = (text or "").strip()
    if not text:
        return None
    m = _KEYWORD.match(text)
    if m:
        word = _plain(m.group("word")).rstrip(".")
        family = next(f for f, words in KEYWORDS.items() if any(word.startswith(x) for x in words))
        num = m.group("num")
        if num:
            if num[0].isalpha() and len(num) > 2 and ordinal_value(num):  # «Sección primera», «Capítol segon»
                path = (ordinal_value(num),)
            elif num.lower() in ("único", "unico", "única", "unica", "únic", "unic"):
                path = (1,)
            elif num[0].isdigit():
                path = _dotted(num)
            elif roman_value(num.split(".")[0]) and (len(num) > 1 or num in "IVX"):
                head, *tail = num.split(".")
                path = (roman_value(head), *(int(t) for t in tail))
            elif len(num) == 1:
                path = (_letter_value(num),)
            else:
                return None
            rest = m.group("rest").strip(" .:-–")
            if rest[:1].islower() and family != "clause":
                return None  # «Anexo I de este pliego…», «Capítulo 3 de la Ley…»: a sentence naming it
            return Marker(family, path, m.group(0)[: len(m.group(0)) - len(m.group("rest"))].strip(), rest)
        if family in ("annex", "chapter") and not m.group("rest").strip()[:1].islower():
            return Marker(family, (), m.group("word"), m.group("rest").strip(" .:-–"))
        return None
    head = _ORDINAL_HEAD.match(text)
    if head:
        value = ordinal_value(head.group("word"))
        if value:
            return Marker("ordinal", (value,), head.group(0).strip(), text[head.end():].strip(" .:-–"))
    m = _DECIMAL.match(text)
    if m and not re.match(r"^\d{1,3}(?:\.\d{3})+,\d", text) and not re.match(r"^\d{1,2}[./]\d{1,2}[./]\d{2,4}", text):
        path = _dotted(m.group("num"))
        rest = m.group("rest")
        if path and path[0] <= 99 and (m.group("sep").strip() or len(path) > 1) and rest[:1].isalpha():
            if m.group("letter"):
                path = (*path, _letter_value(m.group("letter")))
            # «1)», «2)»: the numbering of a list more often than of a document's sections; its own family
            family = "paren" if m.group("sep").strip() == ")" and len(path) == 1 else "decimal"
            return Marker(family, path, text[: len(text) - len(rest)].strip(), rest.strip())
        if bare and path and path[0] <= 60 and not m.group("sep").strip() and len(path) == 1 \
                and re.match(r"^[A-ZÀ-Ú]{2}", rest):
            return Marker("decimal", path, m.group("num"), rest.strip())  # «14 GARANTÍAS», «3 Objeto» (summary rows)
        return None
    m = _ROMAN.match(text)
    if m and roman_value(m.group("num")) and m.group("rest")[:1].isalpha():
        return Marker("roman", (roman_value(m.group("num")),), m.group("num"), m.group("rest").strip())
    m = _LETTER.match(text)
    if m and m.group("rest")[:1].isalpha():
        num = m.group("num").replace(".", "")
        path = (_letter_value(num[0]), *(int(d) for d in re.findall(r"\d{1,2}", m.group("num"))))
        return Marker("letter", path, m.group("num"), m.group("rest").strip())
    return None


def follows(prev: tuple, cur: tuple) -> bool:
    """Whether ``cur`` continues a sequence whose last position was ``prev``: next sibling at any depth of
    ``prev`` («2.3» → «2.4», «2.3.1» → «3»), or its first child («2.3» → «2.3.1»)."""
    if not cur:
        return False
    if len(cur) == len(prev) + 1 and cur[:-1] == prev and cur[-1] == 1:
        return True
    for depth in range(min(len(prev), len(cur)), 0, -1):
        if len(cur) == depth and cur[:-1] == prev[: depth - 1] and cur[-1] == prev[depth - 1] + 1:
            return True
    return False


def first_of_sequence(path: tuple) -> bool:
    return bool(path) and all(p == 1 for p in path)

"""Query-time spelling candidates grounded in the selected corpus, not an ontology."""

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import zip_longest

LETTERS = "a-zA-Zа-яА-ЯёЁ"
BOUNDARY = LETTERS + "0-9"
CODE = (
    rf"[0-9]+[{LETTERS}0-9]*"
    rf"(?:[-_/][{LETTERS}0-9]+|\.[0-9]+[{LETTERS}0-9]*)*"
    rf"(?:\([{LETTERS}0-9]+\)[0-9]*)*"
)
DESIGNATION = re.compile(
    rf"(?<![{BOUNDARY}])(?:[{LETTERS}]+(?:/[{LETTERS}]+)*[ ._\t-]*{CODE}"
    rf"|[0-9]+[-_]?[{LETTERS}]+[ ._\t-]*{CODE})(?![{BOUNDARY}])"
)
OBJECT = re.compile(
    rf"(?<![{BOUNDARY}])(?:obj(?:ect|ekt)?|объект(?:а|у|ом|е)?|工程|项目)"
    rf"[ ._\t-]*(?:№|#)?\s*(?P<before>{CODE})"
    rf"|(?<![{BOUNDARY}])(?P<after>{CODE})\s*(?:号)?(?:工程|项目)",
    re.IGNORECASE,
)
CYRILLIC = dict(
    zip(
        "абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
        "a b v g d e yo zh z i y k l m n o p r s t u f kh ts ch sh shch ' y ' e yu ya".split(),  # noqa: SIM905
    )
)
DEFAULT_ALIASES = (
    "拉格=LaGG",
    "拉=La",
    "米格=MiG",
    "雅克=Yak",
    "伊尔=Il",
    "LaG-3=LaGG-3",
)


def prepared(text):
    text = re.sub("[‐‑‒–—−]", "-", unicodedata.normalize("NFKC", text))
    return "".join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in text)


def folded(text):
    return "".join(CYRILLIC.get(c, c) for c in prepared(text).casefold())


def signature(text):
    """Numeric runs stay separate: T-34-85 never becomes T-3485."""
    return tuple(re.findall(r"[a-z]+|[0-9]+|[/()]", folded(text)))


def spelling_key(text):
    value = prepared(text).replace("_", "-")
    return re.sub(rf"^([{LETTERS}]+(?:/[{LETTERS}]+)*)[ .\t-]+(?=[0-9])", r"\1", value)


def strict_key(text):
    return spelling_key(folded(text))


@dataclass(frozen=True)
class Reference:
    text: str
    key: str
    parts: tuple


def references(text):
    text = prepared(text)
    occupied = []
    found = []
    for match in OBJECT.finditer(text):
        code = match["before"] or match["after"]
        occupied.append(match.span())
        found.append(
            (
                match.start(),
                Reference(
                    match[0], "object:" + folded(code), ("object", *signature(code))
                ),
            )
        )
    for match in DESIGNATION.finditer(text):
        if any(match.start() < end and match.end() > start for start, end in occupied):
            continue
        found.append(
            (
                match.start(),
                Reference(match[0], strict_key(match[0]), signature(match[0])),
            )
        )
    yield from (ref for _, ref in sorted(found, key=lambda item: item[0]))


def query_references(question, aliases=DEFAULT_ALIASES):
    lookup = prepared(question)
    for rule in aliases:
        left, separator, right = rule.partition("=")
        left, right = left.strip(), right.strip()
        if separator and left and right:
            lookup = re.sub(
                rf"(?<![{BOUNDARY}]){re.escape(left)}(?=[ ._-]*[0-9]|(?![{BOUNDARY}]))",
                lambda _, replacement=right: replacement,
                lookup,
                flags=re.IGNORECASE,
            )
    prose_context = re.search(r"[A-Za-zА-Яа-я]", question) and not re.search(
        r"[\u4e00-\u9fff]", question
    )
    for ref in references(lookup):
        # Corpus membership alone cannot turn 'is 3 years' into an IS model.
        if (
            prose_context
            and lookup.strip() != ref.text
            and re.match(r"^(is|am|an|to|i|и|в|с)\s+[0-9]", ref.text)
        ):
            continue
        yield ref


@dataclass(frozen=True)
class Match:
    written: str
    candidates: tuple[str, ...]


@dataclass(frozen=True)
class Query:
    original: str
    matches: tuple[Match, ...]

    @property
    def labels(self):
        return tuple(
            dict.fromkeys(label for match in self.matches for label in match.candidates)
        )

    @property
    def text(self):
        additions = [label for label in self.labels if label not in self.original]
        return self.original + ("\n" + " ".join(additions) if additions else "")


class Catalogue:
    def __init__(self, texts=()):
        self.counts = Counter()
        self.exact = defaultdict(set)
        self.loose = defaultdict(set)
        for text in texts:
            for ref in references(text):
                self.counts[ref.text] += 1
                self.exact[ref.key].add(ref.text)
                self.loose[ref.parts].add(ref.text)

    @classmethod
    def merge(cls, catalogues):
        merged = cls()
        for catalogue in catalogues:
            merged.counts.update(catalogue.counts)
            for key, labels in catalogue.exact.items():
                merged.exact[key].update(labels)
            for key, labels in catalogue.loose.items():
                merged.loose[key].update(labels)
        return merged

    def expand(self, question, aliases=DEFAULT_ALIASES, max_aliases=8):
        pools = {}
        for ref in query_references(question, aliases):
            candidates = self.exact.get(ref.key) or self.loose.get(ref.parts, ())
            written_key = spelling_key(ref.text)
            explicit = {
                label for label in candidates if spelling_key(label) == written_key
            }
            ordered = sorted(explicit or candidates, key=lambda x: (-self.counts[x], x))
            if ordered:
                pools[ref.text] = ordered
        # Allocate one spelling per mention before additional alternatives.
        selections = {written: [] for written in pools}
        selected = set()
        for labels in zip_longest(*pools.values()):
            for written, label in zip(pools, labels):
                if label is not None and (
                    label in selected or len(selected) < max_aliases
                ):
                    selected.add(label)
                    selections[written].append(label)
            if len(selected) >= max_aliases:
                break
        return Query(
            question,
            tuple(
                Match(written, tuple(labels))
                for written, labels in selections.items()
                if labels
            ),
        )

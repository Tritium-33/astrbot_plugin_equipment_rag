"""Query-time spelling candidates grounded in the selected corpus, not an ontology."""

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import zip_longest

LETTERS = "a-zA-Zа-яА-ЯёЁ"
BOUNDARY = LETTERS + "0-9"
PREFIX = rf"[{LETTERS}]+(?:/[{LETTERS}]+)*"
QUALIFIED_HEAD = rf"(?:{PREFIX}[ ._\t-]*[0-9]|[0-9]+[-_]?[{LETTERS}]+[ ._\t-]*[0-9])"


def name_patterns(split_lists):
    guard = rf"(?!{QUALIFIED_HEAD})" if split_lists else ""
    code = (
        rf"[0-9]+[{LETTERS}0-9]*"
        rf"(?:[-_][{LETTERS}0-9]+|\.[0-9]+[{LETTERS}0-9]*"
        rf"|[ \t]*/[ \t]*{guard}(?:{PREFIX}[ ._\t-]+[0-9]+[{LETTERS}0-9]*|[{LETTERS}0-9]+)"
        rf"|\([{LETTERS}0-9]+\)[0-9]*)*"
    )
    designation = re.compile(
        rf"(?<![{BOUNDARY}])(?:{PREFIX}[ ._\t-]*{code}"
        rf"|[0-9]+[-_]?[{LETTERS}]+[ ._\t-]*{code})(?![{BOUNDARY}])"
    )
    obj = re.compile(
        rf"(?<![{BOUNDARY}])(?:obj(?:ect|ekt)?|объект(?:а|у|ом|е)?|工程|项目)"
        rf"[ ._\t-]*(?:№|#)?\s*(?P<before>{code})"
        rf"|(?<![{BOUNDARY}])(?P<after>{code})\s*(?:号)?(?:工程|项目)",
        re.IGNORECASE,
    )
    return designation, obj


NAME_PATTERNS = {split: name_patterns(split) for split in (False, True)}
BIBLIOGRAPHIC = re.compile(
    r"(?:pdf|tom|том|vol(?:ume)?|page|pp?)\.?[ \t]+[0-9]+(?:[-./][0-9]+)*",
    re.IGNORECASE,
)
PROSE_NUMBER = re.compile(r"(?:is|am|an|to|i|и|в|с)\s+[0-9]+")
QUANTITY_TAIL = re.compile(
    r"\s+(?:years?|months?|days?|hours?|minutes?|seconds?|году|года|лет)\b",
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
    """Retain explicit numeric segments before optional-separator fallback."""
    return tuple(re.findall(r"[a-z]+|[0-9]+|[/()]", strict_key(text)))


def spelling_key(text):
    value = re.sub(r"[ \t]*/[ \t]*", "/", prepared(text)).replace("_", "-")
    value = re.sub(r"/(?=[0-9])", "-", value)
    return re.sub(rf"^([{LETTERS}]+(?:/[{LETTERS}]+)*)[ .\t-]+(?=[0-9])", r"\1", value)


def strict_key(text):
    return spelling_key(folded(text))


def separator_positions(key):
    position = 0
    positions = set()
    for part in key.split("-")[:-1]:
        position += len(part)
        positions.add(position)
    return positions


@dataclass(frozen=True)
class Reference:
    text: str
    key: str
    parts: tuple


def references(text, *, split_lists=False):
    text = prepared(text)
    designation, obj = NAME_PATTERNS[split_lists]
    occupied = []
    found = []
    for match in obj.finditer(text):
        code = match["before"] or match["after"]
        occupied.append(match.span())
        found.append(
            (
                match.start(),
                match.end(),
                Reference(
                    match[0], "object:" + strict_key(code), ("object", *signature(code))
                ),
            )
        )
    for match in designation.finditer(text):
        if any(match.start() < end and match.end() > start for start, end in occupied):
            continue
        found.append(
            (
                match.start(),
                match.end(),
                Reference(match[0], strict_key(match[0]), signature(match[0])),
            )
        )
    previous_end = 0
    for start, end, ref in sorted(found, key=lambda item: item[0]):
        if BIBLIOGRAPHIC.fullmatch(ref.text):
            continue
        if PROSE_NUMBER.fullmatch(ref.text) and QUANTITY_TAIL.match(text[end:]):
            continue
        before = start - 1
        while before >= 0 and text[before].isspace():
            before -= 1
        # A slash list continues an already parsed name, not an arbitrary path.
        if (
            before >= 0
            and text[before] == "/"
            and text[previous_end:start].strip() != "/"
        ):
            continue
        yield ref
        previous_end = end


def query_references(question, aliases=DEFAULT_ALIASES):
    lookup = prepared(question)
    if isinstance(aliases, str):
        aliases = aliases.splitlines()
    elif not isinstance(aliases, (list, tuple)):
        aliases = DEFAULT_ALIASES
    for rule in aliases:
        if not isinstance(rule, str):
            continue
        left, separator, right = rule.partition("=")
        left, right = left.strip(), right.strip()
        if separator and left and right:
            lookup = re.sub(
                rf"(?<![{BOUNDARY}]){re.escape(left)}(?=[ ._-]*[0-9]|(?![{BOUNDARY}]))",
                lambda _, replacement=right: replacement,
                lookup,
                flags=re.IGNORECASE,
            )
    yield from references(lookup)


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
        self.compact = defaultdict(set)
        for text in texts:
            for ref in references(text):
                self.counts[ref.text] += 1
                self.exact[ref.key].add(ref.text)
                self.loose[ref.parts].add(ref.text)
                self.compact[ref.key.replace("-", "")].add(ref.key)

    @classmethod
    def merge(cls, catalogues):
        merged = cls()
        for catalogue in catalogues:
            merged.counts.update(catalogue.counts)
            for key, labels in catalogue.exact.items():
                merged.exact[key].update(labels)
            for key, labels in catalogue.loose.items():
                merged.loose[key].update(labels)
            for key, keys in catalogue.compact.items():
                merged.compact[key].update(keys)
        return merged

    def candidates(self, ref):
        candidates = self.exact.get(ref.key) or self.loose.get(ref.parts, ())
        if candidates:
            return candidates
        boundaries = separator_positions(ref.key)
        return {
            label
            for key in self.compact.get(ref.key.replace("-", ""), ())
            if boundaries <= separator_positions(key)
            for label in self.exact[key]
        }

    def resolve(self, ref):
        candidates = self.candidates(ref)
        if candidates:
            yield ref, candidates
            return
        # Prefer a complete source expression before splitting a slash list.
        parts = list(references(ref.text, split_lists=True))
        if len(parts) < 2:
            return
        start = 0
        while start < len(parts):
            for end in range(len(parts), start, -1):
                joined = "/".join(part.text for part in parts[start:end])
                combined = list(references(joined))
                candidates = self.candidates(combined[0]) if len(combined) == 1 else ()
                if candidates:
                    yield combined[0], candidates
                    start = end
                    break
            else:
                start += 1

    def expand(self, question, aliases=DEFAULT_ALIASES, max_aliases=8):
        pools = {}
        resolved = (
            item
            for ref in query_references(question, aliases)
            for item in self.resolve(ref)
        )
        for ref, candidates in resolved:
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

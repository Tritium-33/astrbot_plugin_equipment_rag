"""Query-time spelling candidates grounded in the selected corpus, not an ontology."""

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from functools import cache
from itertools import zip_longest

LETTERS = "a-zA-Zа-яА-ЯёЁ"
BOUNDARY = LETTERS + "0-9"
PREFIX = rf"[{LETTERS}]+(?:\s*/\s*[{LETTERS}]+)*"
CODE = (
    rf"[0-9]+[{LETTERS}0-9]*"
    rf"(?:[-_][{LETTERS}0-9]+|\.[0-9]+[{LETTERS}0-9]*"
    rf"|\s*/\s*(?:{PREFIX}[ ._\t-]+[0-9]+[{LETTERS}0-9]*|[{LETTERS}0-9]+)"
    rf"|\([{LETTERS}0-9]+\)[0-9]*)*"
)
DESIGNATION = re.compile(
    rf"(?<![{BOUNDARY}])(?:{PREFIX}[ ._\t-]*{CODE}"
    rf"|[0-9]+[-_]?[{LETTERS}]+[ ._\t-]*{CODE})(?![{BOUNDARY}])"
)
OBJECT = re.compile(
    rf"(?<![{BOUNDARY}])(?:obj(?:ect|ekt)?|объект(?:а|у|ом|е)?|工程|项目)"
    rf"[ ._\t-]*(?:№|#)?\s*(?P<before>{CODE})"
    rf"|(?<![{BOUNDARY}])(?P<after>{CODE})\s*(?:号)?(?:工程|项目)",
    re.IGNORECASE,
)
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
    value = re.sub(r"\s*/\s*", "/", prepared(text)).replace("_", "-")
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
                match.end(),
                Reference(
                    match[0], "object:" + strict_key(code), ("object", *signature(code))
                ),
            )
        )
    occupied_index = 0
    for match in DESIGNATION.finditer(text):
        while (
            occupied_index < len(occupied)
            and occupied[occupied_index][1] <= match.start()
        ):
            occupied_index += 1
        if occupied_index < len(occupied) and occupied[occupied_index][0] < match.end():
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


def query_text(question, aliases=DEFAULT_ALIASES):
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
    return lookup


def query_references(question, aliases=DEFAULT_ALIASES):
    yield from references(query_text(question, aliases))


@dataclass(frozen=True)
class Match:
    written: str
    candidates: tuple[str, ...]
    components: tuple["Match", ...] = ()

    def walk(self):
        yield self
        for component in self.components:
            yield from component.walk()

    def primary(self):
        if self.candidates:
            yield self
        else:
            for component in self.components:
                yield from component.primary()

    def as_dict(self):
        return {
            "written": self.written,
            "source_forms": list(self.candidates),
            "components": [part.as_dict() for part in self.components],
        }


@dataclass(frozen=True)
class Query:
    original: str
    matches: tuple[Match, ...]
    labels: tuple[str, ...] = ()

    @property
    def text(self):
        additions = [label for label in self.labels if label not in self.original]
        return self.original + ("\n" + " ".join(additions) if additions else "")

    @property
    def expression_context(self):
        def lines_for(node, depth=0):
            label = "组成项" if depth else "表达式"
            forms = " → " + "、".join(node.candidates) if node.candidates else ""
            yield f"{'  ' * depth}{label}：{node.written}{forms}"
            for part in node.components:
                yield from lines_for(part, depth + 1)

        lines = [
            line
            for match in self.matches
            if match.components
            for line in lines_for(match)
        ]
        return "【查询写法信息】\n" + "\n".join(lines) if lines else ""


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
        written_key = spelling_key(ref.text)
        explicit = {label for label in candidates if spelling_key(label) == written_key}
        ordered = tuple(
            sorted(explicit or candidates, key=lambda x: (-self.counts[x], x))
        )
        # Partition slash spans using source spellings, keeping numeric suffixes
        # and qualified prefixes inside any span that cannot stand alone.
        parts = re.split(r"\s*/\s*", ref.text)
        if len(parts) < 2:
            return Match(ref.text, ordered)

        @cache
        def partition(start):
            if start == len(parts):
                return (), 0
            best = None
            for end in range(len(parts), start, -1):
                if start == 0 and end == len(parts):
                    continue
                joined = "/".join(parts[start:end])
                combined = list(references(joined))
                if len(combined) != 1 or combined[0].text != joined:
                    continue
                rest = partition(end)
                if rest is None:
                    continue
                spans = (combined[0], *rest[0])
                coverage = rest[1] + (
                    len(joined) if self.candidates(combined[0]) else 0
                )
                if best is None or (coverage, -len(spans)) > (best[1], -len(best[0])):
                    best = spans, coverage
            return best

        best = partition(0)
        components = tuple(self.resolve(part) for part in best[0]) if best else ()
        return Match(ref.text, ordered, components)

    def query_roots(self, question, aliases):
        lookup = query_text(question, aliases)
        spans, offset = [], 0
        for ref in references(lookup):
            start = lookup.index(ref.text, offset)
            offset = start + len(ref.text)
            spans.append((start, offset, ref))
        index = 0
        while index < len(spans):
            start, end, chosen = spans[index]
            next_index = index + 1
            for following in range(index + 1, len(spans)):
                left, right, _ = spans[following]
                if not re.fullmatch(r"[ \t]+", lookup[end:left]):
                    break
                end = right
                written = lookup[start:end]
                compact = re.sub(r"[ \t]+", "", written)
                parsed = list(references(compact))
                if (
                    len(parsed) == 1
                    and parsed[0].text == compact
                    and self.candidates(parsed[0])
                ):
                    chosen = replace(parsed[0], text=written)
                    next_index = following + 1
            yield self.resolve(chosen)
            index = next_index

    def expand(self, question, aliases=DEFAULT_ALIASES, max_aliases=8):
        roots = list(self.query_roots(question, aliases))
        primary = {
            node.written: node.candidates for root in roots for node in root.primary()
        }
        components = {
            node.written: node.candidates
            for root in roots
            for node in root.walk()
            if node.written not in primary and node.candidates
        }
        selections = defaultdict(list)
        selected = {}
        # Whole expressions share the budget before component-only alternatives.
        for pools in (primary, components):
            for labels in zip_longest(*pools.values()):
                for written, label in zip(pools, labels):
                    if label is None:
                        continue
                    covered = pools is components and any(
                        label in full for full in selected
                    )
                    if covered or label in selected or len(selected) < max_aliases:
                        selections[written].append(label)
                        if not covered:
                            selected[label] = None

        def selected_match(node):
            return replace(
                node,
                candidates=tuple(selections[node.written]),
                components=tuple(selected_match(part) for part in node.components),
            )

        matches = tuple(selected_match(root) for root in roots)
        return Query(
            question,
            tuple(
                root for root in matches if any(node.candidates for node in root.walk())
            ),
            tuple(selected),
        )

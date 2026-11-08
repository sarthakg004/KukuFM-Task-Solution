"""Parse screenplays and prepare source-linked model input.

Times are episode-relative. Overlapping audiovisual entries retain source order.
"""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
import re


@dataclass
class Entry:
    source_id: str
    source_line: int
    kind: str
    label: str
    speaker: str | None
    start: str
    end: str
    start_seconds: float
    end_seconds: float
    text: str


@dataclass
class Scene:
    number: int
    heading: str
    start: str
    end: str
    start_seconds: float
    end_seconds: float
    entries: list[Entry]


@dataclass
class Episode:
    number: int
    source_file: str
    logline: str
    continuity_notes: str
    scenes: list[Scene]
    warnings: list[str]

    @property
    def entries(self) -> list[Entry]:
        """All entries in original source order, without filtering technical cues."""
        return [entry for scene in self.scenes for entry in scene.entries]

    @property
    def duration_seconds(self) -> float:
        return max(scene.end_seconds for scene in self.scenes)

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation."""
        return asdict(self)


def timestamp_seconds(value: str) -> float:
    """Convert MM:SS.fraction or HH:MM:SS.fraction to seconds."""
    pattern = r"\d{2,}:\d{2}(?::\d{2})?(?:\.\d+)?"
    if not re.fullmatch(pattern, value):
        raise ValueError(f"Invalid timestamp: {value!r}")

    parts = value.split(":")
    seconds = float(parts[-1])
    minutes = int(parts[-2])

    if seconds >= 60 or (len(parts) == 3 and minutes >= 60):
        raise ValueError(f"Invalid timestamp: {value!r}")

    hours = int(parts[0]) if len(parts) == 3 else 0
    return hours * 3600 + minutes * 60 + seconds


def _field(text: str, name: str) -> str:
    pattern = rf"^{re.escape(name)}:[ \t]*(.*)$"
    match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)

    if not match or not match[1].strip():
        raise ValueError(f"Missing or empty {name}")

    return match[1].strip()


def _interval(start: str, end: str) -> tuple[float, float]:
    left = timestamp_seconds(start)
    right = timestamp_seconds(end)

    if right <= left:
        raise ValueError(f"End must be after start: {start}–{end}")

    return left, right


class ScreenplayParser:
    """Read files and preserve the original source order, text and timestamps."""

    SCENE_PATTERN = re.compile(r"^SCENE\s+(\d+)[ \t]*$", re.MULTILINE)
    ENTRY_PATTERN = re.compile(r"^\*\*(.+?):\*\*[ \t]*$", re.MULTILINE)
    HEADER_PATTERN = (
        r"\s*Scene heading:[^\n]+\nScene start:[^\n]+\n"
        r"Scene end:[^\n]+\n\s*Actions and dialogue:\s*"
    )

    def parse_folder(self, folder: str | Path) -> list[Episode]:
        folder = Path(folder)
        if not folder.is_dir():
            raise FileNotFoundError(f"Screenplay directory not found: {folder}")

        paths = sorted(folder.glob("*.txt"))
        if not paths:
            raise ValueError(f"No .txt screenplays found in {folder}")

        episodes = [self.parse_file(path) for path in paths]
        return _ordered_episodes(episodes)

    def parse_file(self, path: str | Path) -> Episode:
        path = Path(path)
        text = path.read_text(encoding="utf-8-sig")

        try:
            return self._parse_episode(text, path)
        except ValueError as exc:
            raise ValueError(f"{path.name}: {exc}") from exc

    def _parse_episode(self, text: str, path: Path) -> Episode:
        number = self._episode_number(text, path)
        body_start, body_end = self._script_bounds(text)
        body = text[body_start:body_end]
        matches = list(self.SCENE_PATTERN.finditer(body))

        if not matches or body[:matches[0].start()].strip():
            raise ValueError("Missing scene or unexpected content before first scene")

        scenes = []
        warnings = []
        entry_count = 0

        for index, match in enumerate(matches):
            stop = matches[index + 1].start() if index + 1 < len(matches) else len(body)
            chunk = body[match.end():stop]
            offset = body_start + match.end()
            scene = self._parse_scene(
                chunk, int(match[1]), number, entry_count, text, offset
            )

            if any(previous.number == scene.number for previous in scenes):
                raise ValueError(f"Duplicate scene number {scene.number}")

            if scenes and scene.start_seconds < scenes[-1].end_seconds:
                warnings.append(f"Scene {scene.number}: overlaps or precedes previous scene")

            for entry in scene.entries:
                outside_start = entry.start_seconds < scene.start_seconds
                outside_end = entry.end_seconds > scene.end_seconds
                if outside_start or outside_end:
                    warnings.append(
                        f"{entry.source_id}: entry extends outside scene {scene.number}"
                    )

            scenes.append(scene)
            entry_count += len(scene.entries)

        return Episode(
            number=number,
            source_file=str(path),
            logline=_field(text[:body_start], "EPISODE LOGLINE"),
            continuity_notes=_field(text[:body_start], "CONTINUITY NOTES"),
            scenes=scenes,
            warnings=warnings,
        )

    def _parse_scene(
        self,
        chunk: str,
        scene_number: int,
        episode_number: int,
        entry_count: int,
        source: str,
        source_offset: int,
    ) -> Scene:
        headings = list(self.ENTRY_PATTERN.finditer(chunk))
        if not headings:
            raise ValueError(f"Scene {scene_number} has no entries")

        header = chunk[:headings[0].start()]
        if not re.fullmatch(self.HEADER_PATTERN, header, re.IGNORECASE):
            raise ValueError(f"Unexpected header content in scene {scene_number}")

        start = _field(header, "Scene start")
        end = _field(header, "Scene end")
        left, right = _interval(start, end)
        entries = []

        for index, heading in enumerate(headings):
            stop = headings[index + 1].start() if index + 1 < len(headings) else len(chunk)
            block = chunk[heading.end():stop].strip()
            source_line = source.count("\n", 0, source_offset + heading.start()) + 1
            entry = self._parse_entry(
                block=block,
                label=heading[1],
                source_id=f"E{episode_number:03d}-B{entry_count + index + 1:04d}",
                source_line=source_line,
            )
            entries.append(entry)

        return Scene(
            number=scene_number,
            heading=_field(header, "Scene heading"),
            start=start,
            end=end,
            start_seconds=left,
            end_seconds=right,
            entries=entries,
        )

    @staticmethod
    def _parse_entry(block: str, label: str, source_id: str, source_line: int) -> Entry:
        pattern = r"Start: \(([^)]+)\)\s*\|\s*End: \(([^)]+)\)[ \t]*\n([\s\S]+)"
        match = re.fullmatch(pattern, block)

        if not match or not match[3].strip() or "\nStart:" in match[3]:
            raise ValueError(f"Malformed entry: {source_id} ({label})")

        start_seconds, end_seconds = _interval(match[1], match[2])
        kinds = {"CAMERA": "camera", "ACTION": "action", "SOUND FX": "sound"}
        kind = kinds.get(label, "dialogue")
        speaker = label.split(" (", 1)[0] if kind == "dialogue" else None

        return Entry(
            source_id=source_id,
            source_line=source_line,
            kind=kind,
            label=label,
            speaker=speaker,
            start=match[1],
            end=match[2],
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            text=match[3].strip(),
        )

    @staticmethod
    def _episode_number(text: str, path: Path) -> int:
        header = _field(text, "EPISODE NUMBER")
        match = re.fullmatch(r"Episode\s+(\d+)", header)
        if not match:
            raise ValueError(f"Invalid episode number: {header!r}")

        number = int(match[1])
        filename = re.match(r"episode-(\d+)", path.name)
        if filename and int(filename[1]) != number:
            raise ValueError("Filename and header episode numbers disagree")

        return number

    @staticmethod
    def _script_bounds(text: str) -> tuple[int, int]:
        start_marker = "--- SCRIPT START ---"
        end_marker = "--- SCRIPT END ---"

        if text.count(start_marker) != 1 or text.count(end_marker) != 1:
            raise ValueError("Expected exactly one pair of script markers")

        start = text.index(start_marker) + len(start_marker)
        end = text.index(end_marker)
        trailing = text[end + len(end_marker):].strip()

        if end < start or trailing:
            raise ValueError("Invalid script markers or unexpected trailing content")

        return start, end


@dataclass
class ScreenplayWindow:
    window_id: str
    episode_numbers: list[int]
    text: str

    @property
    def character_count(self) -> int:
        return len(self.text)


def _ordered_episodes(episodes: Sequence[Episode]) -> list[Episode]:
    ordered = sorted(episodes, key=lambda episode: episode.number)
    numbers = [episode.number for episode in ordered]

    if len(numbers) != len(set(numbers)):
        raise ValueError("Duplicate episode numbers")

    return ordered


class ScreenplayPreprocessor:
    """Prepare full text and overlapping windows from a parsed screenplay pack."""

    def __init__(self, folder: str | Path):
        self.episodes = ScreenplayParser().parse_folder(folder)

    def full_text(self) -> str:
        return self._concatenate(self.episodes)

    def create_windows(
        self,
        window_size: int = 10,
        stride: int = 5,
    ) -> list[ScreenplayWindow]:
        valid_types = type(window_size) is int and type(stride) is int
        if not valid_types or not 1 <= stride <= window_size:
            raise ValueError("Require integers with 1 <= stride <= window_size")

        windows = []
        for start in range(0, len(self.episodes), stride):
            subset = self.episodes[start:start + window_size]
            windows.append(ScreenplayWindow(
                window_id=f"W{len(windows) + 1:03d}",
                episode_numbers=[episode.number for episode in subset],
                text=self._concatenate(subset),
            ))

            if start + window_size >= len(self.episodes):
                break

        return windows

    def _concatenate(self, episodes: Sequence[Episode]) -> str:
        return "\n\n".join(self._format_episode(episode) for episode in episodes)

    @staticmethod
    def _format_episode(episode: Episode) -> str:
        lines = [
            f"Episode {episode.number}",
            "CONTEXT ONLY (not selectable footage):",
            f"Logline: {episode.logline}",
            f"Continuity notes: {episode.continuity_notes}",
        ]
        if episode.warnings:
            lines.append("Source warnings: " + "; ".join(episode.warnings))

        lines.append("SCREENPLAY (timestamps are episode-relative MM:SS.fraction):")
        for scene in episode.scenes:
            lines.extend([
                "",
                f"Scene {scene.number}: {scene.heading} [{scene.start}–{scene.end}]",
            ])
            for entry in scene.entries:
                lines.append(
                    f"[{entry.source_id} | {entry.start}–{entry.end}] "
                    f"{entry.label}: {entry.text}"
                )

        return "\n".join(lines)

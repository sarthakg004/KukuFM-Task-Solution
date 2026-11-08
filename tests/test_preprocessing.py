"""Parser checks using temporary input, independent of the assignment pack."""

import tempfile
import unittest
from pathlib import Path

from scripts.preprocessing import ScreenplayPreprocessor


TEMPLATE = '''EPISODE NUMBER: Episode {number}
EPISODE LOGLINE: A warning interrupts a routine shift.
CONTINUITY NOTES: Synthetic parser fixture.

--- SCRIPT START ---

SCENE 01
Scene heading: INT. CONTROL ROOM – NIGHT
Scene start: 00:00.00
Scene end: 00:30.00

Actions and dialogue:

**ACTION:**
Start: (00:00.00) | End: (00:05.00)
The warning light turns red.

**OPERATOR (Urgent):**
Start: (00:05.00) | End: (00:30.00)
"Close the door."

--- SCRIPT END ---
'''


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        for number in (1, 2):
            path = self.folder / f'episode-{number:03d}-synthetic.txt'
            path.write_text(TEMPLATE.format(number=number))

    def test_parsing_preserves_source_links(self):
        pack = ScreenplayPreprocessor(self.folder)
        self.assertEqual([episode.number for episode in pack.episodes], [1, 2])
        text = pack.full_text()
        for episode in pack.episodes:
            self.assertFalse(episode.warnings)
            for entry in episode.entries:
                self.assertEqual(text.count(f'[{entry.source_id} |'), 1)

    def test_large_window_still_covers_a_small_pack(self):
        windows = ScreenplayPreprocessor(self.folder).create_windows(10, 5)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].episode_numbers, [1, 2])


if __name__ == '__main__':
    unittest.main()

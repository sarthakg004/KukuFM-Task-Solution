"""JSON/CSV exports preserve scores and ordered multi-segment footage."""

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.export_results import RankingExporter


class ExportTests(unittest.TestCase):
    def test_multisegment_csv_preserves_text_scores_and_segment_order(self):
        assessment = {
            name: {'score': score, 'reason': f'{name}: concrete evidence'}
            for name, score in [('hook', 4), ('cliffhanger', 3), ('clarity', 4), ('escalation', 3)]
        }
        assessment['concerns'] = ['Review the cut', 'Context gap']
        promo = {
            'rank': 1, 'score_rank': 2, 'candidate_id': 'P001', 'title': 'A warning',
            'hook': 'She says, "Stop!"\nThe door closes.',
            'cliffhanger': 'Can she escape?', 'why_it_works': 'Clear stakes, unresolved danger.',
            'posting_rationale': 'Mara’s strongest opening.', 'recommended': True,
            'score': 88.75, 'duration_seconds': 45, 'assessment': assessment,
            'resolved_segments': [
                {'episode': 3, 'start': '00:01.00', 'end': '00:21.00', 'start_seconds': 1, 'end_seconds': 21},
                {'episode': 4, 'start': '00:05.00', 'end': '00:30.00', 'start_seconds': 5, 'end_seconds': 30},
            ],
        }
        original = {'source_model': 'generator', 'judge_model': 'judge', 'candidates': [promo]}
        before = copy.deepcopy(original)
        with tempfile.TemporaryDirectory() as directory:
            paths = RankingExporter(directory).export(original)
            self.assertEqual(json.loads(paths['json'].read_text()), original)
            with paths['csv'].open(encoding='utf-8-sig', newline='') as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(original, before)
        self.assertEqual(len(rows), 2)
        self.assertEqual([row['episode'] for row in rows], ['3', '4'])
        self.assertEqual([row['segment_order'] for row in rows], ['1', '2'])
        for row in rows:
            self.assertEqual(row['hook'], promo['hook'])
            self.assertEqual(row['posting_rationale'], promo['posting_rationale'])
            self.assertEqual(row['hook_score'], '4')
            self.assertEqual(row['total_score'], '88.75')
            self.assertEqual(row['promo_duration_seconds'], '45')
        self.assertEqual(rows[1]['start_timestamp'], '00:05.00')
        self.assertEqual(rows[1]['segment_duration_seconds'], '25')

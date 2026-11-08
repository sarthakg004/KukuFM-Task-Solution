"""Export an existing ranking as JSON and one CSV row per ordered segment."""

import argparse
import csv
import json
from pathlib import Path


CRITERIA = ('hook', 'cliffhanger', 'clarity', 'escalation')


class RankingExporter:
    """Preserve the complete JSON and flatten its promo segments for review."""

    def __init__(self, output_dir: str | Path = 'outputs'):
        self.output_dir = Path(output_dir)

    def export(self, result: dict) -> dict[str, Path]:
        promos = sorted(result['candidates'], key=lambda promo: promo['rank'])
        rows = [
            self._row(result, promo, segment, order)
            for promo in promos
            for order, segment in enumerate(promo['resolved_segments'], 1)
        ]
        fields = [
            'rank', 'score_rank', 'candidate_id', 'title', 'segment_order',
            'episode', 'start_timestamp', 'end_timestamp',
            'segment_duration_seconds', 'promo_duration_seconds',
            'hook', 'cliffhanger', 'why_it_works', 'posting_rationale',
            'standalone_context', 'recommended',
            'hook_score', 'cliffhanger_score', 'clarity_score', 'escalation_score',
            'total_score',
            'hook_reason', 'cliffhanger_reason', 'clarity_reason', 'escalation_reason',
            'concerns', 'source_model', 'judge_model',
        ]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.output_dir / 'Final_Ranked_promos.json'
        csv_path = self.output_dir / 'Final_Ranked_promos.csv'

        # Export the saved judgments as recorded; no scores are recalculated.
        json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
        with csv_path.open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        return {'json': json_path, 'csv': csv_path}

    @staticmethod
    def _row(result: dict, promo: dict, segment: dict, order: int) -> dict:
        assessment = promo['assessment']
        row = {
            'rank': promo['rank'],
            'score_rank': promo.get('score_rank', ''),
            'candidate_id': promo['candidate_id'],
            'title': promo['title'],
            'segment_order': order,
            'episode': segment['episode'],
            'start_timestamp': segment['start'],
            'end_timestamp': segment['end'],
            'segment_duration_seconds': round(segment['end_seconds'] - segment['start_seconds'], 3),
            'promo_duration_seconds': promo['duration_seconds'],
            'hook': promo['hook'],
            'cliffhanger': promo['cliffhanger'],
            'why_it_works': promo['why_it_works'],
            'posting_rationale': promo['posting_rationale'],
            'standalone_context': promo.get('standalone_context', ''),
            'recommended': promo['recommended'],
            'total_score': promo['score'],
            'concerns': '; '.join(assessment.get('concerns', [])),
            'source_model': result.get('source_model', ''),
            'judge_model': result.get('judge_model', ''),
        }
        for name in CRITERIA:
            row[f'{name}_score'] = assessment[name]['score']
            row[f'{name}_reason'] = assessment[name]['reason']
        return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_json', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs'))
    args = parser.parse_args()
    result = json.loads(args.input_json.read_text())
    for kind, path in RankingExporter(args.output_dir).export(result).items():
        print(f'{kind.upper()}: {path}')


if __name__ == '__main__':
    main()

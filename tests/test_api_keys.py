"""Dynamic key discovery works with any count and nonconsecutive numbering."""

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.api_keys import discover_gemini_keys
from scripts.promo_generator import GenerationConfig, PromoGenerator


class KeyDiscoveryTests(unittest.TestCase):
    @patch.dict(os.environ, {
        'GEMINI_API_KEY10': 'test-ten',
        'GEMINI_API_KEY2': 'test-two',
        'GEMINI_API_KEY': 'test-base',
        'OTHER_API_KEY': 'not-gemini',
    }, clear=True)
    def test_discovers_any_number_of_keys_in_numeric_order(self):
        with tempfile.TemporaryDirectory() as directory:
            names = discover_gemini_keys(env_path=Path(directory) / '.env')
        self.assertEqual(names, ['GEMINI_API_KEY', 'GEMINI_API_KEY2', 'GEMINI_API_KEY10'])

    @patch.dict(os.environ, {
        'GEMINI_API_KEY1': 'same-test-value',
        'GEMINI_API_KEY2': 'same-test-value',
        'GEMINI_API_KEY3': '   ',
        'GEMINI_API_KEY4': '',
    }, clear=True)
    def test_skips_empty_keys_and_duplicate_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            names = discover_gemini_keys(env_path=Path(directory) / '.env')
        self.assertEqual(names, ['GEMINI_API_KEY1'])

    @patch.dict(os.environ, {'GEMINI_API_KEY1': 'shell-value'}, clear=True)
    def test_loads_dotenv_and_preserves_environment_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / '.env'
            path.write_text('GEMINI_API_KEY1=file-value\nGEMINI_API_KEY7=extra-value\n')
            self.assertEqual(discover_gemini_keys(env_path=path), [
                'GEMINI_API_KEY1', 'GEMINI_API_KEY7',
            ])
            self.assertEqual(os.environ['GEMINI_API_KEY1'], 'shell-value')

    @patch.dict(os.environ, {}, clear=True)
    def test_no_configured_keys_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(discover_gemini_keys(env_path=Path(directory) / '.env'), [])

    @patch('scripts.promo_generator.discover_gemini_keys', return_value=['AUTO_KEY'])
    @patch.object(PromoGenerator, 'generate_with_keys')
    def test_batch_does_not_require_a_manual_key_list(self, generate, discover):
        generate.return_value = {'candidates': []}
        window = MagicMock(window_id='W001', episode_numbers=[1], text='fixture')
        with tempfile.TemporaryDirectory() as directory:
            generator = PromoGenerator(GenerationConfig(), directory)
            with contextlib.redirect_stdout(io.StringIO()):
                result = generator.generate_windows([window], [], request_delay_seconds=0)
        self.assertEqual(result['windows'][0]['status'], 'success')
        self.assertEqual(generate.call_args.args[2], ['AUTO_KEY'])
        discover.assert_called_once_with(None)


if __name__ == '__main__':
    unittest.main()

import unittest
from unittest.mock import patch
from src.agents.conversational import (
    DEFAULT_GROQ_MODEL, _load_llm, _llm_comment, _llm_commentary,
    build_conversational_workflow, format_answer,
)

RESULT = {
    'status': 'completed', 'series_id': 'FOODS_1_001_CA_1_validation',
    'selected_model': 'SBA', 'selection_reason': 'historical validation',
    'diagnostics': {'demand_type': 'Lumpy'},
    'forecast_result': {'forecast': [0.5] * 28, 'total_expected_demand': 14.0},
}

class FakeGroq:
    def __init__(self, content='The selected method adjusts for intermittent sales.'):
        self.content = content
        self.calls = []
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        msg = type('Message', (), {'content': self.content})()
        choice = type('Choice', (), {'message': msg})()
        return type('Response', (), {'choices': [choice]})()

class GroqTests(unittest.TestCase):
    def test_offline_when_key_missing(self):
        with patch.dict('os.environ', {}, clear=True):
            self.assertEqual(_load_llm(), (None, 'groq'))

    def test_no_implicit_paid_openai(self):
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'not-used'}, clear=True):
            self.assertEqual(_load_llm(), (None, 'groq'))

    def test_groq_explanation(self):
        fake = FakeGroq()
        answer = format_answer(RESULT, fake, 'test-model', 'groq')
        self.assertIn('AI interpretation', answer)
        self.assertIn('14.00 units', answer)
        self.assertEqual(fake.calls[0]['model'], 'test-model')
        self.assertEqual(fake.calls[0]['max_completion_tokens'], 300)
        self.assertEqual(fake.calls[0]['reasoning_effort'], 'low')
        self.assertFalse(fake.calls[0]['include_reasoning'])
        self.assertEqual(fake.calls[0]['response_format'], {'type':'json_object'})
        self.assertEqual(len(fake.calls[0]['messages']), 2)
        self.assertNotIn('forecast', str(fake.calls[0]['messages'][1]['content']).lower().split('selection')[0])

    def test_reject_number_in_model_explanation(self):
        answer = format_answer(RESULT, FakeGroq('Forecast 999 units'), 'test', 'groq')
        self.assertNotIn('999', answer)
        self.assertIn('numerical claim', answer)

    def test_verified_number_is_allowed(self):
        outcome = _llm_commentary(
            RESULT, FakeGroq('{"explanation":"Expected demand is 14 units, while uncertainty remains."}'),
            'test', 'groq')
        self.assertEqual(outcome['llm_status'], 'success')
        self.assertIn('14 units', outcome['ai_explanation'])

    def test_invented_numerical_claim_is_rejected(self):
        outcome = _llm_commentary(
            RESULT, FakeGroq('{"explanation":"Demand should improve by 37%."}'), 'test', 'groq')
        self.assertEqual(outcome['llm_status'], 'invalid_response')
        self.assertEqual(outcome['llm_rejection_reason'], 'unsupported_numeric_claim')
        self.assertTrue(outcome['llm_content_returned'])
        self.assertGreater(outcome['llm_response_chars'], 0)

    def test_excessively_long_response_is_rejected(self):
        outcome = _llm_commentary(RESULT, FakeGroq('x' * 701), 'test', 'groq')
        self.assertEqual(outcome['llm_status'], 'invalid_response')
        self.assertEqual(outcome['llm_rejection_reason'], 'excessive_length')
        self.assertEqual(outcome['llm_response_chars'], 701)

    def test_api_error_falls_back(self):
        class Broken(FakeGroq):
            def create(self, **kwargs):
                raise RuntimeError('rate limited')
        self.assertIn('14.00 units', format_answer(RESULT, Broken(), 'test', 'groq'))

    def test_missing_key_records_fallback(self):
        outcome = _llm_commentary(RESULT, None, DEFAULT_GROQ_MODEL, 'groq')
        self.assertEqual(outcome['llm_status'], 'fallback')
        self.assertIn('GROQ_API_KEY', outcome['llm_error'])

    def test_offline_never_calls_groq(self):
        fake = FakeGroq()
        outcome = _llm_commentary(RESULT, fake, DEFAULT_GROQ_MODEL, 'offline')
        self.assertEqual(outcome['llm_status'], 'offline')
        self.assertEqual(fake.calls, [])

    def test_unsupported_model_is_sanitized(self):
        class BadModel(FakeGroq):
            def create(self, **kwargs):
                raise RuntimeError('model not found: unsupported')
        outcome = _llm_commentary(RESULT, BadModel(), 'old-model', 'groq')
        self.assertEqual(outcome['llm_status'], 'fallback')
        self.assertEqual(outcome['llm_error'], 'Groq rejected the configured model.')

    def test_rate_limit_is_sanitized(self):
        class RateLimitError(Exception): pass
        class Limited(FakeGroq):
            def create(self, **kwargs):
                raise RateLimitError('429 secret request details')
        outcome = _llm_commentary(RESULT, Limited(), 'test', 'groq')
        self.assertEqual(outcome['llm_error'], 'Groq rate limit exceeded.')
        self.assertNotIn('secret', outcome['llm_error'])

    def test_empty_response_uses_fallback(self):
        outcome = _llm_commentary(RESULT, FakeGroq(''), 'test', 'groq')
        self.assertEqual(outcome['llm_status'], 'invalid_response')
        self.assertFalse(outcome['ai_explanation'])
        self.assertEqual(outcome['llm_rejection_reason'], 'structured_content')
        self.assertFalse(outcome['llm_content_returned'])

    def test_commentary_preserved_in_graph_state_and_numbers_unchanged(self):
        class ForecastGraph:
            def invoke(self, request):
                return RESULT.copy()
        fake = FakeGroq()
        graph = build_conversational_workflow(
            forecast_graph=ForecastGraph(), llm_client=fake,
            llm_model='configured-model', llm_provider='groq')
        output = graph.invoke({
            'request':'Forecast FOODS_1_001 in store CA_1 for 28 days',
            'forecast_origin_day':1885, 'm5_sales_path':'unused.csv'})
        self.assertEqual(output['llm_status'], 'success')
        self.assertEqual(output['llm_provider'], 'groq')
        self.assertEqual(output['llm_model'], 'configured-model')
        self.assertEqual(output['ai_explanation'], fake.content)
        self.assertEqual(output['forecast_result_state']['forecast_result']['forecast'], [0.5] * 28)
        self.assertEqual(output['forecast_result_state']['forecast_result']['total_expected_demand'], 14.0)

if __name__ == '__main__':
    unittest.main()

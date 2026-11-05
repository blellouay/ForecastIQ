import unittest
from src.agents.conversational import extract_request, format_answer, RequestError

class ParserTests(unittest.TestCase):
    def test_product_and_store(self):
        p=extract_request('Forecast FOODS_1_001 in store CA_1 for the next 28 days using auto')
        self.assertEqual(p['series_id'],'FOODS_1_001_CA_1_validation')
        self.assertEqual(p['model'],'auto')
    def test_canonical(self):
        self.assertEqual(extract_request('forecast FOODS_1_008__TX_1 using NHITS_small')['model'],'NHITS_small')
    def test_refuse_missing_id(self):
        with self.assertRaises(RequestError):extract_request('Forecast product A')
    def test_refuse_wrong_horizon(self):
        with self.assertRaises(RequestError):extract_request('Forecast FOODS_1_001 CA_1 for 14 days')
    def test_origin(self):
        self.assertEqual(extract_request('forecast FOODS_1_001 CA_1 origin d_1857')['forecast_origin_day'],1857)
    def test_no_unverified_numbers_from_llm(self):
        class Fake:
            class responses:
                @staticmethod
                def create(**kwargs):
                    return type('R',(),{'output_text':'Forecast 999 units!'})()
        r={'status':'completed','series_id':'FOODS_1_001_CA_1_validation','selected_model':'SBA','selection_reason':'historical validation', 'forecast_result':{'forecast':[0.5]*28,'total_expected_demand':14}}
        answer=format_answer(r,Fake())
        self.assertIn('14.00',answer)
        self.assertNotIn('999',answer)
    def test_llm_failure_fallback(self):
        class Fake:
            class responses:
                @staticmethod
                def create(**kwargs):raise RuntimeError('offline')
        r={'status':'completed','series_id':'x','selected_model':'TSB','selection_reason':'manual', 'forecast_result':{'forecast':[1]*28,'total_expected_demand':28}}
        self.assertIn('28.00',format_answer(r,Fake()))

if __name__=='__main__':unittest.main()

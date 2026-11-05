"""Run from project root: python -m examples.run_step5"""
import json
import os
from src.agents.conversational import build_conversational_workflow

# Groq free tier when GROQ_API_KEY exists; otherwise deterministic offline explanation.
# Set FORECASTIQ_LLM_PROVIDER=offline to disable external API calls.
graph = build_conversational_workflow()
result = graph.invoke({
    'request': 'Forecast FOODS_1_001 in store CA_1 for the next 28 days using auto',
    'forecast_origin_day': 1885,
    'm5_sales_path': 'data/raw/sales_train_validation.csv',
})
print('Forecast status:', result['status'])
if result.get('forecast_result_state',{}).get('status') == 'completed':
    r = result['forecast_result_state']
    print('Selected model:',r['selected_model'])
print('\n' + result['response'])
print('\nAI provider:', result.get('llm_provider'))
print('AI model:', result.get('llm_model'))
print('AI status:', result.get('llm_status'))
print('AI content returned:', result.get('llm_content_returned'))
print('AI response characters:', result.get('llm_response_chars'))
if result.get('ai_explanation'):
    print('AI explanation:', result['ai_explanation'])
else:
    if result.get('llm_rejection_reason'):
        print('AI rejection reason:', result['llm_rejection_reason'])
    print('AI fallback reason:', result.get('llm_error') or 'No AI explanation was produced.')

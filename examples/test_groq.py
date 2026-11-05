import os
from groq import Groq
from src.agents.conversational import DEFAULT_GROQ_MODEL


def main():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise SystemExit("GROQ_API_KEY is required for this live connectivity example.")
    client = Groq(api_key=key)
    response = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        messages=[{
            "role": "user",
            "content": "Explain intermittent retail demand in one sentence."
        }],
        max_tokens=100,
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()

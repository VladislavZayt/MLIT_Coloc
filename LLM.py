import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key_env = os.getenv("OPENROUTER_API_KEY")


def get_answer_from_LLM(name_openrouter_model: str, prompt: str) -> str | None:

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key_env, 
    )

    try:
        completion = client.chat.completions.create(
            
            model=name_openrouter_model,
            
            # сообщение LLM
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
                ],
            
        )

        response_text = completion.choices[0].message.content
        return response_text

    except Exception as e:
        print(f"Произошла ошибка: {e}")


if __name__ == "__main__":
    print(get_answer_from_LLM("x-ai/grok-4.1-fast:free", "Привет"))

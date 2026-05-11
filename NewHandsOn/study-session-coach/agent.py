from __future__ import annotations

import json

from openai import OpenAI

from tools import SCHEMAS, execute


def _word_count(text: str) -> int:
    return len(text.split())


def run(
    api_key: str,
    notes: str,
    audio_path: str | None,
    image_path: str | None,
) -> str:
    client = OpenAI(api_key=api_key)

    parts: list[str] = []
    if notes:
        parts.append(f"Notes:\n{notes}")
    if audio_path:
        parts.append(f"Audio file to transcribe: {audio_path}")
    if image_path:
        parts.append(f"Image file to describe: {image_path}")

    user_content = "\n\n".join(parts) if parts else "No content provided."

    input_words = _word_count(notes or "")
    target_words = max(40, input_words // 4)

    messages: list = [
        {
            "role": "system",
            "content": (
                "You are a notes summarizer. "
                "Use the available tools to process any audio or image inputs first, then summarize all content together.\n\n"
                f"Output rules:\n"
                f"- Target ~{target_words} words total in the summary (roughly 1/4 of the input length). "
                "Scale bullet count to match: short inputs get 2-3 bullets, long inputs may use up to 6-8 bullets.\n"
                "- Each bullet: one clear sentence. No comma-separated lists inside bullets.\n"
                "- Merge overlapping points across modalities — do not repeat the same idea.\n"
                "- Infer a 2-4 word heading from the content.\n"
                "- Plain markdown only. No intro sentence, no preamble, no bullet sub-items."
            ),
        },
        {"role": "user", "content": user_content},
    ]

    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=SCHEMAS,
            tool_choice="auto",
        )
        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            return msg.content or ""

        for tc in msg.tool_calls:
            result = execute(client, tc.function.name, json.loads(tc.function.arguments))
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )

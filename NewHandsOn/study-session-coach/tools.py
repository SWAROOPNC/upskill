from __future__ import annotations

import base64
from pathlib import Path

from openai import OpenAI

SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "transcribe_audio",
            "description": "Transcribe a spoken audio recap into text using speech-to-text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "audio_path": {"type": "string", "description": "Path to the audio file"},
                },
                "required": ["audio_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_image",
            "description": "Extract text and key concepts from a notes image or whiteboard photo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "Path to the image file"},
                },
                "required": ["image_path"],
            },
        },
    },
]

_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def transcribe_audio(client: OpenAI, audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(model="whisper-1", file=f)
    return result.text


def describe_image(client: OpenAI, image_path: str) -> str:
    path = Path(image_path)
    mime = _MIME.get(path.suffix.lower(), "image/jpeg")
    b64 = base64.b64encode(path.read_bytes()).decode()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract all visible text and describe the key concepts from this study notes image.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content


def execute(client: OpenAI, name: str, args: dict) -> str:
    if name == "transcribe_audio":
        return transcribe_audio(client, args["audio_path"])
    if name == "describe_image":
        return describe_image(client, args["image_path"])
    return f"Unknown tool: {name}"

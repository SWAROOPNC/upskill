from __future__ import annotations

import os

import gradio as gr
from dotenv import load_dotenv

from agent import run

load_dotenv()


def _load_txt(path: str | None) -> str:
    if not path:
        return ""
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def build_summary(
    notes: str,
    txt_file: str | None,
    audio_file: str | None,
    image_file: str | None,
) -> str:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        return "OPENAI_API_KEY not set. Add it to your .env file."

    combined_notes = "\n\n".join(
        t for t in [(notes or "").strip(), _load_txt(txt_file)] if t
    )

    if not any([combined_notes, audio_file, image_file]):
        return "Please provide at least some notes, audio, or an image."
    try:
        return run(key, combined_notes, audio_file, image_file)
    except Exception as exc:
        return f"Error: {exc}"


with gr.Blocks(title="Multi modal notes summarizer") as demo:
    gr.Markdown(
        "# Multi modal notes summarizer\n"
        "Add your notes and optionally an audio recap or image — get a clean summary."
    )

    with gr.Row():
        with gr.Column():
            notes = gr.Textbox(
                label="Notes",
                lines=10,
                placeholder="Paste rough notes here...",
            )
            txt_file = gr.File(
                label="Or upload a .txt file",
                file_types=[".txt"],
            )
            audio_file = gr.Audio(
                label="Audio Recap (optional)",
                type="filepath",
                sources=["upload", "microphone"],
            )
            image_file = gr.Image(
                label="Notes Image (optional)",
                type="filepath",
            )
            submit = gr.Button("Summarise", variant="primary")

        with gr.Column():
            output = gr.Markdown(label="Summary")

    submit.click(
        fn=lambda: "⏳ Generating summary...",
        inputs=[],
        outputs=output,
        queue=False,
    ).then(
        fn=build_summary,
        inputs=[notes, txt_file, audio_file, image_file],
        outputs=output,
    )

if __name__ == "__main__":
    demo.launch()

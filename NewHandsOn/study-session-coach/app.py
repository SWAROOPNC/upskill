from __future__ import annotations

import os

import gradio as gr
from dotenv import load_dotenv

from agent import run

load_dotenv()


def build_summary(
    notes: str,
    audio_file: str | None,
    image_file: str | None,
) -> str:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        return "OPENAI_API_KEY not set. Add it to your .env file."
    if not any([(notes or "").strip(), audio_file, image_file]):
        return "Please provide at least some notes, audio, or an image."
    try:
        return run(key, notes or "", audio_file, image_file)
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
        inputs=[notes, audio_file, image_file],
        outputs=output,
    )

if __name__ == "__main__":
    demo.launch()

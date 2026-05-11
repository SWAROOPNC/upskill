# Multimodal Notes Summarizer

A hands-on project that turns messy study notes, voice recaps, and whiteboard photos into a clean, proportionate summary — powered by a real GPT-4o tool-calling agent.

---

## What It Does

Paste rough notes, upload a `.txt` file, record a voice note, or drop a photo of your whiteboard. Hit **Summarise** and get a concise bullet summary scaled to roughly 1/4 of your input length — not a fixed cap, so a short snippet gets 2–3 bullets while a long article gets 6–8.

---

## Key Concepts Applied

| Concept | How it shows up here |
|---|---|
| **Multimodal inputs** | Text (paste or `.txt` upload), audio (mic or file), image (photo/screenshot) |
| **Tool calling / agent loop** | GPT-4o decides which tools to invoke based on what modalities are present |
| **Real speech-to-text** | OpenAI Whisper-1 API transcribes audio before summarisation |
| **Real vision understanding** | GPT-4o Vision reads images via base64-encoded content blocks |
| **Proportionate summarisation** | Output target = `max(40, input_word_count // 4)` words |
| **Web app delivery** | Gradio UI with instant loading feedback via `.then()` chaining |

---

## Architecture

```
User (Gradio UI)
        │
        │  notes text / .txt upload / audio file / image file
        ▼
   app.py  ──── merges pasted notes + uploaded .txt ────► agent.run()
                                                               │
                                              ┌────────────────┤
                                              │  GPT-4o        │
                                              │  tool-calling  │
                                              │  loop          │
                                              └────┬───────────┘
                                                   │ tool_calls?
                                      ┌────────────┴─────────────┐
                                      │                           │
                               transcribe_audio            describe_image
                               (Whisper-1 API)          (GPT-4o Vision API)
                                      │                           │
                                      └────────────┬─────────────┘
                                                   │ tool results back to model
                                              GPT-4o final
                                              summary response
                                                   │
                                            Gradio Markdown
                                            output panel
```

### Agent loop detail (`agent.py`)

```
while True:
    response = gpt-4o(messages, tools=SCHEMAS, tool_choice="auto")
    if no tool_calls → return final summary
    for each tool_call:
        result = execute(tool_name, args)   # calls Whisper or GPT-4o Vision
        append tool result to messages
    # loop continues until model stops calling tools
```

---

## Project Structure

```
study-session-coach/
├── app.py            # Gradio UI + input merging + build_summary()
├── agent.py          # Tool-calling agent loop + proportionate output logic
├── tools.py          # Tool schemas (SCHEMAS), transcribe_audio(), describe_image()
├── requirements.txt  # Python dependencies
├── .env              # OPENAI_API_KEY — never committed (git-ignored)
├── .gitignore        # Ignores .env, .venv/, __pycache__, .DS_Store
└── README.md
```

### File responsibilities

**`tools.py`**
- `SCHEMAS` — two OpenAI function definitions (`transcribe_audio`, `describe_image`) passed to the model
- `transcribe_audio(client, path)` — calls Whisper-1; returns transcript text
- `describe_image(client, path)` — base64-encodes the image, calls GPT-4o Vision; returns extracted text + concepts
- `execute(client, name, args)` — dispatcher called by the agent loop

**`agent.py`**
- `run(api_key, notes, audio_path, image_path)` — entry point
- Calculates `target_words = max(40, input_words // 4)` and injects it into the system prompt
- Runs the tool-calling loop until GPT-4o stops invoking tools and returns the final summary

**`app.py`**
- Reads `OPENAI_API_KEY` from `.env` (no key field in UI)
- `_load_txt(path)` — reads an uploaded `.txt` file and returns its content as a string
- `build_summary(notes, txt_file, audio_file, image_file)` — merges pasted text + `.txt` upload, calls `agent.run()`
- Loading state: `submit.click(lambda: "⏳ Generating summary...", queue=False).then(build_summary, ...)` — instant feedback before the API round-trip completes

---

## Setup

```bash
cd NewHandsOn/study-session-coach

# create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# install dependencies (force PyPI if your machine has a private index)
pip install -r requirements.txt --index-url https://pypi.org/simple/

# add your OpenAI key
echo "OPENAI_API_KEY=sk-..." > .env

# run
python app.py
```

Open the local Gradio URL printed in the terminal (default: `http://127.0.0.1:7860`).

---

## Inputs

| Input | Type | Required |
|---|---|---|
| Notes (paste) | Free text in the textbox | At least one of these |
| Notes (`.txt` file) | Upload button below the textbox | |
| Audio Recap | Mic recording or uploaded audio file | Optional |
| Notes Image | Photo, screenshot, or whiteboard image | Optional |

Both notes inputs (paste + file) are merged before being sent to the model — you can use either or both.

---

## Output

A markdown bullet summary with an inferred 2–4 word heading. Output length scales with input:

- ~200 word input → ~50 word summary (2–3 bullets)
- ~900 word article → ~225 word summary (6–8 bullets)
- Overlapping points across modalities are merged, not repeated

---

## How the Output Length Is Controlled

```python
# agent.py
input_words = len(notes.split())
target_words = max(40, input_words // 4)

# injected into system prompt:
# "Target ~{target_words} words total in the summary (roughly 1/4 of the input length)."
```

The model scales bullet count naturally — short inputs get 2–3, long inputs get more — instead of hitting a hard cap that makes long-article summaries feel skeletal.

---

## Environment Variables

| Variable | Where | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | `.env` (git-ignored) | Authenticates all OpenAI API calls |

---

## Dependencies

```
gradio>=4.44.0       # web UI
openai>=1.0.0        # GPT-4o, Whisper-1, tool calling
Pillow>=10.4.0       # image file handling
python-dotenv>=1.0.0 # .env loading
```

# AI Research Digest

`ai-research-digest` collects recent AI news, papers, and Hugging Face model updates, asks a model running locally through Ollama to synthesize a concise technical briefing, and writes a dated dark-mode HTML dashboard to your Desktop. Every item retains its direct source link. Research papers include their RSS abstract, a plain-English explanation, and glossary definitions. No OpenAI API key or cloud-model account is required.

## Sources

- OpenAI News
- Anthropic (via the supplied Any Feeds endpoint)
- Redwood Research
- Wired AI
- Slashdot
- arXiv `cs.AI`
- Hugging Face News
- Hugging Face trending models (from the public Hub API and model cards)

For each source, the digest ranks eligible items using engagement metadata exposed by that source—such as reactions, comments, likes, downloads, or trending score—and uses publication recency as the fallback. It then selects up to two items that have not appeared during the preceding seven calendar days. The Hugging Face section gives a short explanation of each selected model and includes benchmark or evaluation details only when the model card supplies them; a trending score is treated as a popularity signal, not an accuracy ranking. An unavailable or malformed source is logged and skipped while the other sources continue processing.

The generated dashboard labels every item by content type, source tier, extraction quality, and publication date. Summaries attribute claims to their source. Research preprints show separate claim, method, evidence, limitations, and plain-English fields. Title-only items are marked as insufficient rather than guessed from, and unsupported numerical claims are omitted. Similar titles and canonical arXiv identifiers are used to remove cross-source duplicates while preferring primary, better-extracted material.

## Seven-day history

Successful runs record canonical source URLs in `.digest_history.json`. Links recorded today remain eligible, so regenerating today's report produces the best current selection even if it was already generated earlier that day. Only links recorded on one of the preceding seven calendar days are suppressed. Failed runs do not mark items as seen, and the first run after upgrading also scans dated digest files from prior days on the Desktop to avoid repeating their links.

The history path can be overridden with `DIGEST_HISTORY_FILE` in `.env`. To intentionally reset the history, delete `.digest_history.json` while the script is not running.

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com/download) running locally
- A locally downloaded Ollama model (the default is `llama3.2:3b`)
- Internet access when generating and viewing the digest (the HTML loads Marked and DOMPurify from a CDN)

## Install

```bash
git clone <your-repository-url> ai-research-digest
cd ai-research-digest
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Install Ollama if needed, then download the default model:

```bash
ollama pull llama3.2:3b
```

The Ollama desktop app normally starts the local service. Alternatively, start it from a terminal:

```bash
ollama serve
```

The default `.env` configuration is:

```dotenv
OLLAMA_MODEL=llama3.2:3b
OLLAMA_URL=http://127.0.0.1:11434
```

You can replace `llama3.2:3b` with any model shown by `ollama list`. The application permits only loopback Ollama URLs, ensuring prompts stay on this computer. Do not commit `.env`; it is excluded by `.gitignore`.

## Run

```bash
python main.py
```

The output is written to:

```text
~/Desktop/ai_research_digest_YYYY-MM-DD.html
```

To choose a different location:

```bash
python main.py --output ./ai_research_digest.html
```

Open the generated file in a web browser. The dated default name preserves earlier daily digests. An explicit `--output` path is replaced when reused.

## Daily cron job (macOS or Linux)

First run the script manually and confirm it works. Then obtain the repository's absolute path with `pwd` and edit your crontab:

```bash
crontab -e
```

The included wrapper checks whether Ollama is running, starts it if necessary, waits until it is healthy, runs the digest, and then stops Ollama only when the wrapper started it. If Ollama was already running, it is left running.

Test the wrapper manually:

```bash
./run_digest_cron.sh
```

For a daily run at 7:00 AM, add one line, replacing `/absolute/path/to` with the real parent directory:

```cron
0 7 * * * /absolute/path/to/ai-research-digest/run_digest_cron.sh >> /absolute/path/to/ai-research-digest/digest-cron.log 2>&1
```

The wrapper resolves the repository and virtual-environment paths itself, so it does not depend on cron's working directory. It recognizes Ollama in `/usr/local/bin`, `/opt/homebrew/bin`, and the standard macOS application location. For a custom installation, set `OLLAMA_BIN` to its absolute path in the crontab. On macOS, cron may need permission to write to the Desktop under **System Settings → Privacy & Security**.

## Troubleshooting

- **Cannot reach local Ollama:** open the Ollama app or run `ollama serve` in another terminal.
- **Model not found:** run `ollama pull llama3.2:3b`, or set `OLLAMA_MODEL` in `.env` to a name reported by `ollama list`.
- **No feed entries:** verify internet access. Individual feed failures are warnings; the run stops only if all feeds fail.
- **Low evidence or insufficient content:** the source exposed only metadata, an RSS snippet, or a title. The dashboard labels this rather than allowing the model to fill in missing facts.
- **Ollama summary failure:** the report still renders using attributed source excerpts, and the failure is recorded as a warning in the log.
- **Dashboard says it cannot load the renderer:** reconnect to the internet and reload the HTML file so the CDN scripts can load.

The local inference call follows Ollama's official [`POST /api/chat`](https://docs.ollama.com/api/chat) interface with streaming disabled. Ollama's local API does not require authentication.

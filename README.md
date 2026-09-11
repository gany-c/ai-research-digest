# AI Research Digest

`ai-research-digest` collects the two newest entries from six AI and technology RSS feeds, asks a model running locally through Ollama to synthesize a concise technical briefing, and writes a dated dark-mode HTML dashboard to your Desktop. Every item retains its direct source link. Research papers include their RSS abstract, a plain-English explanation, and glossary definitions. No OpenAI API key or cloud-model account is required.

## Sources

- OpenAI News
- Anthropic (via the supplied Any Feeds endpoint)
- Redwood Research
- Wired AI
- Slashdot
- arXiv `cs.AI`

An unavailable or malformed feed is logged and skipped; the other feeds continue processing.

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
- **Dashboard says it cannot load the renderer:** reconnect to the internet and reload the HTML file so the CDN scripts can load.

The local inference call follows Ollama's official [`POST /api/chat`](https://docs.ollama.com/api/chat) interface with streaming disabled. Ollama's local API does not require authentication.

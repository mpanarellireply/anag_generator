# SQL Package Generator

Multi-agent system that generates PL/SQL validation functions from an Excel analysis document. Uses LangChain + OpenAI to parse specifications, render SQL via Jinja2 templates, and review the output.

## Pipeline

1. **Parser Agent** - Reads Excel, groups by function name (`FUNZIONE.1`), uses LLM to structure parameters and controls into `FunctionSpec` objects
2. **Generator Agent** - Renders each `FunctionSpec` through `template.j2` into a `.sql` file with TODO placeholders for validation logic
3. **Reviewer Agent** - LLM-based review of generated SQL for correctness and completeness

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:
```
OPENAI_API_KEY=sk-...
```

## Usage

```bash
# Process all functions
python main.py NEBULA_ANG_ControlliB2B.xlsx

# Process a single function
python main.py NEBULA_ANG_ControlliB2B.xlsx --function NFD_CTRL_FBROK

# Force re-parse (ignore cache)
python main.py NEBULA_ANG_ControlliB2B.xlsx --force-parse

# Custom output and cache directories
python main.py NEBULA_ANG_ControlliB2B.xlsx --output-dir out/ --cache-dir .cache/

# Use a different model
python main.py NEBULA_ANG_ControlliB2B.xlsx --model gpt-5-mini
```

Parsed specs are cached in `parser_cache/parsed_specs.json` after the first run. Subsequent runs skip the LLM parsing step unless `--force-parse` is used.

## Project Structure

```
├── main.py                  # CLI entry point
├── template.j2              # Jinja2 SQL template (static)
├── requirements.txt
├── src/
│   ├── models.py            # Pydantic data models
│   ├── excel_parser.py      # Excel reading & grouping
│   ├── orchestrator.py      # Pipeline coordination
│   └── agents/
│       ├── parser_agent.py  # LLM-based spec extraction
│       ├── generator_agent.py  # Template rendering
│       └── reviewer_agent.py   # LLM-based SQL review
├── parser_cache/               # Parsed specs cache
└── output/                  # Generated SQL files
```

## Deploy

### 1. Build and export the Docker image

```bash
# Build the image (use --platform linux/arm64 if the target server is ARM-based)
docker build --platform linux/arm64 -t anag-generator .

# Save the image as a tar file
docker save -o anag-generator.tar anag-generator

# Copy the tar to the remote server
scp anag-generator.tar user@your-server:/tmp/
```

### 2. Load the image on the server

```bash
docker load -i /tmp/anag-generator.tar

# Clean up the tar file
rm /tmp/anag-generator.tar
```

### 3. Run with Docker Compose

Create a `docker-compose.yml` on the server:

```yaml
services:
  anag-generator:
    image: anag-generator
    ports:
      - "${PORT}:${PORT}"
    env_file:
      - .env
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./jobs:/app/jobs
      - ./debug:/app/debug
      - ./output:/app/output
      - ./parser_cache:/app/parser_cache
      - ./logs:/app/logs
```

Place a `.env` file next to it with your configuration, then start the service:

```bash
docker compose up -d
```

The mounted volumes allow you to inspect logs, debug output, generated SQL, parser cache, and jobs directly from the host filesystem.

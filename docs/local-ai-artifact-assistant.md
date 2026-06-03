# Local AI artifact assistant

This repository includes a small local assistant for implementation project
files. It is designed for project artifacts such as handover notes, integration
specifications, SQL scripts, configuration exports, logs, meeting notes, and
MES process descriptions.

The assistant keeps data local:

- files are read from directories you provide;
- the index is stored in `.ai_artifacts/index.json`;
- neural search and answers use a local Ollama server;
- no cloud API keys are required.

## 1. Install local models

Install Ollama from <https://ollama.com/download>, start it, then run:

```bash
scripts/install_ollama_models.sh
```

By default the script pulls:

- `llama3.1:8b` for answers and summaries;
- `nomic-embed-text` for neural search embeddings.

You can override the models:

```bash
GENERATION_MODEL=qwen2.5:7b EMBEDDING_MODEL=nomic-embed-text scripts/install_ollama_models.sh
```

Check the local runtime:

```bash
python3 tools/ai_artifact_assistant.py doctor
```

## 2. Add project artifacts

Create a local folder for files you do not want to commit:

```bash
mkdir -p project_artifacts
```

Copy project materials into that folder, for example:

- MES functional specifications;
- ISA-95 object mappings;
- integration contracts with ERP, WMS, LIMS, SCADA, or PLC layers;
- CSV/JSON/XML samples;
- SQL scripts;
- logs and incident notes;
- meeting minutes and open-question lists.

Binary formats such as PDF, DOCX, and XLSX should be exported to text, Markdown,
CSV, JSON, or XML before indexing.

## 3. Build the index

```bash
python3 tools/ai_artifact_assistant.py index project_artifacts src README.md
```

If Ollama is unavailable, the tool still creates an index and uses lexical
search:

```bash
python3 tools/ai_artifact_assistant.py index project_artifacts --no-embeddings
```

## 4. Search project context

```bash
python3 tools/ai_artifact_assistant.py search "SAP production order confirmation"
```

## 5. Ask questions

```bash
python3 tools/ai_artifact_assistant.py ask "Which MES interfaces are mentioned and what is still missing?"
```

The answer is generated only from indexed context. If the context is incomplete,
the model is instructed to say what is missing.

## 6. Summarize artifacts

Summarize files directly:

```bash
python3 tools/ai_artifact_assistant.py summarize project_artifacts
```

Or summarize the existing index:

```bash
python3 tools/ai_artifact_assistant.py summarize
```

## Operational notes for enterprise projects

- Keep customer data in `project_artifacts/`; it is ignored by Git.
- Rebuild the index after adding or changing files.
- Use separate artifact folders per client or site when projects must stay
  isolated.
- Review generated answers before sending them to customers; the assistant is a
  drafting and analysis aid, not a source of contractual truth.

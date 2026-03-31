# LlamaIndex Tools Integration: Octen

This tool connects to [Octen](https://octen.ai), a fast real-time web search API for AI, to enable your agent to search and retrieve content from the Internet.

To begin, you need to obtain an API key at [octen.ai](https://octen.ai).

## Installation

```bash
pip install llama-index-tools-octen
```

## Usage

```python
import os
from llama_index.tools.octen import OctenToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

octen_tool = OctenToolSpec(
    api_key=os.environ["OCTEN_API_KEY"],
)
agent = FunctionAgent(
    tools=octen_tool.to_tool_list(),
    llm=OpenAI(model="gpt-4.1"),
)

response = await agent.run(
    "What are the latest developments in AI?"
)
print(response)
```

## Available Tools

- `search`: Search the web using Octen for a list of results matching a natural language query.

- `search_and_retrieve_documents`: Search and retrieve full page content as LlamaIndex Documents.

- `search_and_retrieve_highlights`: Search and retrieve highlighted snippets as LlamaIndex Documents.

- `current_date`: Utility for the agent to get today's date (useful for time-filtered searches).

## Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str` | required | Octen API key |
| `verbose` | `bool` | `True` | Print search metadata |
| `max_results` | `int` | `5` | Default number of results |
| `max_characters` | `int` | `2000` | Max characters for full content retrieval |
| `timeout` | `float` | `None` | Request timeout in seconds |

### Per-query Parameters

All search functions support:

| Parameter | Type | Description |
|---|---|---|
| `query` | `str` | Natural language search query |
| `num_results` | `int` | Number of results (overrides default) |
| `include_domains` | `list[str]` | Restrict to these domains |
| `exclude_domains` | `list[str]` | Exclude these domains |
| `search_type` | `str` | `auto`, `keyword`, or `semantic` |
| `start_time` | `str` | Start date filter (ISO 8601) |
| `end_time` | `str` | End date filter (ISO 8601) |
| `time_basis` | `str` | `auto`, `published`, or `crawled` |
| `include_text` | `list[str]` | Text that must appear in results |
| `exclude_text` | `list[str]` | Text to exclude from results |
| `safesearch` | `str` | `off` or `strict` |
| `format` | `str` | `text` or `markdown` |

`search_and_retrieve_highlights` additionally supports:

| Parameter | Type | Description |
|---|---|---|
| `highlight_max_tokens` | `int` | Max tokens for highlighted snippets (default: 200) |

## TinyWorlds LLM Policies

The engine ships with stubbed policies in `tinyworld_minimal.py`. To drive an
agent with an OpenAI model, install the OpenAI SDK and swap in the helper from
`tinyworlds_openai.py`:

```bash
pip install openai
export OPENAI_API_KEY=sk-your-key
```

```python
from tinyworlds_openai import make_openai_policy
from tinyworld_minimal import POLICIES

POLICIES["sheep"] = make_openai_policy(model="gpt-4o-mini")
```

When the simulation runs, the selected agent will receive the full TinyWorlds
prompt and return actions via the OpenAI chat completions API. All responses
are still validated, so invalid JSON falls back to a safe default turn.

Test

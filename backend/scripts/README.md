# `backend/scripts` — manual developer tools

Nothing in the application imports these. They are run by hand when you need
to check something against the real world. They are not tests: the test suite
must run with no network and no provider key.

Both scripts read `NVIDIA_API_KEY` and `NVIDIA_API_BASE_URL` from `.env` at the
repository root and **never print the key**.

## `probe_models.py`

Answers: *"Which model IDs may my key actually invoke?"* Calls `GET /models` on
the configured endpoint and prints the IDs. Use it when setting up a key or
when a model name returns a 403/404.

```bash
cd backend
python -m scripts.probe_models
```

## `measure_token_ceiling.py`

Answers: *"Does our chunk size fit inside the embedding model's token limit,
or is it silently truncated?"* Sends texts of 500, 1000, 2000, 3000, and 4000
characters to the real embedding endpoint and reports acceptance and vector
dimension.

```bash
cd backend
TMPDIR=/tmp python -m scripts.measure_token_ceiling
```

Run it once when choosing an embedding model, and again whenever the model or
`MAX_CHUNK_CHARS` changes.

## Adding a script

- Keep it runnable as `python -m scripts.<name>` from `backend/`.
- Never write secrets to stdout or to a file.
- If it talks to a provider, say so in its docstring and make the failure mode
  explicit (a message, not a traceback full of request details).

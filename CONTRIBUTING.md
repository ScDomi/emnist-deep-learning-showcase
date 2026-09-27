# Contributing

This is a compact portfolio project, so contributions should keep the repo reproducible and easy to review.

## Local checks

```bash
python -m py_compile src/emnist_dl/*.py tests/*.py
pytest -q
```

## Rules of thumb

- Do not commit downloaded EMNIST data.
- Do not commit large model checkpoints unless they are explicitly released as GitHub release assets.
- Prefer CLI-reproducible experiments over notebook-only changes.
- Keep metrics in `reports/` tied to the exact model/config that produced them.

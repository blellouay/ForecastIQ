# Publication checklist

- [ ] Choose and add an open-source license.
- [ ] Initialize Git locally and inspect the staged file list.
- [ ] Confirm ignored raw data, weights, checkpoints, archives, caches, and secrets are not staged.
- [ ] Run `python -m examples.quickstart`.
- [ ] Run `python -m pytest -q -m "not integration"`.
- [ ] Run `python -m ruff check src dashboard examples tests`.
- [ ] Review notebook diffs and the notebook rebrand report.
- [ ] Capture an authentic dashboard screenshot if desired.
- [ ] Decide how optional model artifacts will be distributed; do not invent URLs.
- [ ] Create the GitHub repository with the recommended description and topics.
- [ ] Push only after an explicit final secret and large-file review.

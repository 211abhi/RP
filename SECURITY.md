# Security and data handling

- Commit no raw or derived patient-level data, credentials, access tokens, or DUA-bound
  artifacts.
- Keep `data/`, `artifacts/`, `checkpoints/`, and `logs/` ignored.
- Never place raw identifiers or clinical free text into prompts or logs.
- Treat even deidentified event combinations as sensitive research data.
- Validate archive paths before extraction and do not extract as root.
- Review generated reports for identifiers before sharing.
- Report suspected exposure to the repository owner and follow institutional policy.


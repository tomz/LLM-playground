# 09 — Safety & Red Team

## Categorical evals (gate every release)

- **CBRN uplift**: chem/bio/radiological/nuclear knowledge probes; expert-grader rubric.
- **Cyber**: CTF challenges (Cybench), exploit generation, malware writing.
- **Persuasion / manipulation**: MakeMePay, MakeMeSay style.
- **Autonomy**: METR-style agentic task suite, self-exfiltration probes.
- **Bias / fairness**: BBQ, BOLD, demographic parity probes.
- **Jailbreak robustness**: HarmBench, AdvBench, multi-turn social-engineering suite.

## Run-time defenses

- Pre-deployment: classifier sandwich (input + output) for disallowed content.
- Tool-use sandbox: code execution in gVisor/Firecracker, network egress denied by default.
- Rate limiting + abuse detection on the API edge.
- Audit log of every prompt for X days (subject to privacy policy).

## Responsible Scaling Policy hooks

The codebase exposes `safety.gates.preflight(model_card)` which must return PASS before any checkpoint is promoted to staging. Failures block CD pipeline.

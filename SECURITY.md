# Security Policy

## Supported versions

omlx is installed from git and versioned from the tip of `develop`; there are no
back-supported release branches. Security fixes land on `develop`, and only the
latest commit is supported. Re-run `uv tool install --force
git+https://github.com/ahokinson/omlx` (or `uv sync` in a clone) to update.

## Reporting a vulnerability

Report vulnerabilities privately through [GitHub private vulnerability
reporting](https://github.com/ahokinson/omlx/security/advisories/new) on this
repository. Do not open a public issue for a security bug.

Include the affected commit, a description of the issue, and steps to reproduce
(a proof of concept if you have one). Expect an initial acknowledgement within a
few days. Once a fix is available it will be committed to `develop` and, where
warranted, published as a GitHub Security Advisory with credit to the reporter.

## Threat model

omlx is a local, single-user tool. Its defaults assume the machine and the
loopback interface are trusted:

- **The server binds `127.0.0.1` and has no authentication.** The `api_key` is
  ignored. Do not expose the port (`11434`) to an untrusted network or bind it
  to a public interface; anyone who can reach the port can run inference and
  enumerate loaded models.
- **Models are untrusted code paths.** `omlx pull` downloads arbitrary Hugging
  Face repositories and loads their weights and chat templates; `--convert` runs
  `mlx_lm.convert` on-device. Only pull models from repositories you trust.
- **On-disk state is unencrypted.** Weights live in the HF hub cache
  (`~/.cache/huggingface/hub`) and the registry in `~/.omlx/models.json`,
  protected only by filesystem permissions.

Reports that amount to "the API has no auth" or "a malicious model can misbehave"
are known trade-offs of the local-first design, not vulnerabilities. Issues that
break out of that model — remote-reachable code execution, path traversal out of
the cache, a crafted request escaping the loopback assumption — are in scope.

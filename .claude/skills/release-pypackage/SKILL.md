---
name: release-pypackage
description: Cut a new pto-kernels Python package release - preflight CI checks, version bump, annotated git tag, tag push, and the GitHub release. Use when the user asks to release, cut a release, bump the version, tag a new version, or publish vX.Y.Z of pto-kernels.
---

# Release the pto-kernels Python package

Drives the release process described in `RELEASE.md`. The repo is
`huawei-csl/pto-kernels` on GitHub, mirrored to an internal GitLab
(`pto-kernels-mirror`) that runs the 910B2/910B4 hardware CI.

**Never push a tag or publish a release without the user's explicit go-ahead
for that specific version.** Tags and releases are visible to everyone and are
painful to retract.

## 0. Determine the target version

- Current version: `version = ` line in `pyproject.toml` (single source of truth).
- Ask the user for the new version if they did not give one; suggest the next
  patch bump. Semver, no `v` prefix in files, **with** `v` prefix on the tag.
- Refuse to continue if `git tag -l vX.Y.Z` already shows the tag.

## 1. Preflight (report, do not silently skip)

Run these and show the user a short pass/fail summary. Stop and ask before
continuing if anything is red.

```bash
git switch main && git pull --ff-only          # release from main only
git status --short                             # must be clean of release-relevant edits
gh run list --branch main --limit 10           # GitHub CI green on the release commit
pre-commit run --all-files                     # style gates CI enforces
```

- **Hardware CI**: unit tests on both **910B4** and **910B2** live in the
  internal GitLab mirror pipeline, not GitHub. You cannot see it. Ask the user
  to confirm the mirror pipeline is green and record their answer.
- **Local tests**: if an NPU is present, run `make test`; otherwise say so
  explicitly rather than claiming it passed.

## 2. Version bump

```bash
git grep -n <old-version>      # find every file carrying the version
```

Today only `pyproject.toml` holds it, but always run the grep - docs, CMake
files, or example wheels may start carrying it. Ignore build artifacts,
`node_modules/`, `venv/`, `build/`, and stale `*.whl` files in the repo root.

Edit each real occurrence, then commit on a branch:

```bash
git switch -c release/vX.Y.Z
git add -u && git commit -m "chore(release): bump version to X.Y.Z"
```

`main` takes changes through pull requests, so open one (the
`pypto-developer:github-pr` skill handles the fork/maintainer-edit cases) and
wait for it to merge before tagging. If the user says they can push straight to
`main`, do that instead and say which path you took.

## 3. Tag the release commit

After the bump is on `main`:

```bash
git switch main && git pull --ff-only
git log --oneline -1                                   # show the user what gets tagged
git tag -a vX.Y.Z -m "Release version vX.Y.Z"
```

The `v` prefix on the tag is required. Annotated (`-a`), never lightweight.

## 4. Push the tag (confirm first)

```bash
git push origin vX.Y.Z
```

## 5. GitHub release

Preferred, if `gh auth status` works and the installed `gh` supports it:

```bash
gh release create vX.Y.Z --title "vX.Y.Z" --generate-notes
```

The system `gh` here is old (2.4.0) and may reject `--generate-notes`. On any
failure, do not hand-write notes - give the user the link and the two clicks:

<https://github.com/huawei-csl/pto-kernels/releases/new?tag=vX.Y.Z>
→ "Generate release notes" → "Publish release".

## 6. Post-release

- Trigger / point at the wheel build: the `Multi-platform (x86, aarch64)`
  workflow (`.github/workflows/python-packaging.yml`) is `workflow_dispatch`
  only, so it does **not** run on a tag. Run
  `gh workflow run python-packaging.yml` (or tell the user to dispatch it) and
  check the manylinux wheels for py3.10-3.12 on x86_64 and aarch64 build.
- Close issues/milestones tied to the release.
- Report back: tag, release URL, wheel-build run URL, and anything left manual.

## Reporting

End with a plain list of what actually happened - which preflight checks really
ran, which the user vouched for, and which steps are still pending. Do not
describe a step as done when it was skipped or delegated.

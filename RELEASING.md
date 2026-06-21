# Releasing waxsql

This is a short operator's manual for cutting a release. The
mechanics are mostly automated by `.github/workflows/release.yml`;
this doc covers the human-side steps and the one-time PyPI setup.

## One-time setup: Trusted Publisher on PyPI

The release workflow uses GitHub Actions OIDC (no API token in
GitHub Secrets). The PyPI side needs a one-time configuration:

1. Sign in at <https://pypi.org/> as a maintainer of the `waxsql`
   project.
2. Go to <https://pypi.org/manage/account/publishing/>.
3. Add a Trusted Publisher with:
   - **Publisher**: GitHub Actions
   - **Owner**: `pgexperts`
   - **Repository name**: `waxsql`
   - **Workflow filename**: `release.yml`
   - **Environment name**: `pypi`

The `environment: pypi` clause in the workflow scopes the OIDC
token to a named GitHub environment. You can optionally add
protection rules to that environment (manual approval, required
reviewers, deployment branches) under
**Settings → Environments → pypi** in the GitHub repo — useful
if you want a release to require a click from a second maintainer
before the upload proceeds.

For the very first release ever, PyPI also requires "pending"
Trusted Publisher: register the publisher *before* the project
exists on PyPI, then the first publish creates the project under
the configured publisher's authority.

## Cutting a release

Once Trusted Publisher is configured, the workflow takes care of
the rest. The maintainer's job is:

1. Make sure `main` is green:

   ```sh
   gh run list --workflow=ci.yml --branch=main --limit=1
   ```

2. Bump the version in `pyproject.toml`:

   ```toml
   [project]
   version = "X.Y.Z"
   ```

   Stick to PEP 440. For pre-releases, `1.1.0rc1` / `1.1.0.dev1`
   are accepted by the workflow's tag glob.

3. Update `README.md` and `FUTURE.md` if anything shifts (rates,
   features, etc.).

4. Commit + push the bump:

   ```sh
   git add pyproject.toml README.md FUTURE.md
   git commit -m "Bump version to X.Y.Z"
   git push origin main
   ```

   Wait for CI to pass.

5. Tag the release. The tag MUST be `vX.Y.Z` (the workflow
   verifies tag == pyproject version and fails fast otherwise):

   ```sh
   git tag -a vX.Y.Z -m "waxsql X.Y.Z"
   git push origin vX.Y.Z
   ```

6. Watch the release workflow:

   ```sh
   gh run watch
   ```

   The three jobs run in sequence: `build` → `publish` → `github_release`.

7. Verify the upload:
   - PyPI: <https://pypi.org/project/waxsql/>
   - GitHub Release: <https://github.com/pgexperts/waxsql/releases>
   - In a fresh venv: `pip install waxsql==X.Y.Z && python -c "import waxsql; print(waxsql.__version__)"`

## What the workflow does, step by step

`.github/workflows/release.yml` runs three jobs on a tag push:

**`build`** — checkout, set up Python 3.13, install `build` and
`twine`, verify that the tag's version matches pyproject's version
(belt-and-braces against tagging the wrong commit), build wheel +
sdist, run `twine check --strict` (warnings become failures),
upload both as a workflow artifact.

**`publish`** — depends on build, runs in the `pypi` GitHub
Environment (which is what the PyPI Trusted Publisher binds to).
Downloads the artifact, calls
`pypa/gh-action-pypi-publish@release/v1` with `id-token: write`
permission. The action obtains a short-lived OIDC token from
GitHub, presents it to PyPI, and uploads the distributions.

**`github_release`** — depends on publish, creates a GitHub
Release for the tag with the built distributions attached and an
auto-generated body listing commits since the previous tag. The
maintainer can edit the release notes after the fact.

## Backing out a bad release

PyPI does NOT permit re-uploading a deleted version. If you ship
a broken `1.0.1`, the recovery is to ship `1.0.2` (or `1.0.1.post1`
for an out-of-band tag with no code changes).

For a YANK (mark a release as not-recommended but still
installable for pinning compatibility):

```sh
# In the PyPI web UI: project → manage → release → Yank.
# Or via the API: https://docs.pypi.org/api/upload/#yanking-a-release
```

Yanked versions disappear from `pip install waxsql` (gets the
previous non-yanked version) but stay available when explicitly
pinned (`pip install waxsql==1.0.1` still works).

## TestPyPI dry-run (optional)

If you want to dry-run the publish path before doing the real
release:

1. Set up a parallel Trusted Publisher on TestPyPI:
   <https://test.pypi.org/manage/account/publishing/>. Same shape,
   different host.

2. Temporarily edit `.github/workflows/release.yml`'s publish step
   to point at TestPyPI:

   ```yaml
   - name: Publish to PyPI
     uses: pypa/gh-action-pypi-publish@release/v1
     with:
       repository-url: https://test.pypi.org/legacy/
   ```

3. Push a tag, watch the workflow, verify the result at
   <https://test.pypi.org/project/waxsql/>.

4. Revert the workflow edit before tagging the real release.

Most of the time the local `python -m build && twine check --strict`
covers everything that would surprise on TestPyPI, so this is
rarely needed in practice.

## Versioning

Semantic versioning. Roughly:

- **MAJOR**: breaking changes to the public API surface (the
  `__all__` in `waxsql/__init__.py`).
- **MINOR**: backwards-compatible additions — new generator
  features, new public functions, new optional dependencies.
- **PATCH**: bug fixes, comment improvements, dependency-pin
  updates that don't change behavior.

Pre-release tags (`X.Y.ZrcN`, `X.Y.Z.devN`) ship via the same
workflow and live on PyPI as pre-releases, which `pip install`
will only pick up with `--pre`.

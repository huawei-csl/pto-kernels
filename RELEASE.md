# Release Process

## Prerequisites
- [ ] Make sure that unit tests pass on both 910B4 and 910B2 (check internal CI on pto-kernels-mirror)
- [ ] Make sure that GitHub CI workflow is all green
- [ ] Run the tests once more locally (910B2 or 910B4)

## Release Steps
1. **Commit changes**:Update the version in pyproject.toml (say 0.1.2 -> 0.1.3) and all other files (use `git grep 0.1.2`)
2. **Tag release**: Create a new git tag using (important v on tag name): git tag -a vX.Y.Z -m "Release version vX.Y.Z"
3. **Push tags**: Push the git tag to the repo: git push origin vX.Y.Z.
4. **Create GitHub Release**: (use link https://github.com/huawei-csl/pto-kernels/releases)
   - Go to Releases > Draft a new release
   - Select tag `vX.Y.Z`
   - Generate release notes automatically
   - Click "Publish release"

## Post-Release
- [ ] Verify build artifacts in CI/CD pipeline
- [ ] Close related issues/milestones

# Releasing & Versioning

We use `mike` to manage versioned documentation and `git-cliff` for changelog generation.

## Documentation Versioning

To deploy a new version of the documentation:

```bash
# Deploy a new version
mike deploy v0.3.0 latest --update-aliases

# Set default version
mike set-default latest
```

## Changelog Generation

The `CHANGELOG.md` is automatically updated using `git-cliff` based on conventional commits.

To generate it manually:
```bash
git-cliff -o CHANGELOG.md
```

## Release Workflow

1. Update version in `pyproject.toml`.
2. Generate changelog.
3. Commit and tag.
4. Push tags to GitHub.
5. CI will automatisch build and deploy docs for the new tag.

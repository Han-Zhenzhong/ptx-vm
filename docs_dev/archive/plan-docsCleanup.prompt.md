## Plan: Documentation Cleanup & Reorg

Make the docs navigable and consistent by fixing the broken doc-path assumptions (`user_docs/dev_docs/spec_docs/docs` vs current `docs_user/docs_dev/docs_spec`), establishing one canonical docs entry point, and separating evergreen guides from historical “work log” reports. This keeps links stable, reduces duplication, and makes onboarding/usage docs easy to find.

### Steps 5
1. Choose canonical doc root strategy: “update links to current names” (see considerations).
2. Fix top-level navigation links in [README.md](README.md), [DOCS_INDEX.md](DOCS_INDEX.md), and [DOCS_REORGANIZATION.md](DOCS_REORGANIZATION.md) to match reality.
3. Repair cross-links inside doc indexes: [docs_user/README.md](docs_user/README.md), [docs_dev/README.md](docs_dev/README.md), [docs_spec/README.md](docs_spec/README.md), plus any references like [docs_user/cli_usage_correction.md](docs_user/cli_usage_correction.md).
4. Consolidate and archive “work log” docs: move `*_SUMMARY*`, `*REPORT*`, `*AUDIT*`, `*FIX*` from docs_dev into a dedicated archive area and keep curated guides prominent (e.g., [docs_dev/developer_guide.md](docs_dev/developer_guide.md)).
5. Remove or formalize empty/placeholder paths (e.g., [docs_user/technical_ref](docs_user/technical_ref) and [docs_spec/technical_ref](docs_spec/technical_ref)) by adding index pages or deleting unused stubs.

### Further Considerations 3
1. Canonical strategy: update all links to current `docs_*` names.
2. eliminate all `docs/` mentions.
3. Convert non-portable local links (e.g., in [blog/how-to-generate-ptx-from-cuda-c.md](blog/how-to-generate-ptx-from-cuda-c.md)) to repo-relative links.


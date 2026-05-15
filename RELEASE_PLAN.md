# Anonymous and Public Release Plan

## During ARR Review

Use two anonymized artifacts:

1. `crossalpha_arr_anonymous_code.zip`
   - Main experiment scripts.
   - Backtesting utilities.
   - Similarity and return kernels.
   - AR-Scraper source code, excluding downloaded report corpora.
   - Reviewer README and anonymization notes.

2. `crossalpha_arr_anonymous_data.zip`
   - Metadata dictionaries.
   - Paper result JSON files.
   - A data manifest describing the full release contents.

Recommended upload path:

- Primary: OpenReview supplementary material, with software and data as separate zip files.
- Optional code mirror: Anonymous GitHub, only if the submission form allows a repository link.
- Avoid Google Drive, Dropbox, or other links that can expose viewer/download identity.

## After Acceptance

Publish the non-anonymous release:

- Code: GitHub under the project account.
- Dataset: Hugging Face Datasets for discoverability and programmatic loading.
- Archival DOI: Zenodo release linked to the GitHub repository and/or dataset card.

## Anonymization Rules

- No author names, usernames, emails, local paths, Overleaf project URLs, or personal repository links.
- No `.env`, API keys, logs, `.git`, cache directories, notebooks with execution metadata, or local PDFs.
- Paths in scripts should use `CROSSALPHA_DATA_ROOT` and `CROSSALPHA_PROJECT_ROOT`, not user-specific absolute paths.
- Public URLs should be replaced by "anonymized supplementary link" during review.


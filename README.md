# Chris Merck — Tutorials & Research

This repository hosts the source for my personal site 

## Quickstart

```bash
# 1. Install deps (ideally inside a virtualenv)
pip install -r requirements.txt

# 2. Serve locally with hot-reload
mkdocs serve

# 3. Build static site to the `site/` folder
mkdocs build
```

## Continuous Deployment (optional)

The static output in `site/` can be published automatically with GitHub Pages. Add a workflow (not included here) or configure Pages to deploy the `gh-pages` branch built by CI.

### Deploying with GitHub Pages

1. **Create a GitHub workflow file**:
   Create a file at `.github/workflows/deploy.yml` with the following content:

   ```yaml
   name: Deploy MkDocs

   on:
     push:
       branches:
         - main

   jobs:
     deploy:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v3
         - uses: actions/setup-python@v4
           with:
             python-version: 3.x
         - run: pip install -r requirements.txt
         - run: mkdocs gh-deploy --force
   ```

2. **Configure GitHub Pages**:
   - Go to your repository settings
   - Navigate to "Pages"
   - Set "Source" to "GitHub Actions"

3. **Push to main branch**:
   - The workflow will automatically build and deploy your site to the `gh-pages` branch
   - Your site will be available at `https://<username>.github.io/<repository-name>/`

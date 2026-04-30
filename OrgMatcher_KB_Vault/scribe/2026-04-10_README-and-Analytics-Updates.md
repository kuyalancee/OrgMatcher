# Session Summary â€” 2026-04-10 (README & Analytics Updates)

## Documentation & Repository Polish
- **README.md Overhaul**: Completely updated the `README.md` to reflect the project's evolution from a terminal script to a full-stack application.
- **New Sections Added**: Included a comprehensive "Tech Stack" overview, a clear step-by-step "Methodology" section detailing the NLP workflow (Scraping, Lemmatization, TF-IDF, Cosine Similarity), and updated instructions for running the split frontend/backend architecture locally.
- **Visual Improvements**: 
  - Added dynamic Shields.io status badges (Website, Frontend, Backend) centered at the top of the README.
  - Centered all usage demonstration images and scaled them down to 75% size using HTML tags for a cleaner aesthetic.

## Infrastructure & Analytics
- **CI/CD Discussion**: Clarified deployment triggers: pushing to `main` will automatically trigger Vercel to build and deploy the frontend, while the GCP Cloud Run backend currently requires manual deployment as no automated pipelines (like GitHub Actions) are configured.
- **Vercel Analytics Integration**: Added the `@vercel/analytics` `<Analytics />` component to the frontend `App.jsx` to begin tracking user traffic, page views, and geographic data directly from the Vercel Dashboard. The npm package `@vercel/analytics` needs to be installed locally before the next commit.
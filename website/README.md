# OpenDrug — Project Homepage

A static, dependency-free website introducing the **OpenDrug** benchmark:
a unified, multimodal evaluation framework for drug-related interaction
prediction (DDI, DTI, PPI).

## Structure

```
website/
├── index.html              # Page markup
└── assets/
    ├── css/style.css       # Styles (deep navy + cyan accent)
    ├── js/main.js          # Hero animation, counters, dataset filter, BibTeX copy
    ├── figures/            # Paper figures (framework, benchmark, robustness, ...)
    └── img/favicon.svg
```

## Deploy to GitHub Pages

1. Push this `website/` folder to a GitHub repository (or the repository root
   if you want to serve from `/`).
2. In **Settings → Pages**, choose the branch that contains this folder and
   set the directory to `/website` (or `/` if the site lives at the repo root).
3. Visit the published URL — no build step required.

## Local preview

Open `website/index.html` directly in a browser, or run a tiny static server:

```bash
cd website
python -m http.server 8000
# then open http://localhost:8000
```

## Content overview

| Section | Purpose |
| --- | --- |
| Hero | Animated heterogeneous graph + headline metrics (18 / 30 / 7 / 8 / 5). |
| Overview | 5+3 modality alignment, three-stage pipeline, tasks & stress tests. |
| Datasets | Filterable / searchable table of all 18 datasets. |
| Benchmark | Per-task leaderboard snapshot + main performance figure. |
| Robustness | Headline numbers (−25 %, +5-20 %) + stress test figure. |
| Mechanism | Three case studies (CYP3A4, PDE5/PDE6, Hsp90-Cdc37-CDK4). |
| Cognitive boundaries | Thalidomide, BCR-ABL T315I, p53-MDM2 failures. |
| Framework | Paper Figure 1. |
| Team | Author placeholder cards. |
| Citation | Copy-to-clipboard BibTeX. |

## Notes

- Numbers, datasets and case studies are sourced from the manuscript
  (`intro.tex`, `methods.tex`, `Dataset Details.tex`, `result.tex`).
- Paper / code / data links are intentionally marked **coming soon** until
  the publication venue is confirmed. Replace the BibTeX block in
  `index.html` once the arXiv ID is assigned.
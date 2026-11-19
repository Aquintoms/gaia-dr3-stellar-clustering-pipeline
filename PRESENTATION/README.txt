Gaia DR3 Stellar Clustering - Beamer presentation source

Files
- Gaia_DR3_Stellar_Clustering_Presentation.tex : complete Beamer source
- references.bib : bibliography entries for author-year citations
- assets/image1.png : implemented project architecture figure, extracted from the supplied report
- assets/image6.png : HDBSCAN UMAP cluster figure, extracted from the supplied report
- assets/SU_Logo.png : official Strathmore University logo supplied by the student
- assets/opo9920a.jpg : user-supplied illustrative astronomy photograph for the cover (explicitly not a Gaia model output)

Two PDF variants are supplied separately:
1) Reading copy: each slide is followed by its speaker-notes page.
2) Pympress copy: speaker notes are placed on the right-hand screen.

Rebuild (requires a TeX distribution with Beamer, biblatex-apa, Biber and TikZ):

Reading copy:
  pdflatex -jobname=Gaia_DR3_Stellar_Clustering_Reading Gaia_DR3_Stellar_Clustering_Presentation.tex
  biber Gaia_DR3_Stellar_Clustering_Reading
  pdflatex -jobname=Gaia_DR3_Stellar_Clustering_Reading Gaia_DR3_Stellar_Clustering_Presentation.tex
  pdflatex -jobname=Gaia_DR3_Stellar_Clustering_Reading Gaia_DR3_Stellar_Clustering_Presentation.tex

Pympress copy (PowerShell):
  pdflatex -jobname=Gaia_DR3_Stellar_Clustering_Pympress "\def\PympressMode{1}\input{Gaia_DR3_Stellar_Clustering_Presentation.tex}"
  biber Gaia_DR3_Stellar_Clustering_Pympress
  pdflatex -jobname=Gaia_DR3_Stellar_Clustering_Pympress "\def\PympressMode{1}\input{Gaia_DR3_Stellar_Clustering_Presentation.tex}"
  pdflatex -jobname=Gaia_DR3_Stellar_Clustering_Pympress "\def\PympressMode{1}\input{Gaia_DR3_Stellar_Clustering_Presentation.tex}"

The cover photograph is used only as illustration and is not represented as a Gaia DR3 result.
The actual cluster plot and architecture diagram are taken from the supplied final project report.

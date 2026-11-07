Individual downstream panels for the CVPR draft

plot_downstream.py and plot_downstream.ipynb export twelve separate PDFs to
figures/individual/. Every PDF is exactly 1.65 x 1.35 inches. Filenames and PDF
metadata identify both the pretraining source and downstream target. Only the
downstream dataset name appears as the plot title. There are no pretraining
titles, legends, or y-axis labels. Visible x ticks mark 0.2 and 1 within each
label-budget group; these are percentages of the pretraining dataset.

The left pair uses 5% labeled data and the left y-axis; the right pair uses 1%
and the right y-axis. The logarithmic limits are independent and include every
SD whisker with a small margin. Heights across the two budgets use different
scales. Colors and filled circle/square/triangle/diamond markers are preserved.

The first code cell contains editable settings. The Helvetica-compatible
Liberation Sans font is installed in the current environment. Numeric ticks and
label-budget headers are 8 pt; downstream titles are 9 pt. The compact headers
above the two groups read 5% and 1%; the figure caption defines these as labeled
data budgets. PDFs are saved
without tight cropping so every canvas has identical dimensions.

figures.tex contains one figure* with three rows of four plots: CIFAR-10,
CIFAR-100, and Tiny ImageNet pretraining. Each group has a caption, while the
method legend appears only in the main figure caption. The PDFs are included at
their native 1.65-inch width within the CVPR 6.875-inch text width. Changing that
inclusion width will also change the effective font sizes.

cvpr_preview.tex uses the bundled cvpr.sty, copied from the existing draft preview. The
compiled full document is cvpr_test/cvpr_preview.pdf; cvpr_test/cvpr_layout.pdf
contains the figure page with surrounding two-column body text. render.sh checks
for overfull content, confirms 8/9 pt panel fonts in the compiled page, verifies
all twelve PDF dimensions, and executes the notebook code cells to check that
they produce byte-identical PDFs to the Python script.

Run with the existing environment inside a CPU allocation on Vulcan:
  source /home/mmoslem3/ENV/bin/activate
  python PLOTS/downstream/plot_downstream.py
Or run sbatch PLOTS/downstream/render.sh from the repository root to rebuild everything and run the
CVPR checks. render.sh regenerates the notebook from the Python source; preserve
any edits made only in the notebook before using that rebuild command.
The notebook can run from its directory or the repository root. It needs the
same Matplotlib environment and figures/downstream_results.csv.

The data file retains the two-decimal values reconstructed from 1,269 completed
probe logs and checked against all 216 entries in cifar_downstream.tex. Full
precision values, actual replicate counts, and original table values remain in
the CSV; figures/data_provenance.json records the individual log paths and
hashes. No scores are imputed and no experiments are rerun by this exporter.

The source table mixes these replicate designs: full-data uses 3 encoder seeds;
CIFAR Random uses 9 or 10 saved seeds; CIFAR K-means uses 6; MKDT uses 2 encoder
seeds with 3 probes each; MIPA uses 1 encoder with 5 probes. Tiny ImageNet Random
and K-means use 5 probes, except the Random 0.2% / Aircraft / 1% labeled cell,
which retains 10 older runs. Whiskers are sample SD (ddof=1), not confidence
intervals. The layout preserves these source-table choices.

CVPR typography/layout guidance:
https://github.com/cvpr-org/author-kit/blob/main/sec/2_formatting.tex

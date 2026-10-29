from pathlib import Path
import json,numpy as np
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from matplotlib.font_manager import findfont, FontProperties
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
pdfmetrics.registerFont(TTFont("DejaVu", findfont(FontProperties(family="DejaVu Sans"))))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", findfont(FontProperties(family="DejaVu Sans",weight="bold"))))
r=json.loads(Path('research/results/benchmark.json').read_text());styles=getSampleStyleSheet();styles['BodyText'].fontSize=10;styles['BodyText'].leading=14;styles['Title'].textColor=colors.HexColor('#086c5b')

for name in ['BodyText','Normal']: styles[name].fontName='DejaVu'
for name in ['Title','Heading1','Heading2']: styles[name].fontName='DejaVu-Bold'
styles['BodyText'].fontSize=9;styles['BodyText'].leading=12
story=[]
def para(t,style='BodyText'):story.append(Paragraph(t,styles[style]));story.append(Spacer(1,6))
def table(data,widths):
 t=Table(data,colWidths=widths);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dcebdd')),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#bbcbbb')),('FONTNAME',(0,0),(-1,0),'DejaVu-Bold'),('FONTNAME',(0,1),(-1,-1),'DejaVu'),('FONTSIZE',(0,0),(-1,-1),9),('BOTTOMPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),8)]));story.append(t);story.append(Spacer(1,12))
para('QOLDA Vision','Title');para('Subject-disjoint hand posture learning and an instrumented motion-analysis prototype','Heading2');para('Technical report | 29 October 2026 | Reproducible exploratory benchmark')
para('1. Question and scope','Heading2');para('Can a small trained MLP outperform simple classifiers on hand-posture records from an unseen user identifier? We compare models on public motion-capture records and provide separate tools for future webcam and glove measurements. The public benchmark does not measure MediaPipe accuracy, rehabilitation benefit or QOLDA hardware agreement.')
para('2. Data and provenance','Heading2');para('The UCI Motion Capture Hand Postures dataset (Gardner, Selmic and Kanno; DOI 10.24432/C5TG86; CC BY 4.0) contains 5 gesture labels and unordered visible marker coordinates. Missing markers occur. The source describes 12 users; the downloaded CSV has 14 unique user IDs. This metadata discrepancy remains unresolved: our folds hold out IDs, and we do not assert 14 verified people.')
para(f"After removing the dummy row, the file contains {r['original_non_dummy_rows']:,} records. We sample up to 75 records per class per user ID without replacement (seed 42), yielding {r['sampled_rows']:,} observations. Selected row indices and the original-file SHA-256 are retained. Some user IDs lack classes; unavailable open-set metrics are null rather than fabricated.")
para('3. Method','Heading2');para('Marker order is not a finger identity. We therefore use 18 permutation-invariant descriptors: visible count, coordinate means and standard deviations, radial quantiles, pair-distance quantiles and covariance eigenvalues. Standardization is fitted on training IDs only. In each of 14 folds, one ID is test, the next ID is validation, and the other IDs are training. All models use the same split.')
para('Models are nearest centroid, unweighted kNN (k=5), and an 18-64-32-5 ReLU MLP trained in PyTorch for 50 epochs using Adam (learning rate 0.001, weight decay 0.0001, batch size 128). Hyperparameters are fixed. Validation is reserved for open-set threshold calibration; it is not used to select MLP epochs. Fold checkpoints and predictions are included.')
story.append(PageBreak())
para('4. Measured benchmark results','Heading1');para('Accuracy is averaged equally across held-out IDs. SD describes between-ID variation, not a confidence interval. Macro F1 uses the fixed five-class label set within each fold, including absent classes.')
data=[['Model, full features','Accuracy mean +/- SD','Macro F1']]
for s in r['summary'][:3]:data.append([s['model'],f"{s['accuracy_mean']*100:.2f} +/- {s['accuracy_sd']*100:.2f}%",f"{s['macro_f1_mean']:.3f}"])
table(data,[165,180,95]);story.append(Image('docs/assets/ablation.png',width=440,height=220));story.append(Spacer(1,12))
para('The full MLP reaches 86.83% mean held-ID accuracy versus 80.21% for kNN and 75.53% for centroid. This is a single-seed exploratory result on public Vicon marker data. It cannot be transferred to webcam input: the representation and sensor are different.')
para('Removing marker count reduces MLP accuracy to 80.09%; removing pair-distance features gives 87.25%. More features do not consistently improve performance. Marker count may encode acquisition or occlusion effects; its apparent benefit warrants further testing. No ablation variant is selected and presented as an independently validated best model.')
para('5. Unknown-class experiment','Heading2');para('Class 5 (Grab) is excluded from training and threshold calibration. A distance threshold is the 95th percentile of mean k-neighbor distance for known validation examples. Unknown recall averages 57.44% over IDs with unknown examples; known false rejection averages 11.25% across all IDs. Unknown-detection AUROC averages 0.937 across IDs with both known and unknown examples. Threshold performance differs across users and this evaluates one withheld class, not arbitrary unseen gestures.')
story.append(PageBreak())
para('6. Reliability and movement instrumentation','Heading1');story.append(Image('docs/assets/robustness.png',width=440,height=220));story.append(Spacer(1,10))
para('The plot removes one or two randomly selected visible markers, retaining at least three. These are synthetic occlusion perturbations of real records, not measured lighting effects or MediaPipe failure rates. Real camera capture records every inference attempt, including missing-hand frames, metadata for lighting/view/hand/distance and both screen/world landmarks.')
para('Motion analysis computes a 95th-to-5th-percentile amplitude, mean absolute speed, RMS jerk after specified resampling/smoothing, cycle-period variation and peak-amplitude variability. Static-hold frame displacement provides a jitter descriptor; it includes actual movement. Long tracking gaps invalidate movement analysis rather than being interpolated away. These descriptors are not validated clinical scores.')
para('7. Camera-glove study: implemented tools, pending data','Heading2');para('The browser and serial recorder timestamp observations on the same host epoch clock. A comparison script performs nearest-time pairing within 50 ms, without sample reuse, then reports normalized-proxy MAE, bias, RMSE and correlation. This is receipt-time alignment, not hardware synchronization. Sensor latency, clock adjustment and anatomical correspondence require measurement. Degree-level error needs physical joint-specific calibration and is not reported here.')
para('8. Reproduction and next evidence','Heading2');para('Run research/benchmark.py, then research/plot_results.py. QOLDA_Research_Colab.ipynb embeds the benchmark and downloads the public dataset when Run all is selected. Per-fold predictions, sampled row indices, file hash, split manifest, environment versions and MLP weights accompany this report. Browser code is checked separately from scientific performance.')
para('Remaining evidence: actual QOLDA glove recordings; independently recruited, consenting webcam participants; world-versus-screen and calibration ablations on those recordings; real failure rates by lighting/view; repeat seeds and uncertainty analysis. A camera-benchmark runner is supplied for these future data. No patient study, clinical deployment or therapeutic outcome is claimed.')
para('Reference: Gardner, A., Selmic, R., and Kanno, J. (2014). Motion Capture Hand Postures. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG86. Dataset accessed 29 October 2026.')
def footer(c,d):c.setFont('DejaVu',7);c.setFillColor(colors.HexColor('#68796b'));c.drawString(42,26,'QOLDA Vision | public-data benchmark; experimental prototype');c.drawRightString(553,26,str(d.page))
SimpleDocTemplate('docs/QOLDA_Research_Report.pdf',pagesize=(595,842),rightMargin=48,leftMargin=48,topMargin=42,bottomMargin=45).build(story,onFirstPage=footer,onLaterPages=footer)

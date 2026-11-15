"""Build the Team 9 results PDF from the same CSVs the dashboard page uses."""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]
DATA = sys.argv[2] if len(sys.argv) > 2 else HERE
CHARTS = os.path.join(HERE, "charts")
os.makedirs(CHARTS, exist_ok=True)


def load(name):
    with open(os.path.join(DATA, f"{name}.csv"), newline="") as f:
        return list(csv.DictReader(f))


runs, steps, nccl, loadm = load("runs"), load("steps"), load("nccl"), load("load")
EV = {r["metric"]: r for r in load("eval_scores")}
ev = lambda m, k: float(EV[m][k])
RUN_IDS = ["single-1n", "ddp-2n", "ep-2n"]
LABEL = {"single-1n": "1 node", "ddp-2n": "2 nodes · data parallel", "ep-2n": "2 nodes · expert parallel"}
COL = {"single-1n": "#6b7280", "ddp-2n": "#2563eb", "ep-2n": "#d97706"}
R = {r["run"]: r for r in runs}
base_tps = float(R["single-1n"]["median_tps"])
ddp_mem = float(R["ddp-2n"]["peak_mem_gib"])
SC = {}
for r in runs:
    sp = float(r["median_tps"]) / base_tps
    SC[r["run"]] = {"speedup": sp, "eff": sp / float(r["nodes"]) * 100,
                    "memsave": (1 - float(r["peak_mem_gib"]) / ddp_mem) * 100}
LOAD = {r["method"]: r for r in loadm}
fast = next(r for r in loadm if r["method"].startswith("Fastload"))
orig = next(r for r in loadm if r["method"].startswith("Trainer mmap"))
bw1g = next(float(r["busbw_gbs"]) for r in nccl if r["size_mb"] == "1024")

# ---------- charts ----------
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e5e7eb", "axes.axisbelow": True})


def save(fig, name):
    p = os.path.join(CHARTS, name)
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def line_chart(field, ylabel, name, ymin0=True):
    fig, ax = plt.subplots(figsize=(7, 2.45))
    for run in RUN_IDS:
        pts = sorted((int(r["step"]), float(r[field])) for r in steps if r["run"] == run)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=2.5, lw=1.8, color=COL[run], label=LABEL[run])
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(0, 20, 2))
    if ymin0:
        ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8, loc="best")
    return save(fig, name)


tps_png = line_chart("tps", "Tokens / s", "tps.png")
loss_png = line_chart("loss", "Training loss", "loss.png")

fig, ax = plt.subplots(figsize=(3.4, 2.35))
vals = [float(R[r]["peak_mem_gib"]) for r in RUN_IDS]
bars = ax.bar([LABEL[r].replace(" · ", "\n") for r in RUN_IDS], vals, color=[COL[r] for r in RUN_IDS], width=0.6)
ax.bar_label(bars, labels=[f"{v:.1f}" for v in vals], padding=2, fontsize=8)
ax.set_ylabel("Peak memory per node (GiB)")
ax.set_ylim(0, max(vals) * 1.18)
ax.tick_params(axis="x", labelsize=7.5)
mem_png = save(fig, "mem.png")

fig, ax = plt.subplots(figsize=(3.4, 2.35))
vals = [float(R[r]["median_tps"]) for r in RUN_IDS]
bars = ax.bar([LABEL[r].replace(" · ", "\n") for r in RUN_IDS], vals, color=[COL[r] for r in RUN_IDS], width=0.6)
ax.bar_label(bars, labels=[f"{v:,.0f}\n({SC[r]['speedup']:.2f}×)" for r, v in zip(RUN_IDS, vals)], padding=2, fontsize=7.5)
ax.set_ylabel("Median tokens / s")
ax.set_ylim(0, max(vals) * 1.3)
ax.tick_params(axis="x", labelsize=7.5)
thr_png = save(fig, "thr.png")

fig, ax = plt.subplots(figsize=(7, 2.3))
names = [r["method"] for r in loadm][::-1]
mins = [float(r["load_min"]) for r in loadm][::-1]
cols = ["#dc2626" if "OOM" in r["result"] or "killed" in r["result"] else "#2563eb" for r in loadm][::-1]
b = ax.barh(names, mins, color=cols, height=0.6)
ax.bar_label(b, labels=[f"{m:g} min" for m in mins], padding=3, fontsize=8)
ax.set_xlabel("Minutes to load the model")
ax.set_xlim(0, max(mins) * 1.18)
ax.grid(axis="y", visible=False)
load_png = save(fig, "load.png")

fig, ax = plt.subplots(figsize=(7, 2.2))
sizes = [r["size_mb"] for r in nccl]
bw = [float(r["busbw_gbs"]) for r in nccl]
b = ax.bar([f"{s} MB" for s in sizes], bw, color="#2563eb", width=0.55)
ax.bar_label(b, labels=[f"{v:.1f}" for v in bw], padding=2, fontsize=8)
ax.set_ylabel("Bus bandwidth (GB/s)")
ax.set_xlabel("All-reduce message size")
ax.set_ylim(0, max(bw) * 1.2)
nccl_png = save(fig, "nccl.png")

EV_ORDER = ["class_accuracy", "class_macro_f1", "severity_accuracy", "quote_fidelity", "fully_valid_rate", "json_valid_rate"]
fig, ax = plt.subplots(figsize=(7, 3.0))
import numpy as np
yy = np.arange(len(EV_ORDER))[::-1]
bv = [ev(m, "base") * 100 for m in EV_ORDER]; fv = [ev(m, "finetuned") * 100 for m in EV_ORDER]
b1 = ax.barh(yy + 0.19, bv, height=0.36, color="#9ca3af", label="Base model")
b2 = ax.barh(yy - 0.19, fv, height=0.36, color="#2563eb", label="Fine-tuned (LoRA, step 91)")
ax.bar_label(b1, labels=[f"{v:.0f}" for v in bv], padding=3, fontsize=7.5, color="#4b5563")
ax.bar_label(b2, labels=[f"{v:.0f}" for v in fv], padding=3, fontsize=7.5, color="#1d4ed8", fontweight="bold")
ax.set_yticks(yy); ax.set_yticklabels([EV[m]["label"].replace("Risk class macro-F1", "Risk class macro-F1 (×100)") for m in EV_ORDER], fontsize=8)
ax.set_xlim(0, 112); ax.set_xlabel("% of 152 held-out passages"); ax.grid(axis="y", visible=False)
ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
eval_png = save(fig, "eval.png")

# ---------- document ----------
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=18, leading=22, alignment=TA_LEFT, spaceAfter=4)
SUB = ParagraphStyle("SUB", parent=ss["Normal"], fontSize=8.5, leading=11, textColor=colors.HexColor("#6b7280"))
LEDE = ParagraphStyle("LEDE", parent=ss["Normal"], fontSize=11, leading=15.5, spaceBefore=8, spaceAfter=10)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=12.5, leading=16, spaceBefore=12, spaceAfter=3)
BODY = ParagraphStyle("BODY", parent=ss["Normal"], fontSize=9.5, leading=13, spaceAfter=6)
SMALL = ParagraphStyle("SMALL", parent=BODY, fontSize=8, leading=10.5, textColor=colors.HexColor("#374151"))
CELL = ParagraphStyle("CELL", parent=ss["Normal"], fontSize=8, leading=10)
CARD_L = ParagraphStyle("CL", parent=ss["Normal"], fontSize=7.5, leading=9.5, textColor=colors.HexColor("#6b7280"))
CARD_B = ParagraphStyle("CB", parent=ss["Normal"], fontName="Helvetica-Bold", fontSize=17, leading=20)
CARD_N = ParagraphStyle("CN", parent=ss["Normal"], fontSize=7.5, leading=9.5, textColor=colors.HexColor("#374151"))

W = letter[0] - 1.5 * inch
story = []
story.append(Paragraph("Fine-tuning Nemotron-3-Nano 30B on one vs two GB10 nodes", H1))
story.append(Paragraph("Team 9 · CIQ fb-hack cluster (Fuzzball, NVIDIA GB10 nodes) · 14–15 Nov 2026 · LoRA fine-tune, 20 steps, "
                       "global batch 16, sequence length 3072, identical data and settings in every run", SUB))
story.append(Paragraph(
    f"Two nodes trained <b>{SC['ddp-2n']['speedup']:.2f}×</b> faster than one, at <b>{SC['ddp-2n']['eff']:.1f}%</b> scaling "
    f"efficiency. Splitting the model's experts across the two nodes kept almost that speed "
    f"(<b>{SC['ep-2n']['speedup']:.2f}×</b>) with <b>{SC['ep-2n']['memsave']:.0f}%</b> less memory per node, and all three "
    f"runs learned the same. On 152 passages from companies it never saw, the fine-tuned model got the risk class right "
    f"<b>{ev('class_accuracy','finetuned')*100:.0f}%</b> of the time (base {ev('class_accuracy','base')*100:.0f}%) and the severity "
    f"<b>{ev('severity_accuracy','finetuned')*100:.0f}%</b> (base {ev('severity_accuracy','base')*100:.0f}%).", LEDE))


def card(label, big, note):
    return [Paragraph(label, CARD_L), Paragraph(big, CARD_B), Paragraph(note, CARD_N)]


cards = [
    card("Speedup with 2 nodes (data parallel)", f"{SC['ddp-2n']['speedup']:.2f}×",
         f"{float(R['ddp-2n']['median_tps']):,.0f} vs {base_tps:,.0f} tokens/s on 1 node"),
    card("Peak memory per node, expert parallel", f"{float(R['ep-2n']['peak_mem_gib']):.1f} GiB",
         f"{SC['ep-2n']['memsave']:.0f}% less than data parallel ({ddp_mem:.1f} GiB)"),
    card("Time to load the 59 GB model", f"{float(fast['load_min']):g} min",
         f"was {float(orig['load_min']):g} min reading straight from shared storage"),
    card("Node-to-node bandwidth (RDMA)", f"{bw1g:.1f} GB/s", "NCCL all-reduce, 1 GB messages"),
]
ct = Table([cards], colWidths=[W / 4] * 4)
ct.setStyle(TableStyle([
    ("BOX", (0, 0), (0, 0), 0.6, colors.HexColor("#d1d5db")), ("BOX", (1, 0), (1, 0), 0.6, colors.HexColor("#d1d5db")),
    ("BOX", (2, 0), (2, 0), 0.6, colors.HexColor("#d1d5db")), ("BOX", (3, 0), (3, 0), 0.6, colors.HexColor("#d1d5db")),
    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f9fafb")),
    ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
    ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
]))
story.append(ct)


def img(path, width):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(path).getSize()
    return Image(path, width=width, height=width * ih / iw)


story.append(KeepTogether([
    Paragraph("Two nodes process about twice the tokens per step", H2),
    Paragraph("Tokens processed per second at each training step. Step 0 is slow in every run (one-time warm-up), so "
              "speedups use the median step.", BODY),
    img(tps_png, W),
]))

two = Table([[img(thr_png, W / 2 - 6), img(mem_png, W / 2 - 6)]], colWidths=[W / 2, W / 2])
two.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
story.append(KeepTogether([
    Paragraph("Expert parallel matches the speed with far less memory", H2),
    Paragraph("Left: median tokens/s and speedup over 1 node. Right: peak memory per node. Data parallel keeps a full "
              "model copy on each node; expert parallel keeps half of the experts on each.", BODY),
    two,
]))

story.append(KeepTogether([
    Paragraph("All three runs learn the same", H2),
    Paragraph("Training loss per step. The curves overlap: splitting the batch or the experts across nodes doesn't "
              "change what the model learns.", BODY),
    img(loss_png, W),
]))

# Run summary table
hdr = ["Run", "Strategy", "Nodes", "Step (s)", "Tokens/s", "Speedup", "Effic. %", "Peak mem\n(GiB)", "Final\nloss", "Val\nloss", "Run time\n(min)"]
rows = [hdr]
for rid in RUN_IDS:
    r = R[rid]
    rows.append([rid, Paragraph(r["strategy"], CELL), r["nodes"], f"{float(r['median_step_s']):.1f}", f"{float(r['median_tps']):,.0f}",
                 f"{SC[rid]['speedup']:.2f}×", f"{SC[rid]['eff']:.1f}", f"{float(r['peak_mem_gib']):.1f}",
                 f"{float(r['last_loss']):.3f}", f"{float(r['val_loss']):.3f}", f"{float(r['wall_s']) / 60:.0f}"])
cw = [0.72, 1.25, 0.42, 0.55, 0.62, 0.58, 0.55, 0.62, 0.5, 0.5, 0.64]
scale = W / (sum(cw) * inch)
t = Table(rows, colWidths=[c * inch * scale for c in cw], repeatRows=1)
t.setStyle(TableStyle([
    ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.5), ("FONT", (0, 1), (-1, -1), "Helvetica", 8),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#374151")),
    ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor("#9ca3af")), ("LINEBELOW", (0, 1), (-1, -1), 0.4, colors.HexColor("#e5e7eb")),
    ("ALIGN", (2, 0), (-1, -1), "RIGHT"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
story.append(KeepTogether([
    Paragraph("Run summary", H2),
    Paragraph("Speedup is median tokens/s relative to the 1-node run; scaling efficiency is speedup ÷ number of nodes. "
              "Validation loss is measured on held-out data after step 19. Run time is the training script's wall time "
              "including model loading; the 1-node run predates the loading fix (49-minute load).", BODY),
    t,
]))

erows = [["Metric (152 held-out passages)", "Base model", "Fine-tuned", "Change"]]
for m in EV_ORDER + ["median_latency_s"]:
    b, f, kind = ev(m, "base"), ev(m, "finetuned"), EV[m]["kind"]
    if kind == "sec":
        cells = [f"{b:.1f} s", f"{f:.1f} s", f"{(1 - f / b) * 100:.0f}% faster"]
    elif kind == "f1":
        cells = [f"{b:.3f}", f"{f:.3f}", f"+{f - b:.3f}"]
    else:
        r1 = lambda x: f"{round(x * 1000 + 1e-6) / 10:.1f}"  # half-up, so 0.9605 shows as 96.1
        cells = [r1(b) + "%", r1(f) + "%", f"{(f - b) * 100:+.1f} pts"]
    erows.append([EV[m]["label"]] + cells)
et = Table(erows, colWidths=[W * 0.43, W * 0.17, W * 0.17, W * 0.23], repeatRows=1)
et.setStyle(TableStyle([
    ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.5), ("FONT", (0, 1), (-1, -1), "Helvetica", 8.5),
    ("FONT", (2, 1), (2, -1), "Helvetica-Bold", 8.5), ("TEXTCOLOR", (2, 1), (2, -1), colors.HexColor("#1d4ed8")),
    ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor("#9ca3af")), ("LINEBELOW", (0, 1), (-1, -1), 0.4, colors.HexColor("#e5e7eb")),
    ("ALIGN", (1, 0), (-1, -1), "RIGHT"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
story.append(KeepTogether([
    Paragraph("Did fine-tuning help? Yes, on every measure", H2),
    Paragraph(f"Both models answered the same 152 risk-factor passages from MS, GOOGL, PFE and WMT 10-K filings, companies held "
              f"out of training. For each passage the model must return JSON with the most material risk class, its severity "
              f"and a quote copied word for word from the passage. Fine-tuning doubled severity accuracy "
              f"({ev('severity_accuracy','base')*100:.0f}% to {ev('severity_accuracy','finetuned')*100:.0f}%), lifted risk-class accuracy "
              f"by {(ev('class_accuracy','finetuned')-ev('class_accuracy','base'))*100:.0f} points, kept quotes faithful, and answered "
              f"{(1-ev('median_latency_s','finetuned')/ev('median_latency_s','base'))*100:.0f}% faster because its answers are short "
              f"and in format. Scores are from scripts/08_score.py against the teacher model's labels.", BODY),
    img(eval_png, W), Spacer(1, 6), et,
]))

lrows = [["Approach", "Load (min)", "Result", "Why"]]
for r in loadm:
    lrows.append([Paragraph(r["method"], CELL), f"{float(r['load_min']):g}", Paragraph(r["result"], CELL), Paragraph(r["detail"], CELL)])
lt = Table(lrows, colWidths=[W * 0.27, W * 0.1, W * 0.17, W * 0.46], repeatRows=1)
lt.setStyle(TableStyle([
    ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.5), ("FONT", (0, 1), (-1, -1), "Helvetica", 8),
    ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor("#9ca3af")), ("LINEBELOW", (0, 1), (-1, -1), 0.4, colors.HexColor("#e5e7eb")),
    ("ALIGN", (1, 0), (1, -1), "RIGHT"), ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
story.append(KeepTogether([
    Paragraph("Fixing model loading mattered as much as the second node", H2),
    Paragraph(f"The trainer read the model from shared network storage in small pieces, one round trip at a time. Copying "
              f"one ~5 GB file at a time into memory with a single sequential read, then loading from that copy, cut the "
              f"load from {float(orig['load_min']):g} minutes to about {float(fast['load_min']):g}. Copying the whole model "
              f"into memory at once crashed the trainer (red), because on the GB10 that memory is also the GPU's.", BODY),
    img(load_png, W), Spacer(1, 6), lt,
]))

story.append(KeepTogether([
    Paragraph("The nodes talk over a fast direct link", H2),
    Paragraph(f"NCCL all-reduce bandwidth between the two nodes by message size, measured before training (status PASS). "
              f"Large messages reach {bw1g:.1f} GB/s over RDMA, which is why syncing between nodes cost only a few percent "
              f"of throughput.", BODY),
    img(nccl_png, W),
]))

setup = [Paragraph("Setup and how these numbers were measured", H2)]
for li in [
    "Model: NVIDIA Nemotron-3-Nano-30B-A3B (BF16, mixture-of-experts), LoRA fine-tuning with NeMo AutoModel 26.10 on NVIDIA "
    "GB10 nodes (one GPU each, ~120 GB unified memory).",
    "Every run: 20 steps, global batch 16, sequence length 3072, learning rate 1e-4, same dataset. Only the parallel layout changes.",
    "<b>single-1n</b>: one node, FSDP2. <b>ddp-2n</b>: two nodes, each holding a full model copy and half of each batch. "
    "<b>ep-2n</b>: two nodes, FSDP2 with the experts split across the nodes.",
    "Per-step numbers come from each run's training log; medians, peak memory and losses from the run's summary.json.",
    "The loading comparison is from separate test runs on the same cluster the same day; the pre-warm time is projected "
    "from its read rate, since that test was stopped early.",
    "The loading fix is in the team repo on branch dev (commit 1550a32): scripts/fastload/sitecustomize.py, enabled by "
    "scripts/launch_train.sh.",
    "Evaluation: base Nemotron-3-Nano vs the same model plus the LoRA adapter from step 91 of a 140-step data-parallel run "
    "(lowest validation loss, 0.233; later steps overfit slightly). 152 test passages from 4 companies held out of training, "
    "greedy decoding, max 300 new tokens, scored with scripts/08_score.py against the teacher model's (Nemotron 3 Ultra) labels; "
    "none of the 60 passages queued for human review has been reviewed yet.",
    "Fuzzball workflows: " + ", ".join(f"{rid} {R[rid]['workflow_id']}" for rid in RUN_IDS) + "; evaluation f34b4b31 (fine-tuned), 2a594c92 (base).",
]:
    setup.append(Paragraph("• " + li, SMALL))
story.append(KeepTogether(setup))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawString(0.75 * inch, 0.5 * inch, "Team 9 · Nemotron-3-Nano fine-tuning on GB10 · 14–15 Nov 2026")
    canvas.drawRightString(letter[0] - 0.75 * inch, 0.5 * inch, f"Page {doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(OUT, pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch, topMargin=0.7 * inch,
                        bottomMargin=0.8 * inch, title="Team 9: Nemotron-3-Nano fine-tuning on 1 vs 2 GB10 nodes",
                        author="Team 9", subject="Hackathon results")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)

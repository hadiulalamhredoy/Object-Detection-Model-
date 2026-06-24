

# ============================================================
# CONFIGURATION — Edit these paths as needed
# ============================================================
import os
import shutil
import random
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from sklearn.model_selection import train_test_split
import yaml

# --- Paths ---
"""
YOLO Data Preparation Pipeline
Vehicle Detection Competition — 13 Classes, 4 CCTV Cameras
===========================================================

Dataset format confirmed:
  - train.csv  : image_id column stores filenames with .txt extension
                 (e.g. CCTV01^20260318-110000-20260318-120000_000062.txt)
                 but actual image files on disk are .jpg
  - images/    : flat folder, all .jpg files
  - Coordinates: already normalised YOLO format (0-1), no conversion needed

Run:  python yolo_data_prep.py
Deps: pip install pandas numpy matplotlib opencv-python scikit-learn pyyaml
"""

# ============================================================
# CONFIGURATION  — edit only this block
# ============================================================
from pathlib import Path

BASE_DIR      = Path(".")
TRAIN_IMG_DIR = BASE_DIR / "images"        # flat folder with all .jpg
TEST_IMG_DIR  = BASE_DIR / "images"        # adjust if test is separate
TRAIN_CSV     = BASE_DIR / "train.csv"
LABELS_DIR    = BASE_DIR / "labels"        # generated .txt labels
DATASET_DIR   = BASE_DIR / "dataset"       # final split tree
PLOTS_DIR     = BASE_DIR / "plots"
SAMPLES_DIR   = BASE_DIR / "annotated_samples"
REPORT_PATH   = BASE_DIR / "dataset_report.txt"
YAML_PATH     = BASE_DIR / "dataset.yaml"

VAL_RATIO    = 0.20
RANDOM_SEED  = 42
N_SAMPLES    = 20          # annotated images to save for visual check
IMG_EXTS     = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

CLASS_NAMES = {
    0:  "Rickshaw",    1: "Motorcycle",  2: "Tempu",
    3:  "Sedan Car",   4: "Pickup",      5: "Microbus",
    6:  "Mini Bus",    7: "Mini Truck",  8: "Agro Use",
    9:  "Medium Truck",10: "Large Bus",  11: "Heavy Truck",
    12: "Trailer",
}

# BGR colours, one per class
COLORS = [
    (255, 56,  56 ), (255, 157, 151), (255, 112, 31 ), (255, 178, 29 ),
    (207, 210, 49 ), (72,  249, 10 ), (146, 204, 23 ), (61,  219, 134),
    (26,  147, 52 ), (0,   212, 187), (44,  153, 168), (0,   194, 255),
    (52,  69,  147),
]

# ============================================================
# IMPORTS
# ============================================================
import os, shutil, random, warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
import yaml

for d in [PLOTS_DIR, SAMPLES_DIR, LABELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# HELPERS
# ============================================================

def get_img_path(image_id: str, img_dir: Path) -> Path | None:
    """
    Resolve the real image file for a given image_id.

    train.csv stores image_id with a .txt suffix, e.g.:
        CCTV01^20260318-110000-20260318-120000_000062.txt
    Actual files on disk are .jpg, so we match by STEM.

    Strategy (in order):
      1. image_id as-is  (future-proof if CSV is ever fixed)
      2. stem + each IMG_EXT  (current case: .txt -> .jpg)
    """
    p = img_dir / image_id
    if p.exists():
        return p
    stem = Path(image_id).stem
    for ext in IMG_EXTS:
        for candidate in [img_dir / (stem + ext),
                           img_dir / (stem + ext.upper())]:
            if candidate.exists():
                return candidate
    return None


def label_stem(image_id: str) -> str:
    """Return the label file stem (no ext) for an image_id."""
    return Path(image_id).stem


def hex_color(bgr):
    """BGR tuple -> hex string for matplotlib."""
    return f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}"


# ============================================================
# SECTION 1 — Load dataset
# ============================================================
print("=" * 62)
print("SECTION 1: DATASET LOADING")
print("=" * 62)

df = pd.read_csv(TRAIN_CSV)

# Derived columns used throughout
df["stem"]   = df["image_id"].apply(label_stem)
df["area"]   = df["width"] * df["height"]
df["camera"] = df["image_id"].str.extract(r"^(CCTV\d+)")

print(f"\nShape       : {df.shape}")
print(f"Columns     : {list(df.columns)}")
print(f"\nDtypes:\n{df.dtypes}")
print(f"\nNull counts:\n{df.isnull().sum()}")
print(f"\nFirst 5 rows:\n{df[['image_id','class_id','x_center','y_center','width','height']].head()}")
print(f"\nCoordinate stats:\n{df[['x_center','y_center','width','height']].describe().round(4)}")

# ============================================================
# SECTION 2 — Dataset statistics
# ============================================================
print("\n" + "=" * 62)
print("SECTION 2: DATASET STATISTICS")
print("=" * 62)

total_imgs    = df["image_id"].nunique()
total_objs    = len(df)
total_classes = df["class_id"].nunique()
avg_obj_img   = total_objs / total_imgs

objs_per_class = (df.groupby("class_id").size()
                    .reset_index(name="obj_count"))
objs_per_class["class_name"] = objs_per_class["class_id"].map(CLASS_NAMES)

imgs_per_class = (df.groupby("class_id")["image_id"].nunique()
                    .reset_index(name="img_count"))
imgs_per_class["class_name"] = imgs_per_class["class_id"].map(CLASS_NAMES)

imgs_per_camera = df.groupby("camera")["image_id"].nunique()

print(f"\nTotal training images   : {total_imgs}")
print(f"Total annotated objects : {total_objs}")
print(f"Total classes           : {total_classes}")
print(f"Avg objects / image     : {avg_obj_img:.2f}")
print(f"\nImages per camera:\n{imgs_per_camera.to_string()}")
print(f"\nObjects per class:")
print(objs_per_class[["class_id","class_name","obj_count"]].to_string(index=False))
print(f"\nImages per class:")
print(imgs_per_class[["class_id","class_name","img_count"]].to_string(index=False))

# ============================================================
# SECTION 3 — Class distribution plots
# ============================================================
print("\n" + "=" * 62)
print("SECTION 3: CLASS DISTRIBUTION ANALYSIS")
print("=" * 62)

opc = objs_per_class.sort_values("obj_count", ascending=False)
ipc = imgs_per_class.sort_values("img_count", ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle("Class Distribution", fontsize=16, fontweight="bold")

# Objects per class
bars = axes[0].bar(opc["class_name"], opc["obj_count"],
                   color=[hex_color(COLORS[i]) for i in opc["class_id"]],
                   edgecolor="black", linewidth=0.5)
axes[0].set_title("Objects per Class")
axes[0].set_xlabel("Class"); axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=45)
for b, v in zip(bars, opc["obj_count"]):
    axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+5,
                 str(v), ha="center", va="bottom", fontsize=7)

# Images per class
bars2 = axes[1].bar(ipc["class_name"], ipc["img_count"],
                    color=[hex_color(COLORS[i]) for i in ipc["class_id"]],
                    edgecolor="black", linewidth=0.5)
axes[1].set_title("Images per Class")
axes[1].set_xlabel("Class"); axes[1].set_ylabel("Count")
axes[1].tick_params(axis="x", rotation=45)
for b, v in zip(bars2, ipc["img_count"]):
    axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+1,
                 str(v), ha="center", va="bottom", fontsize=7)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "class_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {PLOTS_DIR}/class_distribution.png")

# Pie chart
fig, ax = plt.subplots(figsize=(10, 8))
ax.pie(opc["obj_count"], labels=opc["class_name"],
       autopct="%1.1f%%",
       colors=[hex_color(COLORS[i]) for i in opc["class_id"]],
       startangle=140, pctdistance=0.8)
ax.set_title("Object Distribution by Class (%)", fontsize=14, fontweight="bold")
plt.savefig(PLOTS_DIR / "class_pie.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {PLOTS_DIR}/class_pie.png")

# Camera distribution
fig, ax = plt.subplots(figsize=(8, 5))
imgs_per_camera.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
ax.set_title("Images per CCTV Camera")
ax.set_xlabel("Camera"); ax.set_ylabel("Image Count")
ax.tick_params(axis="x", rotation=0)
for p in ax.patches:
    ax.text(p.get_x()+p.get_width()/2, p.get_height()+1,
            str(int(p.get_height())), ha="center", va="bottom")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "camera_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {PLOTS_DIR}/camera_distribution.png")

# ============================================================
# SECTION 4 — Bounding box analysis
# ============================================================
print("\n" + "=" * 62)
print("SECTION 4: BOUNDING BOX ANALYSIS")
print("=" * 62)

print(f"\nAvg width   : {df['width'].mean():.4f}")
print(f"Avg height  : {df['height'].mean():.4f}")
print(f"Avg area    : {df['area'].mean():.6f}")

print(f"\nSmallest box:")
print(df.loc[df["area"].idxmin(),
      ["image_id","class_id","width","height","area"]])
print(f"\nLargest box:")
print(df.loc[df["area"].idxmax(),
      ["image_id","class_id","width","height","area"]])

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Bounding Box Statistics", fontsize=16, fontweight="bold")

axes[0,0].hist(df["width"],  bins=50, color="steelblue",  edgecolor="black", alpha=0.8)
axes[0,0].set_title("Width Distribution");  axes[0,0].set_xlabel("Normalised Width")

axes[0,1].hist(df["height"], bins=50, color="darkorange", edgecolor="black", alpha=0.8)
axes[0,1].set_title("Height Distribution"); axes[0,1].set_xlabel("Normalised Height")

axes[0,2].hist(df["area"],   bins=50, color="green",      edgecolor="black", alpha=0.8)
axes[0,2].set_title("Area Distribution");   axes[0,2].set_xlabel("Normalised Area")

sc = axes[1,0].scatter(df["width"], df["height"],
                        c=df["class_id"], cmap="tab20", alpha=0.4, s=5)
axes[1,0].set_title("Width vs Height"); axes[1,0].set_xlabel("Width")
axes[1,0].set_ylabel("Height")
plt.colorbar(sc, ax=axes[1,0], label="class_id")

bbox_cls = df.groupby("class_id")[["width","height"]].mean()
bbox_cls.index = [CLASS_NAMES[i] for i in bbox_cls.index]
bbox_cls.plot(kind="bar", ax=axes[1,1], color=["steelblue","darkorange"])
axes[1,1].set_title("Avg Box Size per Class")
axes[1,1].tick_params(axis="x", rotation=45)
axes[1,1].set_ylabel("Normalised Size")

obj_counts = df.groupby("image_id").size()
axes[1,2].hist(obj_counts, bins=30, color="purple", edgecolor="black", alpha=0.8)
axes[1,2].set_title("Objects per Image Distribution")
axes[1,2].set_xlabel("Object Count"); axes[1,2].set_ylabel("Image Count")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "bbox_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {PLOTS_DIR}/bbox_analysis.png")

# ============================================================
# SECTION 5 — Integrity verification
# ============================================================
print("\n" + "=" * 62)
print("SECTION 5: DATASET INTEGRITY VERIFICATION")
print("=" * 62)

issues = {}

# Build stem -> real filename map for everything on disk
disk_stem_map = {}   # stem -> Path
for ext in IMG_EXTS:
    for p in TRAIN_IMG_DIR.glob(f"*{ext}"):
        disk_stem_map[p.stem] = p
    for p in TRAIN_IMG_DIR.glob(f"*{ext.upper()}"):
        disk_stem_map[p.stem] = p

csv_stems = {label_stem(iid): iid for iid in df["image_id"].unique()}

# image_ids in CSV whose stem is not on disk
missing_from_disk  = [iid for stem, iid in csv_stems.items()
                      if stem not in disk_stem_map]
# stems on disk not in CSV
unannotated_on_disk = [path.name for stem, path in disk_stem_map.items()
                       if stem not in csv_stems]

issues["images_in_csv"]       = len(csv_stems)
issues["images_on_disk"]      = len(disk_stem_map)
issues["missing_from_disk"]   = missing_from_disk
issues["unannotated_on_disk"] = unannotated_on_disk

print(f"\nImages in CSV            : {len(csv_stems)}")
print(f"Images on disk           : {len(disk_stem_map)}")
print(f"CSV entries with no file : {len(missing_from_disk)}")
if missing_from_disk:
    print(f"  Examples → {missing_from_disk[:5]}")
print(f"Files with no annotation : {len(unannotated_on_disk)}")
if unannotated_on_disk:
    print(f"  Examples → {unannotated_on_disk[:5]}")

# Invalid class IDs
bad_class = df[~df["class_id"].isin(CLASS_NAMES)]
issues["invalid_class_ids"] = len(bad_class)
print(f"\nInvalid class IDs        : {len(bad_class)}")
if len(bad_class): print(bad_class.head())

# Invalid coordinates
coord_mask = (
    (df["x_center"] < 0) | (df["x_center"] > 1) |
    (df["y_center"] < 0) | (df["y_center"] > 1) |
    (df["width"]    <= 0) | (df["width"]    > 1) |
    (df["height"]   <= 0) | (df["height"]   > 1)
)
bad_coords = df[coord_mask]
issues["invalid_coords"] = len(bad_coords)
print(f"Invalid coordinates      : {len(bad_coords)}")
if len(bad_coords): print(bad_coords.head())

# Corrupted images (disk files that cv2 can't open)
corrupted = []
for stem, path in disk_stem_map.items():
    try:
        img = cv2.imread(str(path))
        if img is None:
            corrupted.append(path.name)
    except Exception:
        corrupted.append(path.name)
issues["corrupted"] = corrupted
print(f"Corrupted images         : {len(corrupted)}")
if corrupted: print(f"  → {corrupted[:5]}")

print("\n✓ Integrity check complete.")

# ============================================================
# SECTION 6 — YOLO label export
# ============================================================
print("\n" + "=" * 62)
print("SECTION 6: YOLO LABEL EXPORT")
print("=" * 62)

# NOTE: label filename = stem of image_id (always .txt)
exported = 0
for image_id, group in df.groupby("image_id"):
    out_file = LABELS_DIR / (label_stem(image_id) + ".txt")
    lines = [
        f"{int(r.class_id)} {r.x_center:.6f} {r.y_center:.6f} "
        f"{r.width:.6f} {r.height:.6f}"
        for _, r in group.iterrows()
    ]
    out_file.write_text("\n".join(lines) + "\n")
    exported += 1

print(f"Exported {exported} label files → {LABELS_DIR}/")

# ============================================================
# SECTION 7 — Annotated sample visualisation
# ============================================================
print("\n" + "=" * 62)
print("SECTION 7: VISUALISATION")
print("=" * 62)

# Only pick image_ids whose stem resolves to a real disk file
resolvable = [iid for iid in df["image_id"].unique()
              if label_stem(iid) in disk_stem_map]
samples    = random.sample(resolvable, min(N_SAMPLES, len(resolvable)))

saved = 0
for iid in samples:
    img_path = disk_stem_map[label_stem(iid)]
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    H, W = img.shape[:2]
    for _, row in df[df["image_id"] == iid].iterrows():
        cid = int(row["class_id"])
        xc, yc, bw, bh = row["x_center"], row["y_center"], row["width"], row["height"]

        x1 = int((xc - bw/2) * W);  y1 = int((yc - bh/2) * H)
        x2 = int((xc + bw/2) * W);  y2 = int((yc + bh/2) * H)

        color = COLORS[cid % len(COLORS)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        lbl = CLASS_NAMES.get(cid, str(cid))
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
        cv2.putText(img, lbl, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    cv2.imwrite(str(SAMPLES_DIR / img_path.name), img)
    saved += 1

print(f"Saved {saved} annotated samples → {SAMPLES_DIR}/")

# ============================================================
# SECTION 8 — Train / Validation split  (80 / 20, stratified)
# ============================================================
print("\n" + "=" * 62)
print("SECTION 8: TRAIN / VALIDATION SPLIT")
print("=" * 62)

# Dominant class per image for stratification
dom = (df.groupby("image_id")["class_id"]
         .agg(lambda x: x.value_counts().idxmax())
         .reset_index(name="dom_class"))

# Only include images resolvable on disk
dom = dom[dom["image_id"].apply(lambda iid: label_stem(iid) in disk_stem_map)]

train_ids, val_ids = train_test_split(
    dom["image_id"].tolist(),
    test_size=VAL_RATIO,
    random_state=RANDOM_SEED,
    stratify=dom["dom_class"].tolist(),
)

print(f"Train : {len(train_ids)} images")
print(f"Val   : {len(val_ids)} images")

for split, id_list in [("train", train_ids), ("val", val_ids)]:
    img_out   = DATASET_DIR / "images" / split
    label_out = DATASET_DIR / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    for iid in id_list:
        stem = label_stem(iid)

        # Copy image using resolved path
        src_img = disk_stem_map.get(stem)
        if src_img and src_img.exists():
            shutil.copy2(src_img, img_out / src_img.name)

        # Copy label (always <stem>.txt from LABELS_DIR)
        src_lbl = LABELS_DIR / f"{stem}.txt"
        if src_lbl.exists():
            shutil.copy2(src_lbl, label_out / f"{stem}.txt")

print(f"\nDataset tree created → {DATASET_DIR}/")

# Class balance check
train_set = set(train_ids); val_set = set(val_ids)
t_df = df[df["image_id"].isin(train_set)]
v_df = df[df["image_id"].isin(val_set)]

print(f"\n{'Class':<14} {'Train':>8} {'Val':>8} {'Val%':>7}")
print("-" * 42)
for cid, cname in CLASS_NAMES.items():
    tr = (t_df["class_id"] == cid).sum()
    vl = (v_df["class_id"] == cid).sum()
    pct = vl / (tr+vl)*100 if (tr+vl) > 0 else 0
    print(f"{cname:<14} {tr:>8} {vl:>8} {pct:>6.1f}%")

# ============================================================
# SECTION 9 — dataset.yaml
# ============================================================
print("\n" + "=" * 62)
print("SECTION 9: DATASET YAML")
print("=" * 62)

yaml_cfg = {
    "path":  str(DATASET_DIR.resolve()),
    "train": "images/train",
    "val":   "images/val",
    "nc":    len(CLASS_NAMES),
    "names": CLASS_NAMES,
}
with open(YAML_PATH, "w") as f:
    yaml.dump(yaml_cfg, f, default_flow_style=False,
              sort_keys=False, allow_unicode=True)

print(f"Saved: {YAML_PATH}")
print(open(YAML_PATH).read())

# ============================================================
# SECTION 10 — Final report
# ============================================================
print("\n" + "=" * 62)
print("SECTION 10: FINAL DATASET REPORT")
print("=" * 62)

lines = []
def L(t=""): lines.append(t); print(t)

L("=" * 62)
L("  VEHICLE DETECTION COMPETITION — DATASET REPORT")
L("=" * 62)

L("\n[1] OVERVIEW")
L(f"  Training images      : {total_imgs}")
L(f"  Annotated objects    : {total_objs}")
L(f"  Classes              : {total_classes}")
L(f"  Avg objects/image    : {avg_obj_img:.2f}")
L(f"  Cameras              : {', '.join(imgs_per_camera.index.tolist())}")

L("\n[2] CLASS DISTRIBUTION")
L(f"  {'ID':<4} {'Class':<14} {'Objects':>8} {'Images':>8} {'Obj%':>7}")
L("  " + "-" * 46)
for _, r in objs_per_class.iterrows():
    cid = int(r["class_id"])
    ic  = imgs_per_class[imgs_per_class["class_id"]==cid]["img_count"].values[0]
    pct = r["obj_count"]/total_objs*100
    L(f"  {cid:<4} {r['class_name']:<14} {r['obj_count']:>8} {ic:>8} {pct:>6.1f}%")

L("\n[3] BOUNDING BOX STATS")
for col in ["width","height","area"]:
    L(f"  {col:<8} mean={df[col].mean():.4f}  "
      f"min={df[col].min():.4f}  max={df[col].max():.4f}")

L("\n[4] DATA QUALITY")
L(f"  CSV entries missing file : {len(issues['missing_from_disk'])}")
L(f"  Files with no annotation : {len(issues['unannotated_on_disk'])}")
L(f"  Invalid class IDs        : {issues['invalid_class_ids']}")
L(f"  Invalid coordinates      : {issues['invalid_coords']}")
L(f"  Corrupted images         : {len(issues['corrupted'])}")

L("\n[5] TRAIN/VAL SPLIT")
L(f"  Train : {len(train_ids)} images  ({(1-VAL_RATIO)*100:.0f}%)")
L(f"  Val   : {len(val_ids)} images  ({VAL_RATIO*100:.0f}%)")
L(f"  Strategy : stratified by dominant class per image")

L("\n[6] OUTPUT FILES")
L(f"  Labels dir     : {LABELS_DIR}")
L(f"  Dataset dir    : {DATASET_DIR}")
L(f"  YAML           : {YAML_PATH}")
L(f"  Plots dir      : {PLOTS_DIR}")
L(f"  Samples dir    : {SAMPLES_DIR}")
L(f"  Report         : {REPORT_PATH}")

L("\n" + "=" * 62)
L("  PIPELINE COMPLETE ✓")
L("=" * 62)

REPORT_PATH.write_text("\n".join(lines))
print(f"\nReport saved → {REPORT_PATH}")
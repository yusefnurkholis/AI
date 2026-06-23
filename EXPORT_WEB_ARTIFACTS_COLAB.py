# ============================================================
# EKSPOR ARTEFAK UNTUK WEBSITE
# Jalankan setelah tahap Evaluation dan Simpan Model.
# ============================================================
import json
import shutil
from pathlib import Path
from google.colab import files

WEB_EXPORT_DIR = Path("/content/rice_leaf_web_export")
WEB_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

metrics_web = {
    "test_accuracy": float(test_accuracy),
    "test_loss": float(test_loss),
    "weighted_precision": float(precision),
    "weighted_recall": float(recall),
    "weighted_f1_score": float(f1)
}

confusion_web = {
    "labels": CLASS_NAMES,
    "matrix": cm.tolist()
}

with open(WEB_EXPORT_DIR / "metrics.json", "w", encoding="utf-8") as file:
    json.dump(metrics_web, file, indent=4)

with open(WEB_EXPORT_DIR / "confusion_matrix.json", "w", encoding="utf-8") as file:
    json.dump(confusion_web, file, indent=4)

# Salin artefak yang sudah dibuat notebook.
files_to_copy = [
    FINAL_MODEL_PATH,
    CONFIG_PATH,
    HISTORY_PATH,
    REPORT_PATH,
]

for source in files_to_copy:
    source = Path(source)
    if source.exists():
        shutil.copy2(source, WEB_EXPORT_DIR / source.name)
    else:
        print(f"Peringatan: {source.name} belum ditemukan.")

zip_path = shutil.make_archive(
    "/content/rice_leaf_web_artifacts",
    "zip",
    WEB_EXPORT_DIR
)

print("Artefak web berhasil dibuat:", zip_path)
files.download(zip_path)

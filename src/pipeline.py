"""
Yapılandırılabilir klasik görüntü işleme pipeline'ı.

Bir pipeline, JSON'a yazılabilen bir sözlükle (config) tanımlanır:

{
  "intensity":   {"method": "none" | "stretch" | "histeq" | "clahe" | "histmatch", ...},
  "filter":      {"method": "none" | "gaussian" | "median" | "sharpen" | "gaussian+sharpen", ...},
  "segmentation":{"method": "fixed" | "otsu" | "adaptive" | "triangle" | "kmeans", ...},
  "morphology":  [{"op": "open"|"close"|"erode"|"dilate"|"fill_holes", "size": int}, ...],
  "separation":  {"method": "none" | "watershed", ...},
  "min_area":    int
}

Bütün adımlar OpenCV / scikit-image / SciPy'deki klasik algoritmalardır.
Her adımın ara çıktısı `run_pipeline(..., return_steps=True)` ile alınabilir.
"""

from __future__ import annotations

import copy
import time

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import exposure
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu, threshold_triangle
from skimage.measure import label
from skimage.segmentation import watershed

CONNECTIVITY = 2  # 8-komşuluk


# ---------------------------------------------------------------------------
# 1) Yoğunluk dönüşümleri  (çıktı: float32, [0, 1])
# ---------------------------------------------------------------------------
def intensity_transform(img: np.ndarray, cfg: dict, ref_img: np.ndarray | None = None) -> np.ndarray:
    m = cfg.get("method", "none")
    x = img.astype(np.float32) / 255.0
    if m == "none":
        return x
    if m == "stretch":
        # Görüntü başına yüzdelik tabanlı doğrusal germe (contrast stretching)
        lo, hi = np.percentile(img, [cfg.get("p_low", 0.0), cfg.get("p_high", 99.5)])
        return np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
    if m == "histeq":
        return exposure.equalize_hist(img, nbins=256).astype(np.float32)
    if m == "clahe":
        clahe = cv2.createCLAHE(clipLimit=cfg.get("clip", 2.0),
                                tileGridSize=(cfg.get("tiles", 8),) * 2)
        return clahe.apply(img).astype(np.float32) / 255.0
    if m == "histmatch":
        if ref_img is None:
            raise ValueError("histmatch için geliştirme kümesinden referans görüntü gerekli")
        return (exposure.match_histograms(img, ref_img).astype(np.float32) / 255.0)
    if m == "log":
        return (np.log1p(img.astype(np.float32)) / np.log(256.0)).astype(np.float32)
    raise ValueError(f"Bilinmeyen yoğunluk yöntemi: {m}")


# ---------------------------------------------------------------------------
# 2) Uzamsal filtreler  (girdi/çıktı float32)
# ---------------------------------------------------------------------------
def _sharpen(x: np.ndarray, amount: float) -> np.ndarray:
    """Konvolüsyon tabanlı keskinleştirme: 3x3 Laplace çekirdeğiyle x - amount * Lap(x)."""
    lap_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)
    lap = cv2.filter2D(x, -1, lap_kernel, borderType=cv2.BORDER_REFLECT)
    return np.clip(x - amount * lap, 0, 1)


def spatial_filter(x: np.ndarray, cfg: dict) -> np.ndarray:
    m = cfg.get("method", "none")
    if m == "none":
        return x
    if m == "gaussian":
        k = cfg["ksize"]
        return cv2.GaussianBlur(x, (k, k), cfg.get("sigma", 0), borderType=cv2.BORDER_REFLECT)
    if m == "median":
        k = cfg["ksize"]
        # OpenCV medianBlur, k>5 için 8-bit girdi ister
        u8 = np.round(x * 255).astype(np.uint8)
        return cv2.medianBlur(u8, k).astype(np.float32) / 255.0
    if m == "sharpen":
        return _sharpen(x, cfg.get("amount", 1.0))
    if m == "gaussian+sharpen":
        k = cfg["ksize"]
        g = cv2.GaussianBlur(x, (k, k), 0, borderType=cv2.BORDER_REFLECT)
        return _sharpen(g, cfg.get("amount", 1.0))
    raise ValueError(f"Bilinmeyen filtre: {m}")


# ---------------------------------------------------------------------------
# 3) Segmentasyon  (çıktı: bool maske)
# ---------------------------------------------------------------------------
def segment(x: np.ndarray, cfg: dict) -> tuple[np.ndarray, float | None]:
    m = cfg["method"]
    if m == "fixed":
        t = cfg["threshold"] / 255.0  # eşik 0-255 ölçeğinde verilir
        return x > t, cfg["threshold"]
    # Global eşikler her zaman 8-bit (0-255) temsil üzerinde hesaplanır;
    # böylece baseline ile birebir aynı Otsu eşiği elde edilir.
    if m == "otsu":
        u8 = np.round(x * 255).astype(np.uint8)
        t = threshold_otsu(u8)
        return u8 > t, float(t)
    if m == "triangle":
        u8 = np.round(x * 255).astype(np.uint8)
        t = threshold_triangle(u8)
        return u8 > t, float(t)
    if m == "adaptive":
        # OpenCV adaptif eşik (Gauss ağırlıklı yerel ortalama - C)
        u8 = np.round(x * 255).astype(np.uint8)
        method = (cv2.ADAPTIVE_THRESH_GAUSSIAN_C if cfg.get("weight", "gaussian") == "gaussian"
                  else cv2.ADAPTIVE_THRESH_MEAN_C)
        mask = cv2.adaptiveThreshold(u8, 1, method, cv2.THRESH_BINARY,
                                     cfg["block"], -cfg.get("C", 0))
        mask = mask.astype(bool)
        if "global_floor" in cfg:  # çok karanlık arka planda gürültüyü bastırmak için
            mask &= u8 > cfg["global_floor"]
        return mask, None
    if m == "kmeans":
        k = cfg.get("k", 2)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
        data = x.reshape(-1, 1).astype(np.float32)
        cv2.setRNGSeed(0)
        _, lab, centers = cv2.kmeans(data, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        order = np.argsort(centers.ravel())
        fg_clusters = order[cfg.get("bg_clusters", 1):]
        return np.isin(lab.reshape(x.shape), fg_clusters), None
    raise ValueError(f"Bilinmeyen segmentasyon: {m}")


# ---------------------------------------------------------------------------
# 4) Morfoloji
# ---------------------------------------------------------------------------
def _disk(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def morphology(mask: np.ndarray, ops: list[dict]) -> np.ndarray:
    m = mask.astype(np.uint8)
    for op in ops:
        name = op["op"]
        if name == "fill_holes":
            m = ndi.binary_fill_holes(m).astype(np.uint8)
            continue
        se = _disk(op["size"])
        cv_op = {"erode": None, "dilate": None,
                 "open": cv2.MORPH_OPEN, "close": cv2.MORPH_CLOSE}[name]
        if name == "erode":
            m = cv2.erode(m, se)
        elif name == "dilate":
            m = cv2.dilate(m, se)
        else:
            m = cv2.morphologyEx(m, cv_op, se)
    return m.astype(bool)


# ---------------------------------------------------------------------------
# 5) Alan filtresi ve temas eden nesnelerin ayrılması
# ---------------------------------------------------------------------------
def filter_small(labels: np.ndarray, min_area: int) -> np.ndarray:
    """Alanı min_area'dan küçük bileşenleri siler ve yeniden etiketler."""
    if min_area <= 0:
        return labels
    areas = np.bincount(labels.ravel())
    keep = areas >= min_area
    keep[0] = False
    return label(keep[labels], connectivity=CONNECTIVITY)


def separate(mask: np.ndarray, cfg: dict) -> np.ndarray:
    m = cfg.get("method", "none")
    if m == "none":
        return label(mask, connectivity=CONNECTIVITY)
    if m == "watershed":
        # Uzaklık dönüşümü + yerel maksimumlardan işaretleyici + watershed
        dist = ndi.distance_transform_edt(mask)
        if cfg.get("smooth_sigma", 0) > 0:
            dist = ndi.gaussian_filter(dist, cfg["smooth_sigma"])
        comps = label(mask, connectivity=CONNECTIVITY)
        peaks = peak_local_max(dist, min_distance=cfg["min_distance"],
                               threshold_abs=cfg.get("min_peak", 0),
                               labels=comps, exclude_border=False)
        markers = np.zeros(mask.shape, np.int32)
        markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
        ws = watershed(-dist, markers, mask=mask)
        # İşaretleyici düşmeyen (çok küçük) bileşenler kaybolmasın
        orphan = mask & (ws == 0)
        if orphan.any():
            extra = label(orphan, connectivity=CONNECTIVITY)
            ws[extra > 0] = extra[extra > 0] + ws.max()
        return _relabel(ws)
    raise ValueError(f"Bilinmeyen ayırma yöntemi: {m}")


def _relabel(lab: np.ndarray) -> np.ndarray:
    ids = np.unique(lab)
    ids = ids[ids > 0]
    lut = np.zeros(lab.max() + 1, np.int32)
    lut[ids] = np.arange(1, len(ids) + 1)
    return lut[lab]


# ---------------------------------------------------------------------------
# Tam pipeline
# ---------------------------------------------------------------------------
DEFAULT = {
    "intensity": {"method": "none"},
    "filter": {"method": "none"},
    "segmentation": {"method": "otsu"},
    "morphology": [],
    "separation": {"method": "none"},
    "min_area": 0,
}


def make_config(**overrides) -> dict:
    cfg = copy.deepcopy(DEFAULT)
    for k, v in overrides.items():
        cfg[k] = copy.deepcopy(v)
    return cfg


def run_pipeline(img: np.ndarray, cfg: dict, ref_img: np.ndarray | None = None,
                 return_steps: bool = False) -> dict:
    t0 = time.perf_counter()
    x = intensity_transform(img, cfg["intensity"], ref_img)
    xi = x
    x = spatial_filter(x, cfg["filter"])
    xf = x
    mask, thr = segment(x, cfg["segmentation"])
    mask_raw = mask
    mask = morphology(mask, cfg["morphology"])
    labels = separate(mask, cfg["separation"])
    labels = filter_small(labels, cfg["min_area"])
    elapsed = time.perf_counter() - t0
    out = {"count": int(labels.max()), "time_s": elapsed, "threshold": thr, "labels": labels}
    if return_steps:
        out.update({"intensity": xi, "filtered": xf, "mask_raw": mask_raw, "mask": mask})
    return out


def describe(cfg: dict) -> dict:
    """Deney tablosu için okunabilir sütunlar."""
    def kv(d, skip=("method",)):
        return ", ".join(f"{k}={v}" for k, v in d.items() if k not in skip)

    i, f, s, sep = cfg["intensity"], cfg["filter"], cfg["segmentation"], cfg["separation"]
    morph = " → ".join(
        op["op"] if op["op"] == "fill_holes" else f"{op['op']}(r={op['size']})"
        for op in cfg["morphology"]) or "yok"
    if sep["method"] != "none":
        morph += f" + watershed({kv(sep)})"
    return {
        "Yogunluk": i["method"] + (f" ({kv(i)})" if kv(i) else ""),
        "Filtre": f["method"] + (f" ({kv(f)})" if kv(f) else ""),
        "Segmentasyon": s["method"] + (f" ({kv(s)})" if kv(s) else ""),
        "Morfoloji": morph,
        "MinAlan": cfg["min_area"],
    }

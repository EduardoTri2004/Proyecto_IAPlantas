import os

# 1. CONFIGURACIÓN SEGURA DE RECURSOS (MITAD DE NÚCLEOS CPU Y ~1.2 GB RAM)
TOTAL_CORES = os.cpu_count() or 4
SAFE_WORKERS = max(2, min(6, TOTAL_CORES // 2))

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Modo CPU estable

import gc
import cv2
import json
import time
import joblib
import random
import warnings
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

cv2.setNumThreads(1)
torch.set_num_threads(SAFE_WORKERS)
warnings.filterwarnings('ignore')

# ==============================================================================
# PARCHE EN MEMORIA PARA OPENCV 4.11 (Compatible con Fase2_Extraccion.py intacto)
# ==============================================================================
_orig_imread = cv2.imread
_orig_sobel = cv2.Sobel
_orig_laplacian = cv2.Laplacian
_orig_filter2D = cv2.filter2D


def _safe_imread(path_or_array, flags=cv2.IMREAD_COLOR):
    if isinstance(path_or_array, np.ndarray):
        return path_or_array
    return _orig_imread(path_or_array, flags)


def _safe_sobel(src, ddepth, dx, dy, *args, **kwargs):
    if isinstance(src, np.ndarray) and src.dtype == np.float64 and ddepth == cv2.CV_32F:
        src = src.astype(np.float32)
    return _orig_sobel(src, ddepth, dx, dy, *args, **kwargs)


def _safe_laplacian(src, ddepth, *args, **kwargs):
    if isinstance(src, np.ndarray) and src.dtype == np.float64 and ddepth == cv2.CV_32F:
        src = src.astype(np.float32)
    return _orig_laplacian(src, ddepth, *args, **kwargs)


def _safe_filter2D(src, ddepth, kernel, *args, **kwargs):
    if isinstance(src, np.ndarray) and src.dtype == np.float64 and ddepth == cv2.CV_32F:
        src = src.astype(np.float32)
    return _orig_filter2D(src, ddepth, kernel, *args, **kwargs)


cv2.imread = _safe_imread
cv2.Sobel = _safe_sobel
cv2.Laplacian = _safe_laplacian
cv2.filter2D = _safe_filter2D

from Fase1_Segmentacion import LabelStudioParser, EmbryoProcessor
from Fase2_Extraccion import AdvancedBiologicalExtractor
from Fase3_Modelado import EmbryoGateFormerV2


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(42)


# ==============================================================================
# 1. WORKER PARALELO EN RAM PARA EXTRACCIÓN DE LOS 157/119 DESCRIPTORES
# ==============================================================================
def _init_pool_worker():
    cv2.setNumThreads(1)
    _pool_extract_worker.extractor = AdvancedBiologicalExtractor()


def _pool_extract_worker(rgba_crop):
    if rgba_crop is None or rgba_crop.size == 0:
        return None
    h, w = rgba_crop.shape[:2]
    if max(h, w) > 160:
        scale = 160.0 / float(max(h, w))
        rgba_crop = cv2.resize(
            rgba_crop, (max(8, int(w * scale)), max(8, int(h * scale))),
            interpolation=cv2.INTER_AREA
        )
    return _pool_extract_worker.extractor.extract_features(rgba_crop)


# ==============================================================================
# 2. PREDICTOR EN CPU DEL MODELO PROFUNDO ENTRENADO EN FASE 3
# ==============================================================================
class TrainedModelPredictorCPU:
    def __init__(self, model_dir, batch_size=256):
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.device = torch.device('cpu')
        self._load_artifacts()

    def _load_artifacts(self):
        ckpt_path = os.path.join(self.model_dir, 'best_embryo_gateformer.pth')
        scaler_path = os.path.join(self.model_dir, 'yeo_johnson_scaler.pkl')
        cols_path = os.path.join(self.model_dir, 'feature_columns.pkl')
        tree_path = os.path.join(self.model_dir, 'tree_expert_branch.pkl')
        thresh_path = os.path.join(self.model_dir, 'optimal_threshold.pkl')

        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        self.feature_cols = joblib.load(cols_path)
        self.family_indices = checkpoint['family_indices']
        self.scaler = joblib.load(scaler_path)
        self.base_threshold = float(joblib.load(thresh_path)) if os.path.exists(thresh_path) else 0.53

        self.model = EmbryoGateFormerV2(
            family_indices_dict=self.family_indices,
            total_features=len(self.feature_cols),
            d_model=64,
            n_heads=4,
            dropout=0.25
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.tree_models = joblib.load(tree_path) if os.path.exists(tree_path) else None
        if self.tree_models is not None and 'et' in self.tree_models:
            self.tree_models['et'].n_jobs = SAFE_WORKERS

    def _transform_features(self, X_raw):
        if isinstance(self.scaler, list):
            X_scaled = np.zeros_like(X_raw, dtype=np.float64)
            for idx, sc in enumerate(self.scaler):
                X_scaled[:, [idx]] = sc.transform(X_raw[:, [idx]])
        else:
            X_scaled = self.scaler.transform(X_raw)
        return np.clip(np.nan_to_num(X_scaled, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)

    def predict_probabilities(self, feature_dicts_list):
        if not feature_dicts_list:
            return np.array([])

        df = pd.DataFrame(feature_dicts_list)
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        df = df[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        X_raw = df.values.astype(np.float64)
        X_scaled = self._transform_features(X_raw)
        del df, X_raw

        nn_probs_list = []
        with torch.no_grad():
            for start in range(0, len(X_scaled), self.batch_size):
                batch_x = torch.tensor(X_scaled[start:start + self.batch_size], dtype=torch.float32)
                logits = self.model(batch_x)
                probs = torch.sigmoid(logits).numpy()
                nn_probs_list.extend(np.atleast_1d(probs))

        nn_probs = np.array(nn_probs_list)

        if self.tree_models is not None:
            hgb_p = self.tree_models['hgb'].predict_proba(X_scaled)[:, 1]
            et_p = self.tree_models['et'].predict_proba(X_scaled)[:, 1]
            tree_probs = 0.65 * hgb_p + 0.35 * et_p
            return 0.55 * nn_probs + 0.45 * tree_probs

        return nn_probs


# ==============================================================================
# 3. SEGMENTADOR CONTEXTUAL DE PÍXELES (100% AUTÓCTONO A LA IMAGEN ACTUAL)
# ==============================================================================
class PixelEmbryoSegmenter:
    """
    Aprende la firma física y espectral a nivel de píxel (LAB, HSV, relieve de domo DoG,
    suavidad epidérmica, ausencia de poros rojizos y distancia al borde del explante)
    que distingue un Embrión Somático real frente al callo/hoja madre y al agar gris.
    Garantiza que los contornos provengan 100% de la imagen actual y nunca de otras fotos.
    """

    def __init__(self, model_dir, work_dim=320):
        self.model_dir = model_dir
        self.work_dim = work_dim
        self.cache_path = os.path.join(model_dir, 'pixel_context_segmenter.pkl')
        self.clf = None

    @staticmethod
    def compute_pixel_feature_cube(work_bgr):
        """
        Extrae un cubo de 12 mapas físicos por cada píxel (H, W, 12) + la máscara
        estricta de tejido vegetal (global_tissue) que bloquea al 100% el fondo gris.
        """
        h, w = work_bgr.shape[:2]
        lab = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        hsv = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        bgr_f = work_bgr.astype(np.float32)
        gray = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        L, A, B_lab = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
        H_c, S_c, V_c = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # 1. Separar explante vegetal del fondo gris de agar
        low_sat = S_c < np.percentile(S_c, 32)
        a_bg = float(np.median(A[low_sat])) if np.count_nonzero(low_sat) > 40 else 128.0
        b_bg = float(np.median(B_lab[low_sat])) if np.count_nonzero(low_sat) > 40 else 128.0

        chroma_bg = np.sqrt((A - a_bg) ** 2 + (B_lab - b_bg) ** 2) + 0.75 * S_c
        bot_u8 = cv2.normalize(chroma_bg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, global_tissue = cv2.threshold(bot_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        global_tissue = cv2.morphologyEx(global_tissue, cv2.MORPH_CLOSE, k_close)

        # Rellenar pequeños huecos dentro del explante
        inv_bg = cv2.bitwise_not(global_tissue)
        cnts_bg, _ = cv2.findContours(inv_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts_bg:
            if cv2.contourArea(c) < 0.04 * (h * w):
                cv2.drawContours(global_tissue, [c], -1, 255, -1)

        tissue_bool = global_tissue > 0
        l_med = float(np.median(L[tissue_bool])) if np.any(tissue_bool) else 128.0
        s_med = float(np.median(S_c[tissue_bool])) if np.any(tissue_bool) else 80.0

        # 2. Índices cromáticos normalizados al explante actual
        f1_L_rel = (L - l_med) / 50.0
        f2_A_rel = (A - a_bg) / 40.0
        f3_B_rel = (B_lab - b_bg) / 40.0
        f4_S_rel = (S_c - s_med) / 50.0

        rgb_sum = bgr_f[:, :, 0] + bgr_f[:, :, 1] + bgr_f[:, :, 2] + 1e-4
        f5_exg = (2.0 * bgr_f[:, :, 1] - bgr_f[:, :, 2] - bgr_f[:, :, 0]) / rgb_sum
        f6_grvi = (bgr_f[:, :, 1] - bgr_f[:, :, 2]) / (bgr_f[:, :, 1] + bgr_f[:, :, 2] + 1e-4)

        # 3. Relieve de domo multiescala (Diferencia de Gaussianas)
        blur_3 = cv2.GaussianBlur(gray, (3, 3), 0)
        blur_11 = cv2.GaussianBlur(gray, (11, 11), 0)
        sig_macro = max(11, int(min(h, w) * 0.11))
        blur_macro = cv2.GaussianBlur(gray, (0, 0), sigmaX=sig_macro, sigmaY=sig_macro)

        f7_dog_fine = (blur_11 - blur_3) / 30.0
        f8_dog_macro = (blur_macro - blur_3) / 45.0

        # 4. Rugosidad porosa (alta en cáscara de hoja con puntos rojos, baja en domo de embrión)
        micro_diff = np.abs(blur_3 - blur_11)
        f9_roughness = cv2.GaussianBlur(micro_diff, (15, 15), 0) / 20.0

        # 5. Penalización de reflejo especular blanco
        f10_specular = np.clip((L - 165.0) / 40.0, 0.0, 3.0) * np.clip(1.0 - S_c / 150.0, 0.0, 1.0)

        # 6. Distancia normalizada desde el borde del explante (agar)
        pad = 20
        padded_t = np.pad(global_tissue, pad, mode='edge')
        dist_agar = cv2.distanceTransform(padded_t, cv2.DIST_L2, 5)[pad:pad + h, pad:pad + w]
        f11_dist_edge = dist_agar / float(max(1, min(h, w)) * 0.35)

        # 7. Energía de gradiente suavizada (nitidez focal)
        gx = cv2.Sobel(blur_3, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur_3, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        f12_focus = cv2.GaussianBlur(grad_mag, (11, 11), 0) / 40.0

        cube = np.stack([
            f1_L_rel, f2_A_rel, f3_B_rel, f4_S_rel,
            f5_exg, f6_grvi, f7_dog_fine, f8_dog_macro,
            f9_roughness, f10_specular, f11_dist_edge, f12_focus
        ], axis=-1).astype(np.float32)

        return cube, global_tissue, grad_mag

    def fit_or_load(self, train_tasks, resolve_fn, search_dirs, clahe):
        if os.path.exists(self.cache_path):
            print(f"[*] Segmentador Contextual de Píxeles cargado desde: {self.cache_path}")
            self.clf = joblib.load(self.cache_path)
            return

        print("[*] Entrenando Segmentador Contextual de Píxeles sobre muestras de las 900 fotos...")
        # Muestrear uniformemente 150 imágenes de entrenamiento
        step = max(1, len(train_tasks) // 150)
        sampled_tasks = train_tasks[::step][:150]

        X_pix_list, y_pix_list = [], []

        for task in sampled_tasks:
            polys = LabelStudioParser.extract_polygons(task)
            valid_polys = [p for p in polys if len(p) >= 3 and cv2.contourArea(p) > 20]
            if not valid_polys:
                continue

            img_path, _ = resolve_fn(task, search_dirs)
            if img_path is None:
                continue
            img = _orig_imread(img_path)
            if img is None:
                continue

            orig_h, orig_w = img.shape[:2]
            scale = self.work_dim / float(max(orig_h, orig_w))
            work_w, work_h = max(64, int(orig_w * scale)), max(64, int(orig_h * scale))

            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            enhanced = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
            work_bgr = cv2.resize(enhanced, (work_w, work_h), interpolation=cv2.INTER_AREA)

            cube, global_tissue, _ = self.compute_pixel_feature_cube(work_bgr)

            # Construir máscara de Embriones Reales (Clase 1) en resolución de trabajo
            es_mask = np.zeros((work_h, work_w), dtype=np.uint8)
            scaled_polys = [
                np.round(p.astype(np.float32) * scale).astype(np.int32)
                for p in valid_polys
            ]
            cv2.fillPoly(es_mask, scaled_polys, 255)

            # Zona de exclusión dilatada para muestrear negativos puros de Planta Madre / Callo
            excl = cv2.dilate(es_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
            pos_coords = np.column_stack(np.where((es_mask > 0) & (global_tissue > 0)))
            neg_plant_coords = np.column_stack(np.where((excl == 0) & (global_tissue > 0)))
            neg_agar_coords = np.column_stack(np.where(global_tissue == 0))

            if len(pos_coords) > 20:
                idx_p = np.random.choice(len(pos_coords), min(220, len(pos_coords)), replace=False)
                pts_p = pos_coords[idx_p]
                X_pix_list.append(cube[pts_p[:, 0], pts_p[:, 1]])
                y_pix_list.append(np.ones(len(pts_p), dtype=np.int64))

            if len(neg_plant_coords) > 20:
                idx_np = np.random.choice(len(neg_plant_coords), min(240, len(neg_plant_coords)), replace=False)
                pts_np = neg_plant_coords[idx_np]
                X_pix_list.append(cube[pts_np[:, 0], pts_np[:, 1]])
                y_pix_list.append(np.zeros(len(pts_np), dtype=np.int64))

            if len(neg_agar_coords) > 10:
                idx_na = np.random.choice(len(neg_agar_coords), min(60, len(neg_agar_coords)), replace=False)
                pts_na = neg_agar_coords[idx_na]
                X_pix_list.append(cube[pts_na[:, 0], pts_na[:, 1]])
                y_pix_list.append(np.zeros(len(pts_na), dtype=np.int64))

        X_all = np.vstack(X_pix_list)
        y_all = np.concatenate(y_pix_list)

        self.clf = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.06, max_leaf_nodes=31,
            min_samples_leaf=25, l2_regularization=1.0, random_state=42
        )
        self.clf.fit(X_all, y_all)
        joblib.dump(self.clf, self.cache_path)
        del X_all, y_all, X_pix_list, y_pix_list
        gc.collect()
        print(f"[*] Segmentador Contextual de Píxeles entrenado y guardado en: {self.cache_path}")

    def predict_heatmap(self, work_bgr):
        """
        Devuelve el mapa de probabilidad embrionaria P_pix(ES | x, y) en [0, 1]
        estrictamente restringido al tejido vegetal real (0.0 en el fondo de agar).
        """
        cube, global_tissue, grad_mag = self.compute_pixel_feature_cube(work_bgr)
        h, w, c = cube.shape
        flat_x = cube.reshape(-1, c)
        probs = np.zeros(h * w, dtype=np.float32)

        # Solo evaluar píxeles que pertenecen al tejido vegetal (ahorra tiempo y bloquea el agar)
        tissue_flat = (global_tissue.ravel() > 0)
        if np.any(tissue_flat):
            probs[tissue_flat] = self.clf.predict_proba(flat_x[tissue_flat])[:, 1].astype(np.float32)

        heatmap = probs.reshape(h, w)
        heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)
        heatmap[global_tissue == 0] = 0.0
        return heatmap, global_tissue, grad_mag


# ==============================================================================
# 4. ESCÁNER DE CONTORNOS 100% REALES (HEATMAP + WATERSHED + VENTANAS MULTIESCALA)
# ==============================================================================
class NativeContourEmbryoScanner:
    """
    Genera propuestas poligonales siguiendo EXCLUSIVAMENTE los bordes físicos y el tejido
    embrionario de la fotografía actual:
      1. Calcula el mapa de probabilidad contextual P_pix(ES | x, y) de la imagen actual.
      2. Aplica umbrales multinivel + Watershed controlado por crestas de gradiente para
         separar embriones en racimo (como los dos embriones contiguos o racimos de 7).
      3. Recorre ventanas deslizantes multiescala (12% a 70%) aplicando Otsu local
         estrictamente recortado contra global_tissue (0% de fondo de agar permitido).
    """

    def __init__(self, pixel_segmenter, work_dim=320):
        self.pixel_segmenter = pixel_segmenter
        self.work_dim = work_dim
        self.rel_window_scales = [0.13, 0.20, 0.30, 0.44, 0.62]
        self.rel_min_area = 0.0028  # Permite embriones pequeños en racimos de 6-8 sin aceptar poros
        self.rel_min_side = 0.055  # Lado mínimo >= 5.5% de la imagen
        self.rel_max_area = 0.4800  # Soporta embriones macro grandes
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _enhance_illumination(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self.clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)

    @staticmethod
    def _fast_distance_opening(binary_mask, radius_px):
        dist_in = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        eroded = (dist_in >= radius_px).astype(np.uint8)
        if np.count_nonzero(eroded) == 0:
            return np.zeros_like(binary_mask)
        dist_out = cv2.distanceTransform(1 - eroded, cv2.DIST_L2, 5)
        return (dist_out <= radius_px).astype(np.uint8) * 255

    @staticmethod
    def _polygon_mask_iou_and_iomin(candA, candB):
        boxA, boxB = candA['bbox'], candB['bbox']
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        if xB <= xA or yB <= yA:
            return 0.0, 0.0

        ux1, uy1 = min(boxA[0], boxB[0]), min(boxA[1], boxB[1])
        ux2, uy2 = max(boxA[2], boxB[2]), max(boxA[3], boxB[3])
        uw, uh = ux2 - ux1, uy2 - uy1
        if uw <= 0 or uh <= 0:
            return 0.0, 0.0

        scale = 96.0 / float(max(uw, uh)) if max(uw, uh) > 96 else 1.0
        rw, rh = max(4, int(uw * scale)), max(4, int(uh * scale))

        mA = np.zeros((rh, rw), dtype=np.uint8)
        mB = np.zeros((rh, rw), dtype=np.uint8)
        pA = ((candA['polygon'] - np.array([[[ux1, uy1]]], dtype=np.int32)).astype(np.float32) * scale).astype(np.int32)
        pB = ((candB['polygon'] - np.array([[[ux1, uy1]]], dtype=np.int32)).astype(np.float32) * scale).astype(np.int32)

        cv2.fillPoly(mA, [pA], 1)
        cv2.fillPoly(mB, [pB], 1)

        inter = float(np.count_nonzero(mA & mB))
        if inter == 0:
            return 0.0, 0.0
        areaA, areaB = float(np.count_nonzero(mA)), float(np.count_nonzero(mB))
        return inter / (areaA + areaB - inter + 1e-6), inter / (min(areaA, areaB) + 1e-6)

    def _deduplicate_geometries(self, geom_candidates, max_candidates=60):
        if not geom_candidates:
            return []
        geom_candidates.sort(key=lambda d: d['context_prob'] * d['solidity'], reverse=True)
        kept = []
        for cand in geom_candidates:
            is_dup = False
            for k in kept:
                iou, iomin = self._polygon_mask_iou_and_iomin(cand, k)
                if iou > 0.65 or iomin > 0.82:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(cand)
                if len(kept) >= max_candidates:
                    break
        return kept

    def extract_candidates_from_image(self, img_bgr, executor):
        orig_h, orig_w = img_bgr.shape[:2]
        enhanced_orig = self._enhance_illumination(img_bgr)
        rgba_full = cv2.cvtColor(enhanced_orig, cv2.COLOR_BGR2BGRA)

        scale_w = self.work_dim / float(max(orig_h, orig_w))
        work_w = max(64, int(orig_w * scale_w))
        work_h = max(64, int(orig_h * scale_w))
        inv_scale = 1.0 / scale_w

        work_bgr = cv2.resize(enhanced_orig, (work_w, work_h), interpolation=cv2.INTER_AREA)
        work_gray = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2GRAY)
        work_area = float(work_h * work_w)
        work_base = float(min(work_h, work_w))

        min_w_area = max(25.0, self.rel_min_area * work_area)
        max_w_area = self.rel_max_area * work_area
        min_w_side = max(11, int(self.rel_min_side * work_base))

        # 1. Calcular Mapa Contextual de Probabilidad Píxel a Píxel sobre la imagen actual
        heatmap, global_tissue, grad_mag = self.pixel_segmenter.predict_heatmap(work_bgr)
        heatmap_u8 = (np.clip(heatmap, 0.0, 1.0) * 255.0).astype(np.uint8)

        geom_proposals = []

        def _register_native_contour(cnt, off_x=0, off_y=0, win_w=None, win_h=None):
            if len(cnt) < 5:
                return
            lx, ly, lw, lh = cv2.boundingRect(cnt)
            if win_w is not None and win_h is not None:
                walls_touched = (
                        int(lx <= 1) + int(ly <= 1)
                        + int((lx + lw) >= (win_w - 1))
                        + int((ly + lh) >= (win_h - 1))
                )
                if walls_touched >= 2:
                    return

            abs_w_cnt = cnt + np.array([[[off_x, off_y]]], dtype=np.int32)
            wx, wy, ww, wh = cv2.boundingRect(abs_w_cnt)
            if max(ww, wh) < min_w_side or min(ww, wh) < int(min_w_side * 0.40):
                return

            # REGLA ESTRICTA DE COHERENCIA FÍSICA:
            # Recortar la máscara del candidato contra global_tissue para que NUNCA pise el agar gris
            local_poly = abs_w_cnt - np.array([[[wx, wy]]], dtype=np.int32)
            local_mask = np.zeros((wh, ww), dtype=np.uint8)
            cv2.fillPoly(local_mask, [local_poly], 255)
            roi_tissue = global_tissue[wy:wy + wh, wx:wx + ww]
            clipped_mask = cv2.bitwise_and(local_mask, roi_tissue)

            if np.count_nonzero(clipped_mask) < min_w_area:
                return

            # Re-extraer el contorno exacto ya recortado al tejido vegetal
            c_cnts, _ = cv2.findContours(clipped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not c_cnts:
                return
            clean_local_cnt = max(c_cnts, key=cv2.contourArea)
            w_area = float(cv2.contourArea(clean_local_cnt))
            if not (min_w_area <= w_area <= max_w_area):
                return

            eps = 0.008 * cv2.arcLength(clean_local_cnt, True)
            smooth_local = cv2.approxPolyDP(clean_local_cnt, eps, True)
            if len(smooth_local) < 5:
                smooth_local = clean_local_cnt

            perim = cv2.arcLength(smooth_local, True)
            if perim <= 0:
                return
            circularity = (4.0 * np.pi * w_area) / (perim ** 2)
            hull_area = cv2.contourArea(cv2.convexHull(smooth_local))
            solidity = float(w_area / (hull_area + 1e-6))
            if solidity < 0.45 or circularity < 0.12:
                return

            final_w_cnt = smooth_local + np.array([[[wx, wy]]], dtype=np.int32)
            fx, fy, fw, fh = cv2.boundingRect(final_w_cnt)
            if max(fw, fh) < min_w_side or float(max(fw, fh)) / (min(fw, fh) + 1e-6) > 3.6:
                return

            # Evaluar la probabilidad contextual media de los píxeles dentro del contorno
            f_mask = np.zeros((fh, fw), dtype=np.uint8)
            cv2.fillPoly(f_mask, [final_w_cnt - np.array([[[fx, fy]]], dtype=np.int32)], 255)
            f_bool = f_mask > 0
            mean_context_prob = float(np.mean(heatmap[fy:fy + fh, fx:fx + fw][f_bool]))
            max_context_prob = float(np.max(heatmap[fy:fy + fh, fx:fx + fw][f_bool]))

            # Rechazar candidatos cuyos píxeles tienen baja probabilidad contextual de ser embrión
            if mean_context_prob < 0.24 and max_context_prob < 0.45:
                return

            # Proyectar contorno real a la resolución nativa de la imagen
            orig_poly = np.clip(
                np.round(final_w_cnt.astype(np.float32) * inv_scale),
                [0, 0], [orig_w - 1, orig_h - 1]
            ).astype(np.int32)
            ox, oy, ow, oh = cv2.boundingRect(orig_poly)
            if ow > 12 and oh > 12:
                geom_proposals.append({
                    'bbox': [ox, oy, ox + ow, oy + oh],
                    'polygon': orig_poly,
                    'area': float(cv2.contourArea(orig_poly)),
                    'context_prob': mean_context_prob,
                    'solidity': solidity
                })

        # --- A. Segmentación Multinivel + Watershed sobre el Mapa de Calor Píxel a Píxel ---
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges_canny = cv2.Canny(work_gray, 25, 85)

        for th_prob in (0.30, 0.40, 0.50, 0.60, 0.70, 0.78):
            level_mask = ((heatmap >= th_prob) & (global_tissue > 0)).astype(np.uint8) * 255
            if np.count_nonzero(level_mask) < min_w_area:
                continue
            level_mask = cv2.morphologyEx(level_mask, cv2.MORPH_OPEN, k_open)

            # 1. Contornos directos de las islas de probabilidad
            cnts, _ = cv2.findContours(level_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for c in cnts:
                _register_native_contour(c)

            # 2. Separación con bordes internos de Canny (para embriones que se tocan en racimo)
            cut_mask = cv2.bitwise_and(level_mask, cv2.bitwise_not(edges_canny))
            cut_mask = cv2.morphologyEx(cut_mask, cv2.MORPH_OPEN, k_open)
            cnts_cut, _ = cv2.findContours(cut_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for c in cnts_cut:
                _register_native_contour(c)

            # 3. Watershed controlado por picos de (Distancia * Probabilidad_Píxel)
            dist = cv2.distanceTransform(level_mask, cv2.DIST_L2, 5) * heatmap
            if dist.max() > 1.5:
                for peak_r in (0.35, 0.52):
                    _, sure_fg = cv2.threshold(dist, peak_r * dist.max(), 255, cv2.THRESH_BINARY)
                    sure_fg = sure_fg.astype(np.uint8)
                    sure_bg = cv2.dilate(level_mask, k_open, iterations=2)
                    unknown = cv2.subtract(sure_bg, sure_fg)

                    num_m, markers = cv2.connectedComponents(sure_fg)
                    if num_m > 1:
                        markers = markers + 1
                        markers[unknown == 255] = 0
                        ws = cv2.watershed(work_bgr, markers)
                        for m_id in range(2, num_m + 1):
                            basin = (ws == m_id).astype(np.uint8) * 255
                            w_cnts, _ = cv2.findContours(basin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                            for wc in w_cnts:
                                _register_native_contour(wc)

        # --- B. Corte de Protuberancias Marginales en O(N) sobre la Silueta de la Planta ---
        pad_px = int(work_base * 0.30) + 4
        padded_tissue = np.pad(global_tissue, pad_px, mode='edge')
        for rel_r in (0.06, 0.10, 0.15, 0.22):
            r_px = max(5.0, work_base * rel_r)
            opened_pad = self._fast_distance_opening(padded_tissue, r_px)
            opened = opened_pad[pad_px:pad_px + work_h, pad_px:pad_px + work_w]
            protrusions = cv2.subtract(global_tissue, opened)
            protrusions = cv2.morphologyEx(protrusions, cv2.MORPH_OPEN, k_open)
            cnts, _ = cv2.findContours(protrusions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for c in cnts:
                _register_native_contour(c)

        # --- C. Ventanas Deslizantes Multiescala sobre el Mapa de Calor y Luminancia ---
        for rel_s in self.rel_window_scales:
            win_size = int(np.clip(work_base * rel_s, min_w_side + 6, int(work_base * 0.72)))
            stride = max(12, int(win_size * 0.45))

            for y in range(0, max(1, work_h - win_size + 1), stride):
                y2 = min(work_h, y + win_size)
                for x in range(0, max(1, work_w - win_size + 1), stride):
                    x2 = min(work_w, x + win_size)
                    cur_h, cur_w = y2 - y, x2 - x

                    patch_tissue = global_tissue[y:y2, x:x2]
                    if np.count_nonzero(patch_tissue) < 0.18 * patch_tissue.size:
                        continue

                    patch_heat = heatmap_u8[y:y2, x:x2]
                    if np.max(patch_heat) < 65:
                        continue

                    patch_gray = work_gray[y:y2, x:x2]
                    patch_edge = edges_canny[y:y2, x:x2]

                    _, m_heat = cv2.threshold(patch_heat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    _, m_dark = cv2.threshold(patch_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                    for m_cand in (m_heat, m_dark):
                        m_cand = cv2.bitwise_and(m_cand, patch_tissue)
                        for use_ed in (True, False):
                            m_w = cv2.bitwise_and(m_cand, cv2.bitwise_not(patch_edge)) if use_ed else m_cand
                            m_w = cv2.morphologyEx(m_w, cv2.MORPH_OPEN, k_open)
                            cnts, _ = cv2.findContours(m_w, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                            for c in cnts:
                                _register_native_contour(c, x, y, win_w=cur_w, win_h=cur_h)

        # 1. Deduplicar geometrías antes de extraer las 157/119 características
        unique_geoms = self._deduplicate_geometries(geom_proposals, max_candidates=55)

        # 2. Extraer el vector hiperdimensional en paralelo para cada candidato
        rgba_crops = []
        for item in unique_geoms:
            x1, y1, x2, y2 = item['bbox']
            w, h = x2 - x1, y2 - y1
            abs_cnt = item['polygon']
            local_poly = abs_cnt - np.array([[[x1, y1]]], dtype=np.int32)
            local_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(local_mask, [local_poly], 255)
            crop_rgba = rgba_full[y1:y2, x1:x2].copy()
            crop_rgba[:, :, 3] = local_mask
            rgba_crops.append(crop_rgba)

        extracted_feats = list(executor.map(_pool_extract_worker, rgba_crops, chunksize=8))
        del rgba_crops

        valid_proposals, valid_features = [], []
        for item, feats in zip(unique_geoms, extracted_feats):
            if feats is not None:
                valid_proposals.append(item)
                valid_features.append(feats)

        return valid_proposals, valid_features

    def select_final_detections(self, proposals, gateformer_probs,
                                fused_conf_th=0.60, poly_iou_th=0.20, poly_iomin_th=0.48):
        """
        Combina la probabilidad morfológica/textural de EmbryoGateFormerV2 (Nivel Polígono)
        con la probabilidad contextual de PixelEmbryoSegmenter (Nivel Píxel en la foto)
        y aplica NMS Poligonal para evitar solapamientos.
        """
        if not proposals or len(gateformer_probs) == 0:
            return []

        scored_cands = []
        for prop, p_gate in zip(proposals, gateformer_probs):
            item = prop.copy()
            p_ctx = float(prop['context_prob'])
            # Score fusionado: 65% Red Profunda EmbryoGateFormerV2 + 35% Probabilidad Píxel Contextual
            fused_prob = 0.65 * float(p_gate) + 0.35 * p_ctx
            item['prob'] = fused_prob
            item['gate_prob'] = float(p_gate)
            cx = 0.5 * (prop['bbox'][0] + prop['bbox'][2])
            cy = 0.5 * (prop['bbox'][1] + prop['bbox'][3])
            item['center'] = (cx, cy)
            item['eq_radius'] = np.sqrt(prop['area'] / np.pi)
            scored_cands.append(item)

        scored_cands.sort(key=lambda x: x['prob'], reverse=True)

        # NMS Poligonal Exacto
        non_overlapping = []
        for cand in scored_cands:
            keep = True
            for acc in non_overlapping:
                p_iou, p_iomin = self._polygon_mask_iou_and_iomin(cand, acc)
                dist_c = np.hypot(cand['center'][0] - acc['center'][0], cand['center'][1] - acc['center'][1])
                min_r = min(cand['eq_radius'], acc['eq_radius'])
                if p_iou >= poly_iou_th or p_iomin >= poly_iomin_th or dist_c < (0.58 * min_r):
                    keep = False
                    break
            if keep:
                non_overlapping.append(cand)

        accepted = [
            c for c in non_overlapping
            if c['prob'] >= fused_conf_th and c['gate_prob'] >= 0.45
        ][:7]

        # Garantizar al menos 1 detección en fotos con 1 embrión claro
        if len(accepted) == 0 and len(non_overlapping) > 0:
            accepted = [non_overlapping[0]]

        return accepted


# ==============================================================================
# 5. ORQUESTADOR DE LA FASE 4
# ==============================================================================
class Phase4CountingOrchestrator:
    def __init__(self, json_path, count_images_dir, raw_images_dir, model_dir, output_dir, train_limit=900):
        self.json_path = json_path
        self.count_images_dir = count_images_dir
        self.raw_images_dir = raw_images_dir
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.vis_dir = os.path.join(output_dir, 'visualizaciones_conteo')
        self.train_limit = train_limit

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)

        self.predictor = TrainedModelPredictorCPU(model_dir)
        self.pixel_segmenter = PixelEmbryoSegmenter(model_dir, work_dim=320)
        self.scanner = NativeContourEmbryoScanner(self.pixel_segmenter, work_dim=320)

    @staticmethod
    def _resolve_image_path(task, search_dirs):
        raw_name = task['data']['image'].split('/')[-1]
        clean_name = raw_name.split('-', 1)[-1] if '-' in raw_name else raw_name
        base_name = os.path.splitext(clean_name)[0]

        for s_dir in search_dirs:
            if not os.path.exists(s_dir):
                continue
            for fname in os.listdir(s_dir):
                if os.path.splitext(fname)[0] == base_name:
                    return os.path.join(s_dir, fname), fname
        return None, base_name

    def run_stage_4a_blind_embryo_only_test(self, test_tasks, executor):
        print("\n" + "=" * 75)
        print("  ETAPA 4A: EVALUACIÓN CIEGA EXCLUSIVA SOBRE EMBRIONES ETIQUETADOS (ES)")
        print("=" * 75)
        processor = EmbryoProcessor()
        crops = []

        for task in test_tasks:
            img_path, _ = self._resolve_image_path(task, [self.count_images_dir, self.raw_images_dir])
            if img_path is None:
                continue
            img = _orig_imread(img_path)
            if img is None:
                continue

            img_h, img_w = img.shape[:2]
            enhanced = processor._enhance_illumination(img)
            rgba_img = cv2.cvtColor(enhanced, cv2.COLOR_BGR2BGRA)
            polys = LabelStudioParser.extract_polygons(task)
            valid_polys = [p for p in polys if len(p) >= 3 and cv2.contourArea(p) > 15]

            for poly in valid_polys:
                x, y, w, h = cv2.boundingRect(poly)
                x1, y1, x2, y2 = max(0, x), max(0, y), min(img_w, x + w), min(img_h, y + h)
                if x2 > x1 and y2 > y1:
                    local_poly = poly - np.array([x1, y1])
                    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
                    cv2.fillPoly(mask, [local_poly], 255)
                    crop = rgba_img[y1:y2, x1:x2].copy()
                    crop[:, :, 3] = mask
                    crops.append(crop)

        pos_feats = [f for f in executor.map(_pool_extract_worker, crops, chunksize=16) if f is not None]
        del crops
        gc.collect()

        probs = self.predictor.predict_probabilities(pos_feats)
        recognized_050 = np.sum(probs >= 0.50)
        total_es = len(probs)

        print(f"  Total de Embriones (ES) Reales Evaluados   : {total_es}")
        print(
            f"  Embriones Reconocidos Correctamente        : {recognized_050}/{total_es} ({(recognized_050 / max(1, total_es)) * 100:.2f}%)")
        print(
            f"  Probabilidad Media Asignada por la Red     : {np.mean(probs) * 100:.2f}% (Mediana: {np.median(probs) * 100:.2f}%)")
        print("=" * 75)

    def _draw_and_save_visualization(self, img_bgr, gt_polygons, detections, pred_count, img_name):
        vis = img_bgr.copy()
        for poly in gt_polygons:
            if len(poly) >= 3:
                cv2.polylines(vis, [poly], True, (0, 220, 0), 2)

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cv2.drawContours(vis, [det['polygon']], -1, (255, 50, 255), 2)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 1)
            cv2.putText(
                vis, f"{det['prob']:.2f}", (x1, max(15, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA
            )

        gt_count = len([p for p in gt_polygons if len(p) >= 3])
        err = abs(pred_count - gt_count)
        banner = np.zeros((42, vis.shape[1], 3), dtype=np.uint8)
        text = f"{img_name} | GT Real: {gt_count} ES (Verde) | Conteo IA: {pred_count} ES (Magenta) | Error Abs: {err}"
        cv2.putText(banner, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        final_vis = np.vstack([banner, vis])
        out_path = os.path.join(self.vis_dir, f"CONTEO_{os.path.splitext(img_name)[0]}.png")
        cv2.imwrite(out_path, final_vis)
        del vis, final_vis

    def run_full_pipeline(self):
        t0 = time.time()
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        train_tasks = data[:self.train_limit]
        val_tasks = data[int(self.train_limit * 0.9):self.train_limit]
        test_tasks = data[self.train_limit:]

        print(
            f"[*] Modo Paralelo Autóctono (16 GB RAM): {SAFE_WORKERS}/{TOTAL_CORES} núcleos CPU activos (0% polígonos de otras fotos)")
        self.pixel_segmenter.fit_or_load(
            train_tasks, self._resolve_image_path, [self.raw_images_dir, self.count_images_dir], self.scanner.clahe
        )

        with ProcessPoolExecutor(max_workers=SAFE_WORKERS, initializer=_init_pool_worker) as executor:
            # 1. Etapa 4A: Evaluar exclusivamente los 307 embriones etiquetados (ES)
            self.run_stage_4a_blind_embryo_only_test(test_tasks, executor)

            # 2. Calibrar umbral fusionado sobre 30 fotos estratificadas de Validación
            print("\n[*] Calibrando umbral de fusión (Píxel + EmbryoGateFormerV2) en Validación...")
            val_sample = val_tasks[::3][:30]
            val_cache = []
            for idx, task in enumerate(val_sample, 1):
                img_path, _ = self._resolve_image_path(task, [self.raw_images_dir, self.count_images_dir])
                if img_path is None:
                    continue
                img = _orig_imread(img_path)
                if img is None:
                    continue
                gt_polys = [p for p in LabelStudioParser.extract_polygons(task) if len(p) >= 3]
                valid_props, feats = self.scanner.extract_candidates_from_image(img, executor)
                probs = self.predictor.predict_probabilities(feats)
                val_cache.append((len(gt_polys), valid_props, probs))
                del img, feats
                if idx % 10 == 0:
                    print(
                        f"    -> Calibración en Validación: {idx}/{len(val_sample)} fotos procesadas ({time.time() - t0:.1f}s)...")

            best_conf_th = 0.58
            best_loss = float('inf')
            if val_cache:
                total_val_gt = sum(gt_c for gt_c, _, _ in val_cache)
                for c_th in np.linspace(0.48, 0.76, 29):
                    preds = [
                        len(self.scanner.select_final_detections(vp, pr, fused_conf_th=c_th))
                        for _, vp, pr in val_cache
                    ]
                    mae = np.mean([abs(p - g) for p, (g, _, _) in zip(preds, val_cache)])
                    vol_diff = abs(sum(preds) - total_val_gt) / max(1.0, len(val_cache))
                    loss = mae + 0.45 * vol_diff
                    if loss < best_loss:
                        best_loss = loss
                        best_conf_th = float(c_th)

            del val_cache
            gc.collect()
            print(f"[*] Umbral de fusión calibrado -> τ_fused = {best_conf_th:.3f}")

            # 3. Conteo Final y Guardado de las 100 Imágenes Reservadas
            print(
                f"\n[*] Iniciando Conteo en las {len(test_tasks)} imágenes de Test Final y guardando las 100 visualizaciones...")
            records = []

            for idx, task in enumerate(test_tasks, 1):
                img_path, fname = self._resolve_image_path(task, [self.count_images_dir, self.raw_images_dir])
                if img_path is None:
                    continue
                img = _orig_imread(img_path)
                if img is None:
                    continue

                gt_polys = [p for p in LabelStudioParser.extract_polygons(task) if len(p) >= 3]
                gt_count = len(gt_polys)

                valid_props, feats = self.scanner.extract_candidates_from_image(img, executor)
                probs = self.predictor.predict_probabilities(feats)

                detections = self.scanner.select_final_detections(
                    valid_props, probs, fused_conf_th=best_conf_th
                )

                pred_count = len(detections)
                abs_err = abs(pred_count - gt_count)

                # Guardar las 100 visualizaciones completas
                self._draw_and_save_visualization(img, gt_polys, detections, pred_count, fname)

                records.append({
                    'image_name': fname,
                    'candidatos_evaluados': len(valid_props),
                    'conteo_real_GT': gt_count,
                    'conteo_predicho_IA': pred_count,
                    'error_absoluto': abs_err,
                    'error_relativo_pct': (abs_err / max(1, gt_count)) * 100.0
                })

                del img, valid_props, feats, detections
                if idx % 20 == 0:
                    gc.collect()

                if idx % 5 == 0 or idx == len(test_tasks):
                    curr_mae = np.mean([r['error_absoluto'] for r in records])
                    print(
                        f"    -> [{idx:3d}/{len(test_tasks)}] {fname:<12} | Real: {gt_count:2d} | IA: {pred_count:2d} | MAE acumulado: {curr_mae:.2f} ({time.time() - t0:.1f}s)")

        df_report = pd.DataFrame(records)
        csv_path = os.path.join(self.output_dir, 'reporte_conteo_100_imagenes.csv')
        df_report.to_csv(csv_path, index=False)

        y_gt = df_report['conteo_real_GT'].values
        y_pr = df_report['conteo_predicho_IA'].values
        mae = mean_absolute_error(y_gt, y_pr)
        rmse = np.sqrt(mean_squared_error(y_gt, y_pr))
        r2 = r2_score(y_gt, y_pr) if len(y_gt) > 1 else 0.0
        counting_acc = max(0.0, (1.0 - np.sum(np.abs(y_gt - y_pr)) / (np.sum(y_gt) + 1e-6)) * 100.0)
        exact_or_off_by_1 = np.mean(np.abs(y_gt - y_pr) <= 1) * 100.0

        print("\n" + "=" * 75)
        print("  REPORTE FINAL DE CUANTIFICACIÓN EN LAS 100 IMÁGENES DE PRUEBA (FASE 4)")
        print("=" * 75)
        print(f"  Tiempo total de ejecución           : {time.time() - t0:.1f} segundos")
        print(f"  Imágenes completas evaluadas        : {len(df_report)}")
        print(f"  Total Embriones Somáticos Reales    : {np.sum(y_gt)}")
        print(f"  Total Embriones Contados por la IA  : {np.sum(y_pr)}")
        print(f"  Exactitud Global de Conteo (Volumen): {counting_acc:.2f}%")
        print(f"  Fotos con Error <= 1 embrión        : {exact_or_off_by_1:.1f}% de las imágenes")
        print(f"  MAE  (Error Absoluto Medio / foto)  : {mae:.2f} embriones")
        print(f"  RMSE (Error Cuadrático Medio)       : {rmse:.2f} embriones")
        print(f"  Coeficiente de Determinación (R²)   : {r2:.4f}")
        print("=" * 75)
        print(f"[*] Tabla detallada guardada en      : {csv_path}")
        print(f"[*] 100 imágenes anotadas en         : {self.vis_dir}")


if __name__ == "__main__":
    JSON_PATH = 'project-3-at-2026-10-14-11-48-9adb768b.json'
    COUNT_IMAGES_DIR = './Fase 1 Segmentacion/2_final_counting_Images'
    RAW_IMAGES_DIR = './Fotografías selectas'
    MODEL_DIR = './Fase 3 Modelado'
    OUTPUT_DIR = './Fase 4 Conteo Final'

    orchestrator = Phase4CountingOrchestrator(
        json_path=JSON_PATH,
        count_images_dir=COUNT_IMAGES_DIR,
        raw_images_dir=RAW_IMAGES_DIR,
        model_dir=MODEL_DIR,
        output_dir=OUTPUT_DIR,
        train_limit=900
    )
    orchestrator.run_full_pipeline()
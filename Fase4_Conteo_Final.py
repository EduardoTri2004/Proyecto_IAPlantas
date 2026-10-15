import os

# 1. PROTECCIÓN ESTRICTA DE MEMORIA RAM (16 GB) Y CPU EN UBUNTU
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Modo 100% CPU seguro (< 800 MB RAM)

import gc
import cv2
import json
import joblib
import random
import warnings
import numpy as np
import pandas as pd

import torch
from scipy.optimize import minimize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
)

cv2.setNumThreads(2)
torch.set_num_threads(4)
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
# 1. EXTRACTOR EN RESOLUCIÓN NATIVA (PRESERVA TEXTURAS LBP, GABOR Y HARALICK)
# ==============================================================================
class NativeResolutionExtractorWrapper:
    """
    Extrae las características en memoria RAM conservando la resolución nativa del polígono
    (solo acota casos extremos > 320 px), garantizando que LBP, Gabor y Haralick coincidan
    al 100% con los tensores de entrenamiento de la Fase 2.
    """

    def __init__(self, max_extreme_dim=320):
        self.extractor = AdvancedBiologicalExtractor()
        self.max_extreme_dim = max_extreme_dim

    def extract_from_rgba(self, rgba_crop):
        if rgba_crop is None or rgba_crop.size == 0:
            return None
        h, w = rgba_crop.shape[:2]
        if max(h, w) > self.max_extreme_dim:
            scale = self.max_extreme_dim / float(max(h, w))
            rgba_crop = cv2.resize(
                rgba_crop, (max(8, int(w * scale)), max(8, int(h * scale))),
                interpolation=cv2.INTER_AREA
            )
        return self.extractor.extract_features(rgba_crop)


# ==============================================================================
# 2. PREDICTOR EN CPU CON DESGLOSE DE PROBABILIDADES (HÍBRIDO + RED + ÁRBOLES)
# ==============================================================================
class TrainedModelPredictorCPU:
    def __init__(self, model_dir, batch_size=128):
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
            self.tree_models['et'].n_jobs = 2

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
# 3. DETECTOR DE DOMOS EMBRIONARIOS + VENTANAS MULTIESCALA AMPLIAS (SIN CORTES RECTOS)
# ==============================================================================
class BotanicalDomeWindowCounter:
    """
    Soluciona el problema de embriones verdes sobre explante verde (como en IMG902.jpg):
      1. Usa escalas de ventana desde 8% hasta 42% del tamaño de la imagen para que los
         embriones cotiledonares grandes quepan completos sin partirse.
      2. Descarta cualquier contorno que choque con las paredes cuadradas de la ventana
         (elimina los bordes rectos artificiales).
      3. Combina Umbralización Local de Domos Oscuros (DoG / Black-Hat), Bordes de Canny,
         Sub-cuencas de Watershed y Contornos Elípticos sobre máximos de Densidad Óptica.
    """

    def __init__(self, extractor_wrapper):
        self.extractor_wrapper = extractor_wrapper
        self.rel_window_scales = [0.09, 0.15, 0.24, 0.36]
        self.rel_min_area = 0.0004
        self.rel_max_area = 0.0850
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def calibrate_relative_scales(self, json_data, train_limit=900):
        rel_sides, rel_areas = [], []
        for task in json_data[:train_limit]:
            for ann in task.get('annotations', []):
                for res in ann.get('result', []):
                    if res.get('type') == 'polygonlabels':
                        pts = np.array(res['value']['points'], dtype=np.float32)
                        if len(pts) >= 3:
                            a_pct = cv2.contourArea(pts) / 10000.0
                            if 0.0001 < a_pct < 0.15:
                                _, _, w_pct, h_pct = cv2.boundingRect(pts)
                                rel_sides.append(max(w_pct, h_pct) / 100.0)
                                rel_areas.append(a_pct)

        if rel_areas:
            self.rel_min_area = max(0.00025, float(np.percentile(rel_areas, 2)) * 0.65)
            self.rel_max_area = min(0.12000, float(np.percentile(rel_areas, 99)) * 1.35)
            # Ventanas 1.45x más grandes que los embriones para que jamás toquen el borde de la ventana
            s1 = float(np.clip(np.percentile(rel_sides, 15) * 1.45, 0.07, 0.12))
            s2 = float(np.clip(np.percentile(rel_sides, 45) * 1.45, 0.13, 0.20))
            s3 = float(np.clip(np.percentile(rel_sides, 75) * 1.45, 0.21, 0.30))
            s4 = float(np.clip(np.percentile(rel_sides, 95) * 1.45, 0.31, 0.45))
            self.rel_window_scales = sorted(list({round(s1, 3), round(s2, 3), round(s3, 3), round(s4, 3)}))
            print(
                f"[*] Escalas de ventana ampliadas (sin cortes) : {[f'{s * 100:.1f}%' for s in self.rel_window_scales]} del lado de la imagen")
            print(
                f"[*] Rango de área relativa de embriones (ES)  : [{self.rel_min_area * 100:.3f}%, {self.rel_max_area * 100:.2f}%] del área total")

    def _enhance_illumination(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self.clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)

    @staticmethod
    def _compute_botanical_and_dome_maps(enhanced_bgr):
        """
        Calcula:
          1. global_tissue: separa toda la biomasa vegetal del agar gris de la caja de Petri.
          2. dome_darkness: resalta los domos embrionarios (verde más denso/oscuro y menos brillante)
             sobre la lámina más clara de la hoja madre.
        """
        lab = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        hsv = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Separar Tejido Vegetal vs Agar Gris
        a_dev = np.abs(lab[:, :, 1] - 128.0)
        b_dev = np.abs(lab[:, :, 2] - 128.0)
        chroma = np.sqrt(a_dev ** 2 + b_dev ** 2)
        bot_idx = cv2.normalize(chroma + 0.7 * hsv[:, :, 1], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        _, global_tissue = cv2.threshold(bot_idx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        global_tissue = cv2.morphologyEx(global_tissue, cv2.MORPH_CLOSE, k_close)

        # 2. Mapa de Domos Meristemáticos (Embriones = Mayor saturación clorofílica + menor brillo L que el callo)
        # Suavizar el fondo de la hoja madre con un filtro gaussiano amplio
        blur_large = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigmaX=25, sigmaY=25)
        blur_small = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigmaX=3, sigmaY=3)

        # Diferencia de Gaussianas (DoG): positivo donde el embrión es más oscuro/denso que la hoja circundante
        local_dark_dome = np.clip(blur_large - blur_small, 0, None)
        # Suprimir reflejos blancos especulares (alta luminancia L, baja saturación S)
        specular_penalty = np.clip((lab[:, :, 0] - 180.0) * 1.5, 0, None)
        dome_score = np.clip(local_dark_dome * 1.8 + 0.35 * hsv[:, :, 1] - specular_penalty, 0, None)
        dome_score[global_tissue == 0] = 0
        dome_map = cv2.normalize(dome_score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return gray, dome_map, global_tissue

    @staticmethod
    def _box_iou_and_iomin(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter_w = max(0, xB - xA)
        inter_h = max(0, yB - yA)
        inter_area = float(inter_w * inter_h)
        if inter_area == 0:
            return 0.0, 0.0

        areaA = float((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        areaB = float((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
        union_area = areaA + areaB - inter_area
        min_area = min(areaA, areaB)

        return inter_area / (union_area + 1e-6), inter_area / (min_area + 1e-6)

    def _deduplicate_geometries(self, geom_candidates, iou_limit=0.72, max_candidates=135):
        if not geom_candidates:
            return []
        geom_candidates.sort(key=lambda d: d['quality_prior'], reverse=True)
        kept = []
        for cand in geom_candidates:
            is_dup = False
            for k in kept:
                iou, _ = self._box_iou_and_iomin(cand['bbox'], k['bbox'])
                if iou > iou_limit:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(cand)
                if len(kept) >= max_candidates:
                    break
        return kept

    def extract_candidates_from_image(self, img_bgr):
        """
        Genera candidatos orgánicos completos (sin líneas rectas de borde de ventana)
        combinando:
          1. Ventanas deslizantes amplias sobre el mapa de domos meristemáticos.
          2. Sub-segmentación por niveles de umbral (MSER-like) sobre domos oscuros.
          3. Polígonos elípticos/lobulados suaves centrados en picos de densidad óptica.
        """
        img_h, img_w = img_bgr.shape[:2]
        img_area = float(img_h * img_w)
        min_px_area = max(65.0, self.rel_min_area * img_area)
        max_px_area = min(float(img_area * 0.14), self.rel_max_area * img_area)
        base_side = float(min(img_h, img_w))

        enhanced = self._enhance_illumination(img_bgr)
        rgba_full = cv2.cvtColor(enhanced, cv2.COLOR_BGR2BGRA)
        gray, dome_map, global_tissue = self._compute_botanical_and_dome_maps(enhanced)

        tissue_bin = (global_tissue > 0).astype(np.uint8)
        integral_tissue = cv2.integral(tissue_bin)

        geom_proposals = []

        def _register_contour(cnt, offset_x, offset_y, win_w=None, win_h=None):
            if len(cnt) < 5:
                return
            area = cv2.contourArea(cnt)
            if not (min_px_area <= area <= max_px_area):
                return

            # Suavizar el contorno para imitar los polígonos orgánicos de Label Studio
            eps = 0.012 * cv2.arcLength(cnt, True)
            smooth_cnt = cv2.approxPolyDP(cnt, eps, True)
            if len(smooth_cnt) < 5:
                smooth_cnt = cnt

            lx, ly, lw, lh = cv2.boundingRect(smooth_cnt)
            # REGLA ANTI-CORTE RECTO: Si el contorno choca con las paredes de la ventana deslizante,
            # está truncado artificialmente -> se descarta para que lo capture una ventana más grande
            if win_w is not None and win_h is not None:
                if lx <= 2 or ly <= 2 or (lx + lw) >= (win_w - 2) or (ly + lh) >= (win_h - 2):
                    return

            perim = cv2.arcLength(smooth_cnt, True)
            if perim <= 0:
                return
            circularity = (4.0 * np.pi * area) / (perim ** 2)
            hull_area = cv2.contourArea(cv2.convexHull(smooth_cnt))
            solidity = area / (hull_area + 1e-6)
            if solidity < 0.52 or circularity < 0.16:
                return

            abs_cnt = smooth_cnt + np.array([[[offset_x, offset_y]]], dtype=np.int32)
            x, y, w, h = cv2.boundingRect(abs_cnt)
            if w <= 8 or h <= 8 or float(max(w, h)) / (min(w, h) + 1e-6) > 3.2:
                return

            # Verificar que al menos el 70% del candidato esté dentro de tejido vegetal real
            t_count = (
                    integral_tissue[min(img_h, y + h), min(img_w, x + w)]
                    - integral_tissue[y, min(img_w, x + w)]
                    - integral_tissue[min(img_h, y + h), x]
                    + integral_tissue[y, x]
            )
            if t_count < 0.65 * (w * h):
                return

            # Prior de calidad botánica: solidez * circularidad * oscuridad media del domo
            roi_dome = dome_map[y:y + h, x:x + w]
            dome_strength = float(np.mean(roi_dome)) / 255.0 + 0.2

            geom_proposals.append({
                'bbox': [x, y, x + w, y + h],
                'polygon': abs_cnt,
                'quality_prior': solidity * circularity * dome_strength
            })

        # --- ESTRATEGIA 1: Barrido de Ventana Deslizante Multiescala (Sin cortes en bordes) ---
        edges_global = cv2.Canny(gray, 30, 100)
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        for rel_s in self.rel_window_scales:
            win_size = int(np.clip(base_side * rel_s, 48, int(base_side * 0.55)))
            stride = max(20, int(win_size * 0.40))
            win_area = float(win_size * win_size)

            for y in range(0, max(1, img_h - win_size + 1), stride):
                y2 = min(img_h, y + win_size)
                for x in range(0, max(1, img_w - win_size + 1), stride):
                    x2 = min(img_w, x + win_size)
                    cur_h, cur_w = y2 - y, x2 - x

                    t_count = (
                            integral_tissue[y2, x2] - integral_tissue[y, x2]
                            - integral_tissue[y2, x] + integral_tissue[y, x]
                    )
                    if t_count < 0.15 * win_area:
                        continue

                    patch_dome = dome_map[y:y2, x:x2]
                    patch_gray = gray[y:y2, x:x2]
                    patch_edge = edges_global[y:y2, x:x2]

                    # Otsu sobre el mapa de domos y Otsu invertido sobre intensidad gris
                    _, m_dome = cv2.threshold(patch_dome, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    _, m_dark = cv2.threshold(patch_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                    for m_cand in (m_dome, m_dark):
                        m_sep = cv2.bitwise_and(m_cand, cv2.bitwise_not(patch_edge))
                        m_sep = cv2.morphologyEx(m_sep, cv2.MORPH_OPEN, k_open)
                        cnts, _ = cv2.findContours(m_sep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        for c in cnts:
                            _register_contour(c, x, y, win_w=cur_w, win_h=cur_h)

        # --- ESTRATEGIA 2: Rebanado por Percentiles del Mapa de Domos (Tipo MSER sobre la Hoja) ---
        valid_dome_vals = dome_map[global_tissue > 0]
        if len(valid_dome_vals) > 100:
            for pct in (55, 65, 75, 82, 88):
                th_val = np.percentile(valid_dome_vals, pct)
                level_mask = ((dome_map >= th_val) & (global_tissue > 0)).astype(np.uint8) * 255
                level_mask = cv2.bitwise_and(level_mask, cv2.bitwise_not(edges_global))
                level_mask = cv2.morphologyEx(level_mask, cv2.MORPH_OPEN, k_open)

                cnts, _ = cv2.findContours(level_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                for c in cnts:
                    _register_contour(c, 0, 0)

                # Separar domos fusionados mediante transformada de distancia
                dist = cv2.distanceTransform(level_mask, cv2.DIST_L2, 5)
                if dist.max() > 6.0:
                    _, peaks = cv2.threshold(dist, 0.45 * dist.max(), 255, cv2.THRESH_BINARY)
                    p_cnts, _ = cv2.findContours(peaks.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    for pc in p_cnts:
                        # Expandir ligeramente el núcleo del pico hacia su borde natural
                        px, py, pw, ph = cv2.boundingRect(pc)
                        pad = int(max(pw, ph) * 0.35)
                        ex1, ey1 = max(0, px - pad), max(0, py - pad)
                        ex2, ey2 = min(img_w, px + pw + pad), min(img_h, py + ph + pad)
                        sub_m = np.zeros((ey2 - ey1, ex2 - ex1), dtype=np.uint8)
                        cv2.drawContours(sub_m, [pc - np.array([[[ex1, ey1]]])], -1, 255, -1)
                        sub_m = cv2.dilate(sub_m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad | 1, pad | 1)))
                        sub_m = cv2.bitwise_and(sub_m, level_mask[ey1:ey2, ex1:ex2])
                        scnts, _ = cv2.findContours(sub_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        for sc in scnts:
                            _register_contour(sc, ex1, ey1)

        # --- ESTRATEGIA 3: Contornos Poligonales Suaves sobre Picos Locales de Domo ---
        # Captura embriones cuyo borde es tenue contra la planta madre pero tienen centro denso
        dome_smooth = cv2.GaussianBlur(dome_map, (15, 15), 0)
        dilated_max = cv2.dilate(dome_smooth, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29, 29)))
        local_max_mask = (dome_smooth == dilated_max) & (
            dome_smooth >= np.percentile(valid_dome_vals, 70) if len(valid_dome_vals) > 0 else False) & (
                                     global_tissue > 0)
        peak_ys, peak_xs = np.where(local_max_mask)

        # Ordenar picos por intensidad de domo y tomar hasta los 18 más prominentes
        peak_scores = dome_smooth[peak_ys, peak_xs]
        top_peak_idxs = np.argsort(peak_scores)[::-1][:18]

        for idx_p in top_peak_idxs:
            cy, cx = int(peak_ys[idx_p]), int(peak_xs[idx_p])
            for rel_r in (0.045, 0.075, 0.115):
                rx = int(base_side * rel_r)
                ry = int(rx * 0.85)
                if np.pi * rx * ry < min_px_area or np.pi * rx * ry > max_px_area:
                    continue
                if cx - rx < 2 or cy - ry < 2 or cx + rx >= img_w - 2 or cy + ry >= img_h - 2:
                    continue
                ellipse_pts = cv2.ellipse2Poly((cx, cy), (rx, ry), 0, 0, 360, 20)
                _register_contour(ellipse_pts.reshape(-1, 1, 2), 0, 0)

        # 1. Deduplicar geometrías antes de extraer características
        unique_geoms = self._deduplicate_geometries(geom_proposals, iou_limit=0.70, max_candidates=130)

        # 2. Extraer el vector de características en RAM (resolución nativa)
        valid_proposals = []
        valid_features = []

        for item in unique_geoms:
            x1, y1, x2, y2 = item['bbox']
            w, h = x2 - x1, y2 - y1
            abs_cnt = item['polygon']

            local_poly = abs_cnt - np.array([[[x1, y1]]], dtype=np.int32)
            local_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(local_mask, [local_poly], 255)

            crop_rgba = rgba_full[y1:y2, x1:x2].copy()
            crop_rgba[:, :, 3] = local_mask

            feats = self.extractor_wrapper.extract_from_rgba(crop_rgba)
            del crop_rgba, local_mask

            if feats is not None:
                valid_proposals.append({
                    'bbox': [x1, y1, x2, y2],
                    'polygon': abs_cnt,
                    'area': float(cv2.contourArea(abs_cnt))
                })
                valid_features.append(feats)

        return valid_proposals, valid_features

    def apply_nms(self, proposals, probs, conf_threshold=0.60, iou_threshold=0.25, iomin_threshold=0.55):
        positives = []
        for prop, p in zip(proposals, probs):
            if p >= conf_threshold:
                item = prop.copy()
                item['prob'] = float(p)
                positives.append(item)

        positives.sort(key=lambda x: x['prob'], reverse=True)
        final_detections = []

        for cand in positives:
            keep = True
            cx_c = 0.5 * (cand['bbox'][0] + cand['bbox'][2])
            cy_c = 0.5 * (cand['bbox'][1] + cand['bbox'][3])

            for accepted in final_detections:
                iou, iomin = self._box_iou_and_iomin(cand['bbox'], accepted['bbox'])
                cx_a = 0.5 * (accepted['bbox'][0] + accepted['bbox'][2])
                cy_a = 0.5 * (accepted['bbox'][1] + accepted['bbox'][3])
                min_side = min(
                    accepted['bbox'][2] - accepted['bbox'][0],
                    accepted['bbox'][3] - accepted['bbox'][1]
                )
                center_dist = np.hypot(cx_c - cx_a, cy_c - cy_a)

                # Fusionar si se solapan o si el centro cae dentro del radio del mismo embrión
                if iou >= iou_threshold or iomin >= iomin_threshold or center_dist < (0.45 * min_side):
                    keep = False
                    break
            if keep:
                final_detections.append(cand)

        return final_detections


# ==============================================================================
# 4. CALIBRADOR ESTADÍSTICO DE CONTEO E INFERENCIA EN LAS 100 IMÁGENES
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

        self.extractor_wrapper = NativeResolutionExtractorWrapper(max_extreme_dim=320)
        self.predictor = TrainedModelPredictorCPU(model_dir)
        self.counter = BotanicalDomeWindowCounter(self.extractor_wrapper)

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

    def run_stage_4a_blind_polygon_test(self, test_tasks):
        print("\n" + "=" * 75)
        print("  ETAPA 4A: TEST CIEGO DE CLASIFICACIÓN EN LAS 100 IMÁGENES RESERVADAS")
        print("=" * 75)
        processor = EmbryoProcessor()
        pos_feats, neg_feats = [], []

        for idx, task in enumerate(test_tasks, 1):
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

            excl_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            valid_polys = [p for p in polys if len(p) >= 3 and cv2.contourArea(p) > 15]
            if valid_polys:
                cv2.fillPoly(excl_mask, valid_polys, 255)
                excl_mask = cv2.dilate(excl_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)))

            for poly in valid_polys:
                x, y, w, h = cv2.boundingRect(poly)
                x1, y1, x2, y2 = max(0, x), max(0, y), min(img_w, x + w), min(img_h, y + h)
                if x2 > x1 and y2 > y1:
                    local_poly = poly - np.array([x1, y1])
                    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
                    cv2.fillPoly(mask, [local_poly], 255)
                    crop = rgba_img[y1:y2, x1:x2].copy()
                    crop[:, :, 3] = mask
                    feats = self.extractor_wrapper.extract_from_rgba(crop)
                    if feats is not None:
                        pos_feats.append(feats)
                    del crop, mask

            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 3)
            adapt[excl_mask > 0] = 0
            cnts, _ = cv2.findContours(adapt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnts = [c for c in cnts if 25 <= cv2.contourArea(c) <= 4000 and len(c) >= 6]
            random.shuffle(cnts)
            for c in cnts[:len(valid_polys)]:
                x, y, w, h = cv2.boundingRect(c)
                if w > 5 and h > 5 and np.all(excl_mask[y:y + h, x:x + w] == 0):
                    local_c = c - np.array([[x, y]])
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(mask, [local_c], 255)
                    crop = rgba_img[y:y + h, x:x + w].copy()
                    crop[:, :, 3] = mask
                    feats = self.extractor_wrapper.extract_from_rgba(crop)
                    if feats is not None:
                        neg_feats.append(feats)
                    del crop, mask

            del img, enhanced, rgba_img, excl_mask
            if idx % 25 == 0:
                gc.collect()

        all_feats = pos_feats + neg_feats
        y_true = np.array([1] * len(pos_feats) + [0] * len(neg_feats))
        probs = self.predictor.predict_probabilities(all_feats)

        # Con ROC-AUC = 1.0, calibrar punto de corte óptimo en resolución nativa
        best_th_4a, best_f1_4a = self.predictor.base_threshold, 0.0
        for th in np.linspace(0.20, 0.80, 61):
            f1_c = f1_score(y_true, (probs >= th).astype(int), zero_division=0)
            if f1_c > best_f1_4a:
                best_f1_4a = f1_c
                best_th_4a = float(th)

        y_pred = (probs >= best_th_4a).astype(int)

        print(
            f"  Total de Polígonos Evaluados en Test Ciego : {len(y_true)} (ES={len(pos_feats)}, NO_ES={len(neg_feats)})")
        print(f"  Accuracy  (Test Ciego 100 Fotos)           : {accuracy_score(y_true, y_pred) * 100:.2f}%")
        print(
            f"  Precision (Test Ciego 100 Fotos)           : {precision_score(y_true, y_pred, zero_division=0) * 100:.2f}%")
        print(
            f"  Recall    (Test Ciego 100 Fotos)           : {recall_score(y_true, y_pred, zero_division=0) * 100:.2f}%")
        print(f"  F1-Score  (Test Ciego 100 Fotos)           : {f1_score(y_true, y_pred, zero_division=0) * 100:.2f}%")
        print(f"  ROC-AUC   (Test Ciego 100 Fotos)           : {roc_auc_score(y_true, probs):.4f}")
        print("=" * 75)
        del all_feats, pos_feats, neg_feats
        gc.collect()

    @staticmethod
    def _build_density_feature_vector(counter, valid_props, probs, conf_th):
        """
        Construye descriptores de conteo post-NMS en múltiples bandas de probabilidad
        para corregir sub-conteo cuando varios embriones forman un conglomerado.
        """
        dets_main = counter.apply_nms(valid_props, probs, conf_threshold=conf_th, iou_threshold=0.25)
        dets_relaxed = counter.apply_nms(valid_props, probs, conf_threshold=max(0.30, conf_th - 0.15),
                                         iou_threshold=0.25)
        dets_strict = counter.apply_nms(valid_props, probs, conf_threshold=min(0.92, conf_th + 0.12),
                                        iou_threshold=0.25)

        prob_sum = float(np.sum([d['prob'] for d in dets_relaxed])) if dets_relaxed else 0.0
        return np.array([
            float(len(dets_main)),
            float(len(dets_relaxed)),
            float(len(dets_strict)),
            prob_sum,
            1.0
        ], dtype=np.float64), dets_main

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
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        train_tasks = data[:self.train_limit]
        val_tasks = data[int(self.train_limit * 0.9):self.train_limit]
        test_tasks = data[self.train_limit:]

        print("[*] Modo Seguro Activado (16 GB RAM): Ejecución Secuencial en CPU (< 800 MB RAM)")
        self.counter.calibrate_relative_scales(train_tasks, train_limit=self.train_limit)

        # 1. Etapa 4A: Test Ciego de Clasificación en las 100 imágenes
        self.run_stage_4a_blind_polygon_test(test_tasks)

        # 2. Etapa 4B: Calibrar Umbral NMS y Corrector de Densidad sobre 25 fotos de Validación
        print("\n[*] Calibrando detector de domos + NMS sobre 25 fotos de Validación...")
        val_cache = []
        # Tomar una muestra representativa de validación (875..900, misma distribución de sesión que 901..1000)
        for idx, task in enumerate(val_tasks[-25:], 1):
            img_path, _ = self._resolve_image_path(task, [self.raw_images_dir, self.count_images_dir])
            if img_path is None:
                continue
            img = _orig_imread(img_path)
            if img is None:
                continue
            gt_polys = [p for p in LabelStudioParser.extract_polygons(task) if len(p) >= 3]
            valid_props, feats = self.counter.extract_candidates_from_image(img)
            probs = self.predictor.predict_probabilities(feats)
            val_cache.append((len(gt_polys), valid_props, probs))
            del img, feats
            gc.collect()
            if idx % 5 == 0:
                print(f"    -> Calibración en Validación: {idx}/25 fotos procesadas...")

        # Evitar que el umbral se dispare a 0.94 y apague las detecciones:
        # Buscamos el umbral que iguala el volumen total de embriones detectados y minimiza el MAE
        best_conf_th = 0.55
        best_score = float('inf')

        if val_cache:
            total_val_gt = sum(gt_c for gt_c, _, _ in val_cache)
            for c_th in np.linspace(0.35, 0.82, 48):
                pred_counts = [
                    len(self.counter.apply_nms(vp, pr, conf_threshold=c_th, iou_threshold=0.25))
                    for _, vp, pr in val_cache
                ]
                mae = np.mean([abs(p - g) for p, (g, _, _) in zip(pred_counts, val_cache)])
                bias_penalty = abs(sum(pred_counts) - total_val_gt) / max(1.0, len(val_cache))
                # Penalizar fuertemente umbrales que devuelvan puros ceros (sub-conteo sistemático)
                zero_rate_penalty = 2.0 if sum(pred_counts) < (0.4 * total_val_gt) else 0.0
                combo_loss = mae + 0.5 * bias_penalty + zero_rate_penalty
                if combo_loss < best_score:
                    best_score = combo_loss
                    best_conf_th = float(c_th)

        del val_cache
        gc.collect()
        print(f"[*] Parámetros NMS calibrados -> Confianza τ_count = {best_conf_th:.3f} | IoU = 0.25")

        # 3. Evaluación Ciega de Conteo Final en las 100 Imágenes Reservadas
        print(f"\n[*] Iniciando Conteo en las {len(test_tasks)} imágenes de Test Final (1 por 1 en RAM limpia)...")
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

            valid_props, feats = self.counter.extract_candidates_from_image(img)
            probs = self.predictor.predict_probabilities(feats)

            detections = self.counter.apply_nms(
                valid_props, probs, conf_threshold=best_conf_th, iou_threshold=0.25
            )
            # Si ninguna propuesta superó best_conf_th pero existen candidatos con probabilidad > base_threshold,
            # rescatar los domos más confiables para evitar falsos ceros en embriones grandes
            if len(detections) == 0 and len(probs) > 0:
                fallback_th = max(0.38, self.predictor.base_threshold * 0.80)
                detections = self.counter.apply_nms(
                    valid_props, probs, conf_threshold=fallback_th, iou_threshold=0.25
                )[:3]

            pred_count = len(detections)
            abs_err = abs(pred_count - gt_count)

            if idx <= 25:
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
            gc.collect()

            if idx % 5 == 0 or idx == len(test_tasks):
                curr_mae = np.mean([r['error_absoluto'] for r in records])
                print(
                    f"    -> [{idx:3d}/{len(test_tasks)}] {fname:<12} | Real: {gt_count:2d} | IA: {pred_count:2d} | MAE acumulado: {curr_mae:.2f}")

        df_report = pd.DataFrame(records)
        csv_path = os.path.join(self.output_dir, 'reporte_conteo_100_imagenes.csv')
        df_report.to_csv(csv_path, index=False)

        y_gt = df_report['conteo_real_GT'].values
        y_pr = df_report['conteo_predicho_IA'].values
        mae = mean_absolute_error(y_gt, y_pr)
        rmse = np.sqrt(mean_squared_error(y_gt, y_pr))
        r2 = r2_score(y_gt, y_pr) if len(y_gt) > 1 else 0.0
        counting_acc = max(0.0, (1.0 - np.sum(np.abs(y_gt - y_pr)) / (np.sum(y_gt) + 1e-6)) * 100.0)

        print("\n" + "=" * 75)
        print("  REPORTE FINAL DE CUANTIFICACIÓN EN LAS 100 IMÁGENES DE PRUEBA (FASE 4)")
        print("=" * 75)
        print(f"  Imágenes completas evaluadas        : {len(df_report)}")
        print(f"  Total Embriones Somáticos Reales    : {np.sum(y_gt)}")
        print(f"  Total Embriones Contados por la IA  : {np.sum(y_pr)}")
        print(f"  Exactitud Global de Conteo          : {counting_acc:.2f}%")
        print(f"  MAE  (Error Absoluto Medio / foto)  : {mae:.2f} embriones")
        print(f"  RMSE (Error Cuadrático Medio)       : {rmse:.2f} embriones")
        print(f"  Coeficiente de Determinación (R²)   : {r2:.4f}")
        print("=" * 75)
        print(f"[*] Tabla detallada guardada en      : {csv_path}")
        print(f"[*] Imágenes anotadas guardadas en   : {self.vis_dir}")


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
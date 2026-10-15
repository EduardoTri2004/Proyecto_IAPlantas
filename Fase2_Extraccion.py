import os
import cv2
import glob
import warnings
import numpy as np
import pandas as pd
import mahotas
from skimage.feature import local_binary_pattern
from scipy.stats import skew, kurtosis, entropy
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings('ignore')


class AdvancedBiologicalExtractor:
    """
    Extrae un vector hiperdimensional de 157 características organizadas en 15 familias
    físicas, geométricas, ópticas, espectrales e histológicas (0 NaNs garantizados).
    """

    def __init__(self):
        self.lbp_radius = 2
        self.lbp_points = 8 * self.lbp_radius
        self.lbp_bins = self.lbp_points + 2  # 18 bins fijos
        self.gabor_FilterBank = self._build_gabor_bank()

    def _build_gabor_bank(self):
        bank = []
        for sigma in (1.5, 3.0):
            for lambd in (5.0, 10.0):
                orientations = []
                for theta in np.arange(0, np.pi, np.pi / 4):
                    kernel = cv2.getGaborKernel(
                        (11, 11), sigma, theta, lambd, 0.5, 0, ktype=cv2.CV_32F
                    )
                    orientations.append(kernel)
                bank.append(orientations)
        return bank

    def _safe_fractal_dimension(self, binary_edge):
        h, w = binary_edge.shape
        max_dim = max(h, w, 32)
        power = int(2 ** np.ceil(np.log2(max_dim)))
        padded = np.zeros((power, power), dtype=np.uint8)
        padded[:h, :w] = (binary_edge > 0).astype(np.uint8)

        if padded.sum() == 0:
            return 1.0

        n = int(np.log2(power))
        sizes = 2 ** np.arange(n - 1, 0, -1)
        counts, valid_sizes = [], []

        for k in sizes:
            S = np.add.reduceat(
                np.add.reduceat(padded, np.arange(0, power, k), axis=0),
                np.arange(0, power, k), axis=1
            )
            count = np.count_nonzero((S > 0) & (S < k * k))
            if count > 0:
                counts.append(count)
                valid_sizes.append(k)

        if len(valid_sizes) < 2:
            return 1.0

        coeffs = np.polyfit(np.log(valid_sizes), np.log(counts), 1)
        return float(-coeffs[0])

    def extract_features(self, img_path):
        img_bgra = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img_bgra is None or len(img_bgra.shape) != 3 or img_bgra.shape[2] != 4:
            return None

        b, g, r, alpha = cv2.split(img_bgra)
        mask_bool = alpha > 127
        if np.count_nonzero(mask_bool) < 15:
            return None

        mask_uint8 = (mask_bool.astype(np.uint8)) * 255
        bgr_img = cv2.merge((b, g, r))
        gray_raw = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

        # Fondo = 0; Píxeles internos >= 1
        gray_masked = np.where(mask_bool, np.maximum(gray_raw, 1), 0).astype(np.uint8)
        valid_pixels = gray_raw[mask_bool].astype(np.float64)

        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 5:
            return None

        features = {}

        # =========================================================================
        # F1: Morfometría Adimensional Invariante (5)
        # =========================================================================
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        if area <= 0 or perimeter <= 0:
            return None

        (cx, cy), (axis1, axis2), angle = cv2.fitEllipse(contour)
        major_axis = max(axis1, axis2)
        minor_axis = min(axis1, axis2)
        eq_diameter = np.sqrt(4 * area / np.pi)
        _, circum_radius = cv2.minEnclosingCircle(contour)

        dist_transform = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
        inscribed_radius = float(dist_transform.max()) + 1e-6
        norm_dist_map = dist_transform / inscribed_radius
        norm_dist_vals = norm_dist_map[mask_bool]

        features['geo_eccentricity'] = np.sqrt(max(0.0, 1.0 - (minor_axis / (major_axis + 1e-6)) ** 2))
        features['geo_axis_ratio'] = major_axis / (minor_axis + 1e-6)
        features['geo_eq_to_major'] = eq_diameter / (major_axis + 1e-6)
        features['geo_inscribed_to_circum'] = inscribed_radius / (circum_radius + 1e-6)
        features['geo_internal_thickness_std'] = float(np.std(norm_dist_vals))

        # =========================================================================
        # F2: Índices Topológicos y Polarización Biológica (5)
        # =========================================================================
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        hull_perimeter = float(cv2.arcLength(hull, True))

        hull_indices = cv2.convexHull(contour, returnPoints=False)
        max_defect_depth = 0.0
        mean_defect_depth = 0.0
        if len(hull_indices) > 3 and len(contour) > 3:
            try:
                defects = cv2.convexityDefects(contour, hull_indices)
                if defects is not None:
                    depths = defects[:, 0, 3] / 256.0
                    max_defect_depth = float(np.max(depths)) / (eq_diameter + 1e-6)
                    mean_defect_depth = float(np.mean(depths)) / (eq_diameter + 1e-6)
            except cv2.error:
                pass

        features['morph_solidity'] = area / (hull_area + 1e-6)
        features['morph_circularity'] = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)
        features['morph_convexity'] = hull_perimeter / (perimeter + 1e-6)
        features['morph_apical_notch_max'] = max_defect_depth
        features['morph_apical_notch_mean'] = mean_defect_depth

        # =========================================================================
        # F3: Momentos Invariantes de Hu (Silueta) (7)
        # =========================================================================
        M_bin = cv2.moments(mask_uint8, binaryImage=True)
        hu_bin = cv2.HuMoments(M_bin).flatten()
        for i, hu in enumerate(hu_bin):
            features[f'hu_{i}'] = float(-np.sign(hu) * np.log10(abs(hu) + 1e-15)) if hu != 0 else 0.0

        # =========================================================================
        # F4: Momentos Ortogonales de Zernike (25)
        # =========================================================================
        zernike_moms = mahotas.features.zernike_moments(
            gray_masked, radius=max(int(circum_radius), 2), degree=8
        )
        for i, z in enumerate(zernike_moms):
            features[f'zernike_{i}'] = float(z)

        # =========================================================================
        # F5: Descriptores de Fourier Normalizados (10)
        # =========================================================================
        pts = contour[:, 0, :].astype(np.float64)
        pts_centered = pts - np.mean(pts, axis=0)
        contour_complex = pts_centered[:, 0] + 1j * pts_centered[:, 1]
        fourier_result = np.fft.fft(contour_complex)
        mags = np.abs(fourier_result)
        scale_norm = mags[1] if len(mags) > 1 and mags[1] > 0 else 1.0
        fourier_desc = mags[2:12] / scale_norm
        for i in range(10):
            features[f'fourier_{i}'] = float(fourier_desc[i]) if i < len(fourier_desc) else 0.0

        # =========================================================================
        # F6: Complejidad Fractal de Bordes Internos + Contorno (2)
        # =========================================================================
        edges_internal = cv2.Canny(gray_masked, 40, 120)
        features['fractal_dimension'] = self._safe_fractal_dimension(edges_internal)

        radial_dists = np.linalg.norm(pts_centered, axis=1)
        radial_dists_norm = radial_dists / (np.max(radial_dists) + 1e-6)
        r_hist, _ = np.histogram(radial_dists_norm, bins=16, range=(0, 1), density=True)
        features['radial_contour_entropy'] = float(entropy(r_hist + 1e-12, base=2))

        # =========================================================================
        # F7: Estadísticas Fotométricas de 1er Orden (5)
        # =========================================================================
        mean_val = float(np.mean(valid_pixels))
        std_val = float(np.std(valid_pixels))
        q75, q25 = np.percentile(valid_pixels, [75, 25])
        hist_counts, _ = np.histogram(valid_pixels, bins=64, range=(0, 256), density=True)

        features['stat_coeff_variation'] = float(std_val / (mean_val + 1e-6))
        features['stat_skew'] = float(np.nan_to_num(skew(valid_pixels), nan=0.0))
        features['stat_kurtosis'] = float(np.nan_to_num(kurtosis(valid_pixels), nan=0.0))
        features['stat_entropy'] = float(entropy(hist_counts + 1e-12, base=2))
        features['stat_iqr_norm'] = float((q75 - q25) / (mean_val + 1e-6))

        # =========================================================================
        # F8: Textura de Haralick / GLCM Isotrópica (26)
        # =========================================================================
        try:
            haralick_matrix = mahotas.features.haralick(gray_masked, ignore_zeros=True)
            haralick_mean = haralick_matrix.mean(axis=0)
            haralick_ptp = haralick_matrix.ptp(axis=0)
            if len(haralick_mean) != 13:
                raise ValueError()
        except Exception:
            haralick_mean = np.zeros(13)
            haralick_ptp = np.zeros(13)

        for i in range(13):
            features[f'haralick_mean_{i}'] = float(haralick_mean[i])
            features[f'haralick_range_{i}'] = float(haralick_ptp[i])

        # =========================================================================
        # F9: Micro-Textura LBP Uniforme (18)
        # =========================================================================
        lbp = local_binary_pattern(gray_masked, self.lbp_points, self.lbp_radius, method='uniform')
        valid_lbp = lbp[mask_bool]
        lbp_hist, _ = np.histogram(
            valid_lbp, bins=self.lbp_bins, range=(0, self.lbp_bins), density=True
        )
        for i in range(self.lbp_bins):
            features[f'lbp_{i}'] = float(lbp_hist[i])

        # =========================================================================
        # F10: Frecuencia Espacial de Gabor Invariante a Rotación (8)
        # =========================================================================
        norm_gray = (gray_masked.astype(np.float32)) / (mean_val + 1e-6)
        for i, orientation_group in enumerate(self.gabor_FilterBank):
            responses = []
            for kernel in orientation_group:
                resp = cv2.filter2D(norm_gray, cv2.CV_32F, kernel)
                responses.append(np.abs(resp[mask_bool]))
            stacked_resp = np.mean(responses, axis=0)
            features[f'gabor_iso_mean_{i}'] = float(np.mean(stacked_resp))
            features[f'gabor_iso_std_{i}'] = float(np.std(stacked_resp))

        # =========================================================================
        # F11: Granulometría Morfológica Celular (4)
        # =========================================================================
        total_intensity = float(np.sum(gray_masked[mask_bool])) + 1e-6
        for i, k_size in enumerate([3, 5, 7, 9]):
            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            opened = cv2.morphologyEx(gray_masked, cv2.MORPH_OPEN, se)
            opened_intensity = float(np.sum(opened[mask_bool]))
            features[f'granulometry_{i}'] = (total_intensity - opened_intensity) / total_intensity

        # =========================================================================
        # F12: Densitometría Óptica (Beer-Lambert) (6)
        # =========================================================================
        od_map = -np.log10(np.clip(gray_raw.astype(np.float32) / 255.0, 1e-4, 1.0))
        od_valid = od_map[mask_bool]

        core_mask = (norm_dist_map > 0.50) & mask_bool
        periphery_mask = (norm_dist_map <= 0.50) & mask_bool

        core_od = float(np.mean(od_map[core_mask])) if np.any(core_mask) else float(np.mean(od_valid))
        periph_od = float(np.mean(od_map[periphery_mask])) if np.any(periphery_mask) else float(np.mean(od_valid))

        gx = cv2.Sobel(norm_gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(norm_gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag_map = np.sqrt(gx ** 2 + gy ** 2)
        grad_mag = grad_mag_map[mask_bool]
        laplacian_var = float(np.var(cv2.Laplacian(norm_gray, cv2.CV_32F)[mask_bool]))

        features['optic_od_mean'] = float(np.mean(od_valid))
        features['optic_od_std'] = float(np.std(od_valid))
        features['optic_core_to_periphery_ratio'] = core_od / (periph_od + 1e-6)
        features['optic_core_periphery_diff'] = core_od - periph_od
        features['optic_internal_grad_mean'] = float(np.mean(grad_mag))
        features['optic_laplacian_roughness'] = laplacian_var

        # =========================================================================
        # F13 [NUEVA]: Fito-Espectrometría Invariante a la Iluminación (12)
        # =========================================================================
        rf = r[mask_bool].astype(np.float32)
        gf = g[mask_bool].astype(np.float32)
        bf = b[mask_bool].astype(np.float32)
        rgb_sum = rf + gf + bf + 1e-5

        # Cromaticidad pura (invariante a intensidad de luz)
        r_norm = rf / rgb_sum
        g_norm = gf / rgb_sum
        b_norm = bf / rgb_sum

        # Índices botánicos: Excess Green (ExG), Excess Red (ExR) y Green-Red Vegetation Index (GRVI)
        exg = 2.0 * g_norm - r_norm - b_norm
        exr = 1.4 * r_norm - g_norm
        grvi = (gf - rf) / (gf + rf + 1e-5)

        lab_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        a_centered = lab_img[:, :, 1] - 128.0
        b_centered = lab_img[:, :, 2] - 128.0
        chroma_ab = np.sqrt(a_centered ** 2 + b_centered ** 2)
        ndi_lab = (b_centered - a_centered) / (np.abs(b_centered) + np.abs(a_centered) + 1e-5)

        chroma_core = float(np.mean(chroma_ab[core_mask])) if np.any(core_mask) else 0.0
        chroma_periph = float(np.mean(chroma_ab[periphery_mask])) if np.any(periphery_mask) else 0.0

        features['spec_exg_mean'] = float(np.mean(exg))
        features['spec_exg_std'] = float(np.std(exg))
        features['spec_exr_mean'] = float(np.mean(exr))
        features['spec_grvi_mean'] = float(np.mean(grvi))
        features['spec_grvi_std'] = float(np.std(grvi))
        features['spec_r_chrom_mean'] = float(np.mean(r_norm))
        features['spec_g_chrom_mean'] = float(np.mean(g_norm))
        features['spec_lab_ndi_mean'] = float(np.mean(ndi_lab[mask_bool]))
        features['spec_lab_ndi_std'] = float(np.std(ndi_lab[mask_bool]))
        features['spec_chroma_mean'] = float(np.mean(chroma_ab[mask_bool]))
        features['spec_chroma_std'] = float(np.std(chroma_ab[mask_bool]))
        features['spec_chroma_core_periph_ratio'] = chroma_core / (chroma_periph + 1e-5)

        # =========================================================================
        # F14 [NUEVA]: Coherencia Física 3D y Alineación Normal de Frontera (14)
        # =========================================================================
        # Gradiente del campo de distancia (apunta perpendicularmente desde el borde hacia el centro)
        dx_dist = cv2.Sobel(norm_dist_map, cv2.CV_32F, 1, 0, ksize=3)
        dy_dist = cv2.Sobel(norm_dist_map, cv2.CV_32F, 0, 1, ksize=3)
        dist_grad_mag = np.sqrt(dx_dist ** 2 + dy_dist ** 2) + 1e-6

        # Producto punto normalizado |cos(theta)| entre el gradiente de luz y la normal geométrica
        cos_align = np.abs(gx * dx_dist + gy * dy_dist) / ((grad_mag_map + 1e-6) * dist_grad_mag)
        epidermis_ring = (norm_dist_map > 0.0) & (norm_dist_map <= 0.30) & mask_bool
        cortex_ring = (norm_dist_map > 0.30) & (norm_dist_map <= 0.65) & mask_bool
        meristem_ring = (norm_dist_map > 0.65) & mask_bool

        epi_grad = float(np.mean(grad_mag_map[epidermis_ring])) if np.any(epidermis_ring) else float(np.mean(grad_mag))
        cor_grad = float(np.mean(grad_mag_map[cortex_ring])) if np.any(cortex_ring) else float(np.mean(grad_mag))
        mer_grad = float(np.mean(grad_mag_map[meristem_ring])) if np.any(meristem_ring) else float(np.mean(grad_mag))

        # Correlación radial entre profundidad morfológica e intensidad óptica (Efecto Domo 3D)
        if len(valid_pixels) > 5 and np.std(norm_dist_vals) > 1e-6 and std_val > 1e-6:
            dome_corr = float(np.corrcoef(norm_dist_vals, valid_pixels)[0, 1])
        else:
            dome_corr = 0.0

        # Momentos de Hu ponderados por la Densidad Óptica interna (no solo la silueta binaria)
        od_masked = np.where(mask_bool, od_map, 0.0).astype(np.float64)
        M_od = cv2.moments(od_masked, binaryImage=False)
        hu_od = cv2.HuMoments(M_od).flatten()

        # Desplazamiento entre el centroide geométrico y el centro de masa óptico
        cx_bin = M_bin['m10'] / (M_bin['m00'] + 1e-6)
        cy_bin = M_bin['m01'] / (M_bin['m00'] + 1e-6)
        cx_od = M_od['m10'] / (M_od['m00'] + 1e-6)
        cy_od = M_od['m01'] / (M_od['m00'] + 1e-6)
        centroid_shift = np.hypot(cx_bin - cx_od, cy_bin - cy_od) / (eq_diameter + 1e-6)

        features['coher_normal_align_mean'] = float(np.mean(cos_align[mask_bool]))
        features['coher_normal_align_epidermis'] = float(np.mean(cos_align[epidermis_ring])) if np.any(
            epidermis_ring) else 0.0
        features['coher_epidermis_to_core_grad'] = epi_grad / (mer_grad + 1e-6)
        features['coher_epidermis_to_cortex_grad'] = epi_grad / (cor_grad + 1e-6)
        features['coher_3d_dome_correlation'] = dome_corr
        features['coher_optical_centroid_shift'] = float(centroid_shift)
        features['coher_ring_intensity_ slope'] = (
                                                          (float(np.mean(gray_raw[meristem_ring])) if np.any(
                                                              meristem_ring) else mean_val) -
                                                          (float(np.mean(gray_raw[epidermis_ring])) if np.any(
                                                              epidermis_ring) else mean_val)
                                                  ) / (mean_val + 1e-6)
        for i in range(7):
            h_val = hu_od[i]
            features[f'coher_hu_optical_{i}'] = float(
                -np.sign(h_val) * np.log10(abs(h_val) + 1e-15)) if h_val != 0 else 0.0

        # =========================================================================
        # F15 [NUEVA]: Tensor Hessiano (Blobness vs. Nervaduras) y Sub-Estructura (10)
        # =========================================================================
        # Derivadas de 2do orden (Matriz Hessiana) para detectar nervaduras foliares vs. domos globulares
        gxx = cv2.Sobel(norm_gray, cv2.CV_32F, 2, 0, ksize=3)[mask_bool]
        gyy = cv2.Sobel(norm_gray, cv2.CV_32F, 0, 2, ksize=3)[mask_bool]
        gxy = cv2.Sobel(norm_gray, cv2.CV_32F, 1, 1, ksize=3)[mask_bool]

        trace_h = gxx + gyy
        det_h = gxx * gyy - (gxy ** 2)
        discriminant = np.sqrt(np.maximum(0.0, (trace_h ** 2) / 4.0 - det_h))
        lambda1 = trace_h / 2.0 + discriminant
        lambda2 = trace_h / 2.0 - discriminant

        # Ordenar |lambda1| >= |lambda2|
        abs_l1 = np.maximum(np.abs(lambda1), np.abs(lambda2)) + 1e-6
        abs_l2 = np.minimum(np.abs(lambda1), np.abs(lambda2))
        blobness_ratio = abs_l2 / abs_l1  # Alto (~1) en embriones globulares, bajo (~0) en nervaduras
        structureness = np.sqrt(abs_l1 ** 2 + abs_l2 ** 2)  # Energía de curvatura de 2do orden

        # Auto-segmentación interna de Otsu para verificar si dentro del polígono hay un núcleo coherente
        _, inner_otsu = cv2.threshold(
            valid_pixels.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        inner_fg_ratio = float(np.count_nonzero(inner_otsu)) / len(valid_pixels)

        features['hess_blobness_mean'] = float(np.mean(blobness_ratio))
        features['hess_blobness_std'] = float(np.std(blobness_ratio))
        features['hess_structureness_mean'] = float(np.mean(structureness))
        features['hess_structureness_std'] = float(np.std(structureness))
        features['hess_det_mean'] = float(np.mean(det_h))
        features['hess_det_std'] = float(np.std(det_h))
        features['hess_trace_mean'] = float(np.mean(trace_h))
        features['hess_anisotropy'] = float(np.mean((abs_l1 - abs_l2) / (abs_l1 + abs_l2 + 1e-6)))
        features['hess_inner_otsu_ratio'] = inner_fg_ratio
        features['hess_edge_density'] = float(np.count_nonzero(edges_internal[mask_bool])) / len(valid_pixels)

        # Sanitización final
        for k, v in features.items():
            if not np.isfinite(v):
                features[k] = 0.0

        return features


def _worker_process_image(path):
    extractor = _worker_process_image.extractor
    feats = extractor.extract_features(path)
    if feats is not None:
        return os.path.basename(path), feats
    return os.path.basename(path), None


def _init_worker():
    _worker_process_image.extractor = AdvancedBiologicalExtractor()


class HyperdimensionalOrchestrator:
    def __init__(self, input_dir, output_csv, label_value=1, max_workers=None):
        self.input_dir = input_dir
        self.output_csv = output_csv
        self.label_value = label_value
        self.max_workers = max_workers or os.cpu_count()

    def run(self):
        print(f"\n[*] Escaneando recortes en: {self.input_dir}")
        img_paths = sorted(glob.glob(os.path.join(self.input_dir, "*.png")))
        total_imgs = len(img_paths)

        if total_imgs == 0:
            print(f"[!] No se encontraron imágenes PNG en {self.input_dir}")
            return

        print(f"[*] Extrayendo 15 familias (157 variables) de {total_imgs} muestras con {self.max_workers} núcleos...")
        dataset, filenames, discarded = [], [], 0

        with ProcessPoolExecutor(max_workers=self.max_workers, initializer=_init_worker) as executor:
            futures = {executor.submit(_worker_process_image, path): path for path in img_paths}
            for idx, future in enumerate(as_completed(futures), 1):
                fname, feats = future.result()
                if feats is not None:
                    dataset.append(feats)
                    filenames.append(fname)
                else:
                    discarded += 1
                if idx % 500 == 0 or idx == total_imgs:
                    print(f"    -> Progreso: {idx}/{total_imgs} (Válidos: {len(dataset)}, Descartados: {discarded})")

        df = pd.DataFrame(dataset)
        df.insert(0, 'filename', filenames)
        df['label'] = self.label_value
        df.sort_values(by='filename', inplace=True)

        os.makedirs(os.path.dirname(os.path.abspath(self.output_csv)), exist_ok=True)
        df.to_csv(self.output_csv, index=False)
        print(
            f"[*] Guardado exitoso: {self.output_csv} ({len(df)} filas, {len(df.columns) - 2} características, label={self.label_value})")


if __name__ == "__main__":
    POS_DIR = './Fase 1 Segmentacion/1_train_features_ES'
    POS_CSV = './Fase 2 Extraccion/hyperdimensional_features_ES_raw.csv'
    HyperdimensionalOrchestrator(POS_DIR, POS_CSV, label_value=1).run()

    NEG_DIR = './Fase 1 Segmentacion/1_train_features_NO_ES'
    NEG_CSV = './Fase 2 Extraccion/hyperdimensional_features_NEG_raw.csv'
    if os.path.exists(NEG_DIR):
        HyperdimensionalOrchestrator(NEG_DIR, NEG_CSV, label_value=0).run()

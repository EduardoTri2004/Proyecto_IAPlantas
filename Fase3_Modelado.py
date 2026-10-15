import os
import joblib
import random
import warnings
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


class BiologicalFamilySchema:
    """Agrupa dinámicamente el vector de 157 características en sus 15 familias analíticas."""

    FAMILY_PREFIXES = {
        'F1_Geometria_Adimensional': ['geo_'],
        'F2_Morfologia_Topologica': ['morph_'],
        'F3_Momentos_Hu': ['hu_'],
        'F4_Momentos_Zernike': ['zernike_'],
        'F5_Descriptores_Fourier': ['fourier_'],
        'F6_Complejidad_Fractal': ['fractal_', 'radial_'],
        'F7_Estadistica_Fotometrica': ['stat_'],
        'F8_Textura_Haralick_GLCM': ['haralick_'],
        'F9_MicroTextura_LBP': ['lbp_'],
        'F10_Frecuencia_Gabor': ['gabor_'],
        'F11_Granulometria_Celular': ['granulometry_'],
        'F12_Densitometria_Optica': ['optic_'],
        'F13_FitoEspectral_Invariante': ['spec_'],
        'F14_Coherencia_Frontera_3D': ['coher_'],
        'F15_Tensor_Hessiano_Venacion': ['hess_']
    }

    @classmethod
    def build_family_indices(cls, feature_names):
        family_indices = {}
        for fam_name, prefixes in cls.FAMILY_PREFIXES.items():
            idxs = [
                i for i, col in enumerate(feature_names)
                if any(col.startswith(p) for p in prefixes)
            ]
            if idxs:
                family_indices[fam_name] = idxs
        return family_indices


class PeriodicNumericalTokenizer(nn.Module):
    """
    Embeddings Numéricos Periódicos (Gorishniy et al., NeurIPS) + Proyección SwiGLU.
    Permite a la red aprender fronteras de decisión de alta frecuencia en datos tabulares.
    """

    def __init__(self, in_features, d_model, n_frequencies=4, dropout=0.2):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(in_features, n_frequencies) * 0.5)
        expanded_dim = in_features * (2 * n_frequencies + 1)
        self.w1 = nn.Linear(expanded_dim, d_model * 2)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, in_features) -> expandir con senos y cosenos aprendibles
        angles = 2.0 * np.pi * x.unsqueeze(-1) * self.freqs.unsqueeze(0)
        periodic = torch.cat([
            x,
            torch.sin(angles).reshape(x.size(0), -1),
            torch.cos(angles).reshape(x.size(0), -1)
        ], dim=-1)
        h = self.w1(periodic)
        x_val, x_gate = h.chunk(2, dim=-1)
        out = F.silu(x_gate) * x_val
        return self.norm(self.dropout(out))


class DeepCrossLayerV2(nn.Module):
    """Red de Cruce Explícito Polinomial (DCNv2) para capturar interacciones de alto orden."""

    def __init__(self, in_features, dropout=0.15):
        super().__init__()
        self.w = nn.Linear(in_features, in_features)
        self.norm = nn.LayerNorm(in_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x0, xl):
        cross = x0 * self.w(xl)
        return self.norm(xl + self.dropout(cross))


class GatedResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.25):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        h = F.gelu(self.fc1(x))
        h = self.dropout(h)
        val, gate = self.fc2(h).chunk(2, dim=-1)
        h = val * torch.sigmoid(gate)
        return self.norm(residual + h)


class EmbryoGateFormerV2(nn.Module):
    """
    Arquitectura Híbrida de 2da Generación:
      - Periodic Fourier Family Tokenizers (15 Familias)
      - Squeeze-and-Excitation Family Gate
      - Multi-Head Self-Attention (Transformer Inter-Familiar)
      - Deep Gated Residual Network (GRN)
      - Deep & Cross Network (DCNv2)
      - Multi-Sample Dropout Classifier (5 cabezales promediados anti-overfitting)
    """

    def __init__(self, family_indices_dict, total_features, d_model=64, n_heads=4, dropout=0.25):
        super().__init__()
        self.family_names = list(family_indices_dict.keys())
        self.family_indices = [family_indices_dict[k] for k in self.family_names]
        self.num_families = len(self.family_names)

        # 1. Tokenizadores Periódicos por Familia
        self.family_encoders = nn.ModuleList([
            PeriodicNumericalTokenizer(len(idxs), d_model, n_frequencies=4, dropout=dropout)
            for idxs in self.family_indices
        ])

        # 2. Compuerta de Familia (Family Gate)
        self.family_gate_net = nn.Sequential(
            nn.LayerNorm(total_features),
            nn.Linear(total_features, self.num_families * 2),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(self.num_families * 2, self.num_families),
            nn.Sigmoid()
        )

        # 3. Ruta 1: Atención Inter-Familiar (Transformer)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_families + 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.attention_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.attn_norm = nn.LayerNorm(d_model * 2)

        # 4. Ruta 2: Deep Gated Residual
        deep_dim = self.num_families * d_model
        self.deep_compress = nn.Sequential(
            nn.Linear(deep_dim, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.SiLU()
        )
        self.deep_grn1 = GatedResidualBlock(d_model * 2, dropout=dropout)
        self.deep_grn2 = GatedResidualBlock(d_model * 2, dropout=dropout)

        # 5. Ruta 3: Deep & Cross Network (DCNv2)
        self.input_norm = nn.LayerNorm(total_features)
        self.cross1 = DeepCrossLayerV2(total_features, dropout=dropout * 0.5)
        self.cross2 = DeepCrossLayerV2(total_features, dropout=dropout * 0.5)
        self.wide_proj = nn.Sequential(
            nn.Linear(total_features, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU()
        )

        # 6. Cabezal con Multi-Sample Dropout (5 ramas promediadas para evitar sobreajuste)
        fusion_dim = (d_model * 2) + (d_model * 2) + d_model  # 128 + 128 + 64 = 320
        self.pre_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128)
        )
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in (0.1, 0.2, 0.3, 0.4, 0.5)])
        self.head_fc = nn.Linear(128, 1)

    def forward(self, x, return_gates=False):
        batch_size = x.size(0)
        family_gates = self.family_gate_net(x)

        family_tokens = []
        for i, idxs in enumerate(self.family_indices):
            x_fam = x[:, idxs]
            token = self.family_encoders[i](x_fam)
            gated_token = token * family_gates[:, i].unsqueeze(-1)
            family_tokens.append(gated_token)

        tokens_seq = torch.stack(family_tokens, dim=1)

        # Ruta Atencional: Combinar [CLS] token + Mean Pooling de los tokens familiares
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        attn_input = torch.cat((cls_tokens, tokens_seq), dim=1) + self.pos_embedding
        attn_out = self.attention_encoder(attn_input)
        cls_feat = attn_out[:, 0, :]
        mean_feat = attn_out[:, 1:, :].mean(dim=1)
        attn_context = self.attn_norm(torch.cat([cls_feat, mean_feat], dim=-1))

        # Ruta Deep Residual
        deep_flat = tokens_seq.reshape(batch_size, -1)
        deep_h = self.deep_compress(deep_flat)
        deep_h = self.deep_grn1(deep_h)
        deep_out = self.deep_grn2(deep_h)

        # Ruta Wide DCNv2
        x0 = self.input_norm(x)
        xl = self.cross1(x0, x0)
        xl = self.cross2(x0, xl)
        wide_out = self.wide_proj(xl)

        # Fusión + Multi-Sample Dropout
        fused = torch.cat([attn_context, deep_out, wide_out], dim=-1)
        h = self.pre_head(fused)
        logits = torch.mean(
            torch.stack([self.head_fc(drop(h)) for drop in self.dropouts], dim=0),
            dim=0
        ).squeeze(-1)

        if return_gates:
            return logits, family_gates
        return logits


class BinaryFocalLossWithLogits(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, label_smoothing=0.02):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = label_smoothing

    def forward(self, logits, targets):
        targets_smooth = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets_smooth, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets_smooth + (1.0 - probs) * (1.0 - targets_smooth)
        focal_weight = ((1.0 - p_t) ** self.gamma)
        alpha_t = self.alpha * targets_smooth + (1.0 - self.alpha) * (1.0 - targets_smooth)
        return (alpha_t * focal_weight * bce_loss).mean()


class EmbryoTabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Phase3ModelingPipeline:
    def __init__(self, pos_csv, neg_csv, output_model_dir):
        self.pos_csv = pos_csv
        self.neg_csv = neg_csv
        self.output_model_dir = output_model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        os.makedirs(self.output_model_dir, exist_ok=True)

    @staticmethod
    def _extract_source_image_id(filename_str):
        fname = os.path.basename(str(filename_str))
        if '_ES_' in fname:
            return fname.split('_ES_')[0]
        if '_NOES_' in fname:
            return fname.split('_NOES_')[0]
        return fname.split('_')[0]

    def _fit_safe_yeo_johnson(self, X_train_raw, X_val_raw):
        scaler = PowerTransformer(method='yeo-johnson', standardize=True)
        try:
            X_train = scaler.fit_transform(X_train_raw)
            X_val = scaler.transform(X_val_raw)
        except Exception:
            X_train = np.zeros_like(X_train_raw, dtype=np.float64)
            X_val = np.zeros_like(X_val_raw, dtype=np.float64)
            scaler = []
            for col_idx in range(X_train_raw.shape[1]):
                col_tr = X_train_raw[:, [col_idx]]
                col_va = X_val_raw[:, [col_idx]]
                pt_col = PowerTransformer(method='yeo-johnson', standardize=True)
                try:
                    X_train[:, [col_idx]] = pt_col.fit_transform(col_tr)
                    X_val[:, [col_idx]] = pt_col.transform(col_va)
                    scaler.append(pt_col)
                except Exception:
                    std_col = StandardScaler()
                    X_train[:, [col_idx]] = std_col.fit_transform(col_tr)
                    X_val[:, [col_idx]] = std_col.transform(col_va)
                    scaler.append(std_col)

        X_train = np.clip(np.nan_to_num(X_train, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)
        X_val = np.clip(np.nan_to_num(X_val, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)
        return X_train, X_val, scaler

    @staticmethod
    def _apply_tabular_mixup(bx, by, alpha=0.2):
        """Interpolación convexa suave (Mixup) + ruido gaussiano ligero para evitar memorización."""
        bx_noisy = bx + torch.randn_like(bx) * 0.03
        if alpha > 0 and random.random() < 0.5:
            lam = np.random.beta(alpha, alpha)
            index = torch.randperm(bx.size(0), device=bx.device)
            mixed_x = lam * bx_noisy + (1.0 - lam) * bx_noisy[index]
            mixed_y = lam * by + (1.0 - lam) * by[index]
            return mixed_x, mixed_y
        return bx_noisy, by

    def load_and_prepare_data(self):
        df_pos = pd.read_csv(self.pos_csv)
        df_pos['label'] = 1

        df_neg = pd.read_csv(self.neg_csv)
        df_neg['label'] = 0

        meta_cols = {'filename', 'label', 'source_image'}
        feature_cols = [c for c in df_pos.columns if c not in meta_cols and c in df_neg.columns]

        ordered_cols = ['filename'] + feature_cols + ['label']
        df_full = pd.concat([df_pos[ordered_cols], df_neg[ordered_cols]], axis=0, ignore_index=True)

        for col in feature_cols:
            df_full[col] = pd.to_numeric(df_full[col], errors='coerce')
        df_full[feature_cols] = df_full[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df_full['source_image'] = df_full['filename'].apply(self._extract_source_image_id)

        X_raw = df_full[feature_cols].values.astype(np.float64)
        y = df_full['label'].values.astype(np.int64)
        groups = df_full['source_image'].values

        unique_imgs = len(np.unique(groups))
        print(f"\n[*] Dataset Consolidado: {len(df_full)} recortes de {unique_imgs} fotografías")
        print(f"    -> Embriones Somáticos (Clase 1): {np.sum(y == 1)}")
        print(f"    -> Planta Madre / Callo (Clase 0): {np.sum(y == 0)}")
        print(f"    -> Dimensionalidad activa       : {len(feature_cols)} características en 15 familias.")

        if unique_imgs >= 10:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
            train_idx, val_idx = next(gss.split(X_raw, y, groups=groups))
            X_train_raw, X_val_raw = X_raw[train_idx], X_raw[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
        else:
            X_train_raw, X_val_raw, y_train, y_val = train_test_split(
                X_raw, y, test_size=0.10, random_state=42, stratify=y
            )

        X_train, X_val, scaler = self._fit_safe_yeo_johnson(X_train_raw, X_val_raw)

        joblib.dump(scaler, os.path.join(self.output_model_dir, 'yeo_johnson_scaler.pkl'))
        joblib.dump(feature_cols, os.path.join(self.output_model_dir, 'feature_columns.pkl'))

        return X_train, X_val, y_train, y_val, feature_cols

    def train_and_evaluate(self, epochs=75, batch_size=64, lr=8e-4):
        X_train, X_val, y_train, y_val, feature_cols = self.load_and_prepare_data()
        family_indices = BiologicalFamilySchema.build_family_indices(feature_cols)

        train_loader = DataLoader(EmbryoTabularDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(EmbryoTabularDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

        model = EmbryoGateFormerV2(
            family_indices_dict=family_indices,
            total_features=len(feature_cols),
            d_model=64,
            n_heads=4,
            dropout=0.25
        ).to(self.device)

        criterion = BinaryFocalLossWithLogits(alpha=0.5, gamma=2.0, label_smoothing=0.02)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        best_auc = -1.0
        best_model_path = os.path.join(self.output_model_dir, 'best_embryo_gateformer.pth')

        print(
            f"\n[*] Entrenando EmbryoGateFormerV2 (con Periodic Embeddings + DCNv2 + Mixup) en {self.device.type.upper()}...")
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                bx_mix, by_mix = self._apply_tabular_mixup(bx, by, alpha=0.2)

                optimizer.zero_grad()
                logits = model(bx_mix)
                loss = criterion(logits, by_mix)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * bx.size(0)

            scheduler.step()
            train_loss /= len(train_loader.dataset)

            model.eval()
            val_preds, val_targets = [], []
            with torch.no_grad():
                for bx, by in val_loader:
                    bx = bx.to(self.device)
                    probs = torch.sigmoid(model(bx)).cpu().numpy()
                    val_preds.extend(np.atleast_1d(probs))
                    val_targets.extend(np.atleast_1d(by.numpy()))

            val_preds = np.array(val_preds)
            val_targets = np.array(val_targets)
            val_f1 = f1_score(val_targets, (val_preds >= 0.5).astype(int), zero_division=0)
            val_auc = roc_auc_score(val_targets, val_preds)

            # Guardar el mejor checkpoint guiado por ROC-AUC para preservar la máxima separabilidad
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'family_indices': family_indices,
                    'feature_cols': feature_cols
                }, best_model_path)

            if epoch % 15 == 0 or epoch == 1:
                print(
                    f"    Época [{epoch:02d}/{epochs:02d}] | Loss: {train_loss:.4f} | Val F1: {val_f1:.4f} | Val ROC-AUC: {val_auc:.4f}")

        # --- Entrenar co-ensamble de Árboles (HistGradientBoosting + ExtraTrees) para Fusión Tabular ---
        print("[*] Ajustando rama de Árboles de Gradiente para Sinergia Deep-Tree...")
        hgb = HistGradientBoostingClassifier(
            max_iter=350, learning_rate=0.04, max_leaf_nodes=31, l2_regularization=1.5, random_state=42
        )
        et = ExtraTreesClassifier(n_estimators=300, max_features='sqrt', random_state=42, n_jobs=-1)
        hgb.fit(X_train, y_train)
        et.fit(X_train, y_train)
        joblib.dump({'hgb': hgb, 'et': et}, os.path.join(self.output_model_dir, 'tree_expert_branch.pkl'))

        # --- Evaluación del Mejor Modelo y Calibración de Umbral Óptimo ---
        checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        nn_probs, all_targets, all_gates = [], [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(self.device)
                logits, gates = model(bx, return_gates=True)
                nn_probs.extend(np.atleast_1d(torch.sigmoid(logits).cpu().numpy()))
                all_targets.extend(np.atleast_1d(by.numpy()))
                all_gates.append(gates.cpu().numpy())

        nn_probs = np.array(nn_probs)
        all_targets = np.array(all_targets)
        mean_gates = np.vstack(all_gates).mean(axis=0)

        tree_probs = 0.65 * hgb.predict_proba(X_val)[:, 1] + 0.35 * et.predict_proba(X_val)[:, 1]
        hybrid_probs = 0.55 * nn_probs + 0.45 * tree_probs

        # Buscar el umbral óptimo que maximiza el F1-Score
        best_thresh, best_f1_cal = 0.5, 0.0
        for th in np.linspace(0.30, 0.75, 91):
            score = f1_score(all_targets, (hybrid_probs >= th).astype(int), zero_division=0)
            if score > best_f1_cal:
                best_f1_cal = score
                best_thresh = float(th)

        joblib.dump(best_thresh, os.path.join(self.output_model_dir, 'optimal_threshold.pkl'))
        final_preds = (hybrid_probs >= best_thresh).astype(int)

        print("\n" + "=" * 68)
        print(f"  RESULTADOS FINALES DE VALIDACIÓN (UMBRAL CALIBRADO τ* = {best_thresh:.3f})")
        print("=" * 68)
        print(f"  Accuracy  : {accuracy_score(all_targets, final_preds) * 100:.2f}%")
        print(f"  Precision : {precision_score(all_targets, final_preds, zero_division=0) * 100:.2f}%")
        print(f"  Recall    : {recall_score(all_targets, final_preds, zero_division=0) * 100:.2f}%")
        print(f"  F1-Score  : {f1_score(all_targets, final_preds, zero_division=0) * 100:.2f}%")
        print(
            f"  ROC-AUC   : {roc_auc_score(all_targets, hybrid_probs):.4f}  (Red Pura AUC: {roc_auc_score(all_targets, nn_probs):.4f})")
        print("\n  Matriz de Confusión:")
        print(confusion_matrix(all_targets, final_preds))
        print("\n  Reporte por Clase:")
        print(classification_report(all_targets, final_preds, target_names=['No-Embrión (0)', 'Embrión ES (1)']))

        print("=" * 68)
        print("  IMPORTANCIA DE LAS 15 FAMILIAS (FAMILY GATE ACTIVATION)")
        print("=" * 68)
        fam_importance = sorted(zip(family_indices.keys(), mean_gates), key=lambda x: x[1], reverse=True)
        for rank, (fname, weight) in enumerate(fam_importance, 1):
            print(f"  {rank:02d}. {fname:<32} -> Peso Compuerta: {weight:.4f}")
        print("=" * 68)


if __name__ == "__main__":
    POS_CSV = './Fase 2 Extraccion/hyperdimensional_features_ES_raw.csv'
    NEG_CSV = './Fase 2 Extraccion/hyperdimensional_features_NEG_raw.csv'
    OUTPUT_MODEL_DIR = './Fase 3 Modelado'

    pipeline = Phase3ModelingPipeline(POS_CSV, NEG_CSV, OUTPUT_MODEL_DIR)
    pipeline.train_and_evaluate(epochs=150  , batch_size=64, lr=8e-4)
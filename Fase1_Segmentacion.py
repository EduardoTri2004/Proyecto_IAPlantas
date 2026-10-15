import os
import json
import cv2
import numpy as np
import shutil


class LabelStudioParser:
    """Encapsula la lógica matemática para interpretar las etiquetas poligonales."""

    @staticmethod
    def extract_polygons(task_data):
        polygons = []
        for annotation in task_data.get('annotations', []):
            for result in annotation.get('result', []):
                # Asegurarnos de que procesamos solo las etiquetas de polígonos
                if result.get('type') == 'polygonlabels':
                    val = result['value']
                    points_pct = val['points']

                    # CORRECCIÓN: Las dimensiones están al nivel de 'result', no en 'value'
                    w = result['original_width']
                    h = result['original_height']

                    abs_points = []
                    for pt in points_pct:
                        x = int((pt[0] / 100.0) * w)
                        y = int((pt[1] / 100.0) * h)
                        abs_points.append([x, y])

                    polygons.append(np.array(abs_points, dtype=np.int32))
        return polygons


class EmbryoProcessor:
    """Aplica técnicas de Visión Computacional para extraer el polígono orgánico exacto."""

    def __init__(self):
        # CLAHE mejora el contraste local de las texturas orgánicas para estandarizar la iluminación
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _enhance_illumination(self, img):
        # Operamos solo en luminosidad (L) para no corromper la información de color real
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self.clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    def process_and_save_embryos(self, img_path, polygons, output_dir, original_name):
        img = cv2.imread(img_path)
        if img is None:
            print(f"[!] Error al leer la imagen: {img_path}")
            return

        # 1. Estandarizar iluminación
        enhanced_img = self._enhance_illumination(img)

        # 2. Convertir a RGBA para habilitar la transparencia (Canal Alpha)
        rgba_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2BGRA)
        base_name = os.path.splitext(original_name)[0]

        img_h, img_w = img.shape[:2]

        for idx, poly in enumerate(polygons):
            # Filtro 1: Un polígono real necesita al menos 3 puntos
            if len(poly) < 3:
                continue

            # Crear máscara binaria exacta basada en el polígono
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.fillPoly(mask, [poly], 255)

            # Copiar la imagen y aplicar la máscara al canal Alpha
            embryo_rgba = rgba_img.copy()
            embryo_rgba[:, :, 3] = mask

            # Obtener el Bounding Box del polígono
            x, y, w, h = cv2.boundingRect(poly)

            # Filtro 2: Asegurar que el recorte no se salga de los límites de la imagen original
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(img_w, x + w), min(img_h, y + h)

            # Recortar estrictamente la biomasa
            cropped_polygon = embryo_rgba[y1:y2, x1:x2]

            # Filtro 3: Verificar que el recorte final no esté vacío (Evita el cv2.error)
            if cropped_polygon.size == 0 or w == 0 or h == 0:
                print(f"[*] Ignorando polígono vacío/corrupto en {base_name}_ES_{idx}.")
                continue

            # Guardar resultado como PNG (Obligatorio para conservar la transparencia)
            out_path = os.path.join(output_dir, f"{base_name}_ES_{idx}.png")
            cv2.imwrite(out_path, cropped_polygon)


class DatasetManager:
    """Gestiona la lectura del dataset JSON y la división Train/Test vs Conteo Final."""

    def __init__(self, json_path, images_dir, output_dir, train_limit=900):
        self.json_path = json_path
        self.images_dir = images_dir
        self.output_dir = output_dir
        self.train_limit = train_limit

        # Estructura de carpetas de salida
        self.train_dir = os.path.join(output_dir, '1_train_features_ES')
        self.count_dir = os.path.join(output_dir, '2_final_counting_Images')
        self._prepare_directories()

    def _prepare_directories(self):
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.count_dir, exist_ok=True)

    def run_pipeline(self, parser, processor):
        print("[*] Iniciando procesamiento del dataset...")

        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Crear un mapa de los archivos locales para ignorar diferencias de extensión (.jpg vs .png)
        if not os.path.exists(self.images_dir):
            print(f"[!] Directorio de imágenes no encontrado: {self.images_dir}")
            return

        local_files = os.listdir(self.images_dir)
        # Diccionario que mapea "IMG1" -> "IMG1.jpg"
        local_files_dict = {os.path.splitext(f)[0]: f for f in local_files}

        for index, task in enumerate(data):
            # 1. Extraer nombre crudo de Label Studio (ej: 0c2372b4-IMG1.png)
            raw_name = task['data']['image'].split('/')[-1]

            # 2. Remover el hash de Label Studio dividiendo por el primer guion (ej -> IMG1.png)
            clean_name = raw_name.split('-', 1)[-1] if '-' in raw_name else raw_name

            # 3. Extraer solo el nombre base sin extensión (ej -> IMG1)
            base_name = os.path.splitext(clean_name)[0]

            # 4. Buscar el archivo real en la carpeta local usando el diccionario
            if base_name in local_files_dict:
                real_local_name = local_files_dict[base_name]
                img_full_path = os.path.join(self.images_dir, real_local_name)
            else:
                print(f"[!] Advertencia: Imagen no encontrada localmente -> {base_name}")
                continue

            # Separación: Entrenamiento vs Conteo Final
            if index < self.train_limit:
                polygons = parser.extract_polygons(task)
                processor.process_and_save_embryos(img_full_path, polygons, self.train_dir, real_local_name)
            else:
                dest_path = os.path.join(self.count_dir, real_local_name)
                shutil.copy(img_full_path, dest_path)

        print(f"[*] Pipeline finalizado exitosamente.")
        print(f"[*] Datos de entrenamiento guardados en: {self.train_dir}")
        print(f"[*] Imágenes de conteo final guardadas en: {self.count_dir}")


if __name__ == "__main__":
    # --- Configuración de rutas (Modifica estas rutas según tu entorno Ubuntu) ---
    JSON_PATH = 'project-3-at-2026-10-14-11-48-9adb768b.json'
    IMAGES_DIR = './Fotografías selectas'  # Carpeta con las 1000 fotos originales descargadas
    OUTPUT_DIR = './Fase 1 Segmentacion'  # Carpeta raíz donde se creará el proyecto

    # Inyección de dependencias (Principios SOLID)
    parser = LabelStudioParser()
    processor = EmbryoProcessor()
    manager = DatasetManager(JSON_PATH, IMAGES_DIR, OUTPUT_DIR, train_limit=900)

    # Ejecutar el pipeline
    manager.run_pipeline(parser, processor)
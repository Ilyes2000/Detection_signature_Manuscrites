# signature_detector.py
import numpy as np
import cv2
from ultralytics import YOLO

class SignatureDetector:
    def __init__(self, model_path):
        """
        Initialise un détecteur de signatures basé sur un modèle YOLOv8 (format .pt).
        """
        self.model = YOLO(model_path)

    def detect(self, image_pil, conf_thres=0.5, iou_thres=0.5):
        """
        Détecte des signatures dans une image PIL en utilisant le modèle YOLOv8.

        Arguments :
        - image_pil : L'image au format PIL.
        - conf_thres : Seuil de confiance minimum pour conserver la détection.
        - iou_thres : Seuil d'IoU pour le NMS (non-maximum suppression).

        Retourne :
        - output_image_np : L'image annotée (format NumPy).
        - metrics : Un dictionnaire ou autre structure pour le logging, si désiré.
        """
        # Conversion PIL -> NumPy
        image_np = np.array(image_pil)

        # Exécution de la détection avec YOLOv8
        results = self.model.predict(
            source=image_np,
            conf=conf_thres,
            iou=iou_thres,  # iou param
            verbose=False
        )

        # Récupération de la première prédiction
        boxes = results[0].boxes  # ultralytics Results object

        # --- Dessin des boîtes sur l'image ---
        output_image = image_np.copy()
        if boxes is not None:
            for box in boxes:
                # box.xyxy[0] est un tenseur [x1, y1, x2, y2]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Éventuellement, préparez des métriques custom (ici, juste un exemple)
        metrics = {
            "num_detections": len(boxes) if boxes is not None else 0
        }

        return output_image, metrics

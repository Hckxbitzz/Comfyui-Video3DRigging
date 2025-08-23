import numpy as np
import torch
import cv2
import os
import folder_paths
from PIL import Image
import json

class Basic3DRigging:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "input.mp4", "multiline": False}),
                "mode": (["face_tracking", "motion_vectors", "simple_mesh"],),
                "mesh_points": ("INT", {"default": 100, "min": 10, "max": 500}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("output_video", "mesh_data", "debug_info")
    FUNCTION = "process"
    CATEGORY = "Video"
    OUTPUT_NODE = True

    def process(self, video_path: str, mode: str, mesh_points: int, reference_image: torch.Tensor = None):
        # Verificar que el archivo existe
        if not os.path.exists(video_path):
            # Buscar en la carpeta de input de ComfyUI
            input_dir = folder_paths.get_input_directory()
            video_path = os.path.join(input_dir, video_path)
            if not os.path.exists(video_path):
                return (torch.zeros(1, 3, 256, 256), "{}", "ERROR: Video no encontrado")

        try:
            # Capturar video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return (torch.zeros(1, 3, 256, 256), "{}", "ERROR: No se puede abrir el video")

            frames = []
            success, frame = cap.read()
            frame_count = 0
            
            while success and frame_count < 100:  # Limitar a 100 frames para prueba
                # Procesar frame según el modo
                if mode == "face_tracking":
                    processed_frame = self.detect_face_simple(frame)
                elif mode == "motion_vectors":
                    processed_frame = self.draw_motion_vectors(frame, frame_count)
                else:
                    processed_frame = self.draw_simple_mesh(frame, mesh_points)
                
                frames.append(processed_frame)
                success, frame = cap.read()
                frame_count += 1

            cap.release()

            if not frames:
                return (torch.zeros(1, 3, 256, 256), "{}", "ERROR: No se pudieron leer frames")

            # Convertir a tensor de ComfyUI
            video_tensor = self.frames_to_tensor(frames)
            
            # Datos simples de malla (para demostración)
            mesh_data = json.dumps({
                "vertices": [[i, i, i] for i in range(mesh_points)],
                "faces": [[0, 1, 2]],
                "mode": mode,
                "frames_processed": frame_count
            })
            
            debug_info = f"Procesados {frame_count} frames. Modo: {mode}"
            
            return (video_tensor, mesh_data, debug_info)

        except Exception as e:
            return (torch.zeros(1, 3, 256, 256), "{}", f"ERROR: {str(e)}")

    def detect_face_simple(self, frame):
        """Detección simple de caras usando OpenCV"""
        frame_copy = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detector de caras simple
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            # Dibujar rectángulo
            cv2.rectangle(frame_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Puntos simples para "rigging"
            cv2.circle(frame_copy, (x + w//2, y + h//2), 5, (0, 0, 255), -1)  # Centro
            cv2.circle(frame_copy, (x, y), 3, (255, 0, 0), -1)  # Esquina superior izquierda
            cv2.circle(frame_copy, (x + w, y), 3, (255, 0, 0), -1)  # Esquina superior derecha
        
        return frame_copy

    def draw_motion_vectors(self, frame, frame_count):
        """Dibujar vectores de movimiento simples"""
        frame_copy = frame.copy()
        h, w = frame.shape[:2]
        
        # Crear un grid de puntos
        grid_size = 10
        for i in range(0, h, grid_size):
            for j in range(0, w, grid_size):
                # Simular movimiento con una función seno
                dx = int(5 * np.sin(frame_count * 0.1 + i * 0.01))
                dy = int(5 * np.cos(frame_count * 0.1 + j * 0.01))
                
                # Dibujar vector
                start_point = (j, i)
                end_point = (j + dx, i + dy)
                cv2.arrowedLine(frame_copy, start_point, end_point, (0, 255, 255), 1)
                cv2.circle(frame_copy, start_point, 2, (255, 0, 0), -1)
        
        return frame_copy

    def draw_simple_mesh(self, frame, num_points):
        """Dibujar una malla simple"""
        frame_copy = frame.copy()
        h, w = frame.shape[:2]
        
        # Generar puntos aleatorios
        points = []
        for _ in range(num_points):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            points.append((x, y))
            cv2.circle(frame_copy, (x, y), 3, (0, 0, 255), -1)
        
        # Conectar algunos puntos para formar una malla simple
        for i in range(len(points) - 1):
            for j in range(i + 1, min(i + 3, len(points))):
                cv2.line(frame_copy, points[i], points[j], (255, 0, 0), 1)
        
        return frame_copy

    def frames_to_tensor(self, frames):
        """Convertir frames de OpenCV a tensor de ComfyUI"""
        tensor_frames = []
        for frame in frames:
            # Convertir BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Normalizar y convertir a tensor
            frame_tensor = torch.from_numpy(frame_rgb.astype(np.float32) / 255.0)
            frame_tensor = frame_tensor.permute(2, 0, 1)  # HWC to CHW
            tensor_frames.append(frame_tensor)
        
        return torch.stack(tensor_frames)

# Registro del nodo
NODE_CLASS_MAPPINGS = {
    "Basic3DRigging": Basic3DRigging
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Basic3DRigging": "Basic 3D Rigging"
}

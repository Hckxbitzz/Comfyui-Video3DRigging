import numpy as np
import torch
import cv2
import json
import folder_paths
import os
from PIL import Image
import threading
import subprocess
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from sklearn.decomposition import PCA

# Para instalación de dependencias:
# pip install opencv-python-headless scipy scikit-learn matplotlib

class Video3DRigging:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "input.mp4"}),
                "operation_mode": (["extract_mesh", "rig_character", "integrate_3d"],),
                "tracking_method": (["optical_flow", "feature_matching", "deep_learning"],),
                "mesh_resolution": ("INT", {"default": 1000, "min": 100, "max": 10000, "step": 100}),
                "smoothness": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.1}),
                "keyframe_interval": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                "depth_map": ("IMAGE",),
                "mask": ("MASK",),
                "camera_params": ("STRING", {"default": "{}"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MESH", "STRING", "STRING")
    RETURN_NAMES = ("output_video", "3d_mesh", "animation_data", "debug_info")
    FUNCTION = "process_video"
    CATEGORY = "Video/3D"
    OUTPUT_NODE = True

    def __init__(self):
        self.detector = None
        self.tracker = None
        self.face_mesh = None
        self.pose_model = None
        
    def initialize_models(self):
        """Inicializar modelos de ML solo cuando sea necesario"""
        try:
            # Detector de puntos faciales
            if self.face_mesh is None:
                try:
                    import mediapipe as mp
                    mp_face_mesh = mp.solutions.face_mesh
                    self.face_mesh = mp_face_mesh.FaceMesh(
                        static_image_mode=False,
                        max_num_faces=1,
                        refine_landmarks=True,
                        min_detection_confidence=0.5
                    )
                except ImportError:
                    print("MediaPipe no instalado. Usando OpenCV alternativo")
                    self.face_mesh = None
            
            # Detector de pose
            if self.pose_model is None:
                try:
                    self.pose_model = cv2.dnn.readNetFromTensorflow(
                        "path/to/pose_model.pb",
                        "path/to/pose_model.pbtxt"
                    )
                except:
                    self.pose_model = None
                    
        except Exception as e:
            print(f"Error inicializando modelos: {e}")

    def extract_features(self, frame: np.ndarray) -> Dict:
        """Extraer características y puntos clave del frame"""
        features = {
            "face_landmarks": [],
            "body_joints": [],
            "optical_flow": None,
            "contours": []
        }
        
        # Convertir a RGB para MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detectar puntos faciales
        if self.face_mesh:
            results = self.face_mesh.process(frame_rgb)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = []
                    for idx, lm in enumerate(face_landmarks.landmark):
                        h, w, _ = frame.shape
                        x, y = int(lm.x * w), int(lm.y * h)
                        landmarks.append((x, y, lm.z))
                    features["face_landmarks"] = landmarks
        
        # Detectar contornos
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        features["contours"] = contours
        
        return features

    def generate_3d_mesh(self, points_2d: List, frame: np.ndarray, depth_hint: Optional[np.ndarray] = None) -> Dict:
        """Generar malla 3D a partir de puntos 2D"""
        if len(points_2d) < 3:
            return {"vertices": [], "faces": []}
        
        points_2d = np.array(points_2d, dtype=np.float32)
        
        # Estimación de profundidad simple (puede mejorarse con ML)
        if depth_hint is not None:
            depths = depth_hint[points_2d[:, 1].astype(int), points_2d[:, 0].astype(int)]
        else:
            # Depth estimation basado en posición en frame
            h, w = frame.shape[:2]
            center_x, center_y = w / 2, h / 2
            depths = 1.0 - (np.sqrt((points_2d[:, 0] - center_x)**2 + 
                                  (points_2d[:, 1] - center_y)**2) / max(w, h))
        
        points_3d = np.column_stack([points_2d, depths * 100])  # Escalar profundidad
        
        # Triangulación de Delaunay para crear malla
        try:
            tri = Delaunay(points_2d)
            mesh = {
                "vertices": points_3d.tolist(),
                "faces": tri.simplices.tolist(),
                "uv_coordinates": (points_2d / [w, h]).tolist()
            }
            return mesh
        except:
            # Fallback: malla simple
            return {
                "vertices": points_3d.tolist(),
                "faces": [],
                "uv_coordinates": (points_2d / [w, h]).tolist()
            }

    def create_rig(self, meshes: List[Dict]) -> Dict:
        """Crear sistema de rigging para la malla 3D"""
        rig = {
            "bones": [],
            "weights": [],
            "keyframes": [],
            "hierarchy": []
        }
        
        if not meshes:
            return rig
        
        # Análisis PCA para determinar huesos principales
        all_vertices = np.vstack([np.array(mesh["vertices"]) for mesh in meshes if mesh["vertices"]])
        
        if len(all_vertices) > 2:
            pca = PCA(n_components=3)
            pca.fit(all_vertices)
            
            # Crear huesos basados en componentes principales
            for i in range(min(3, pca.components_.shape[0])):
                direction = pca.components_[i]
                mean_point = np.mean(all_vertices, axis=0)
                
                bone = {
                    "name": f"bone_{i}",
                    "start": (mean_point - direction * 50).tolist(),
                    "end": (mean_point + direction * 50).tolist(),
                    "direction": direction.tolist()
                }
                rig["bones"].append(bone)
        
        return rig

    def integrate_3d_into_video(self, video_path: str, meshes: List[Dict], rig: Dict) -> str:
        """Integrar malla 3D en el video original usando OpenGL"""
        output_path = folder_paths.get_output_directory()
        output_video = os.path.join(output_path, "integrated_output.mp4")
        
        # Capturar video original
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Configurar video de salida
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx < len(meshes) and meshes[frame_idx]:
                # Renderizar malla 3D sobre el frame
                frame_with_mesh = self.render_mesh_on_frame(frame, meshes[frame_idx], rig)
                out.write(frame_with_mesh)
            else:
                out.write(frame)
            
            frame_idx += 1
        
        cap.release()
        out.release()
        
        return output_video

    def render_mesh_on_frame(self, frame: np.ndarray, mesh: Dict, rig: Dict) -> np.ndarray:
        """Renderizar malla 3D sobre el frame usando OpenCV"""
        frame_copy = frame.copy()
        
        # Dibujar vértices
        for vertex in mesh["vertices"]:
            x, y, z = int(vertex[0]), int(vertex[1]), int(vertex[2])
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                color = (0, 255, 0)  # Verde para vértices
                cv2.circle(frame_copy, (x, y), 2, color, -1)
        
        # Dibujar aristas
        for face in mesh.get("faces", []):
            if len(face) >= 3:
                pts = []
                for idx in face:
                    if idx < len(mesh["vertices"]):
                        x, y, z = mesh["vertices"][idx]
                        pts.append((int(x), int(y)))
                
                if len(pts) >= 3:
                    for i in range(len(pts)):
                        cv2.line(frame_copy, pts[i], pts[(i + 1) % len(pts)], (255, 0, 0), 1)
        
        # Dibujar huesos del rig
        for bone in rig.get("bones", []):
            start = tuple(map(int, bone["start"][:2]))
            end = tuple(map(int, bone["end"][:2]))
            cv2.line(frame_copy, start, end, (0, 0, 255), 2)
            cv2.circle(frame_copy, start, 3, (255, 255, 0), -1)
            cv2.circle(frame_copy, end, 3, (255, 0, 255), -1)
        
        return frame_copy

    def process_video(self, video_path: str, operation_mode: str, tracking_method: str,
                     mesh_resolution: int, smoothness: float, keyframe_interval: int,
                     reference_image: Optional[torch.Tensor] = None,
                     depth_map: Optional[torch.Tensor] = None,
                     mask: Optional[torch.Tensor] = None,
                     camera_params: str = "{}") -> Tuple:
        
        # Inicializar modelos
        self.initialize_models()
        
        # Verificar que el video existe
        if not os.path.exists(video_path):
            video_path = os.path.join(folder_paths.get_input_directory(), video_path)
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video no encontrado: {video_path}")
        
        # Procesar video
        cap = cv2.VideoCapture(video_path)
        meshes = []
        features_history = []
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % keyframe_interval == 0:
                # Extraer características
                features = self.extract_features(frame)
                features_history.append(features)
                
                # Generar malla 3D
                all_points = []
                if features["face_landmarks"]:
                    all_points.extend([(x, y) for x, y, z in features["face_landmarks"]])
                
                # Agregar puntos de contornos
                for contour in features["contours"]:
                    if len(contour) > 0:
                        for point in contour[::10]:  # Submuestreo
                            x, y = point[0]
                            all_points.append((x, y))
                
                if all_points:
                    depth_hint = None
                    if depth_map is not None and frame_idx < depth_map.shape[0]:
                        depth_hint = depth_map[frame_idx].numpy() if hasattr(depth_map, 'numpy') else depth_map
                    
                    mesh = self.generate_3d_mesh(all_points, frame, depth_hint)
                    meshes.append(mesh)
                else:
                    meshes.append({"vertices": [], "faces": []})
            
            frame_idx += 1
        
        cap.release()
        
        # Crear sistema de rigging
        rig = self.create_rig(meshes)
        
        # Integrar 3D en video
        output_video_path = self.integrate_3d_into_video(video_path, meshes, rig)
        
        # Cargar video procesado como tensor
        output_video = self.load_video_as_tensor(output_video_path)
        
        # Preparar datos de animación
        animation_data = json.dumps({
            "rig": rig,
            "mesh_count": len(meshes),
            "keyframe_interval": keyframe_interval,
            "features_per_frame": [len(f.get("face_landmarks", [])) for f in features_history]
        })
        
        debug_info = json.dumps({
            "processed_frames": frame_idx,
            "meshes_generated": len([m for m in meshes if m["vertices"]]),
            "total_vertices": sum(len(m["vertices"]) for m in meshes),
            "video_output": output_video_path
        })
        
        return (output_video, json.dumps(meshes), animation_data, debug_info)
    # Para GPU acceleration
    def enable_gpu_acceleration(self):
        try:
            import cupy as cp
            self.gpu_available = True
        except ImportError:
            self.gpu_available = False

# Para exportar a Blender
    def export_to_blender(self, meshes: List[Dict], rig: Dict):
        """Exportar mallas y rig a formato Blender"""
        blender_data = {
            "objects": [],
            "armatures": [],
            "animation_data": rig
    }
    # Implementación de exportación...
    def load_video_as_tensor(self, video_path: str) -> torch.Tensor:
        """Cargar video como tensor para ComfyUI"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convertir BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb.astype(np.float32) / 255.0)
            frame_tensor = frame_tensor.permute(2, 0, 1)  # HWC to CHW
            frames.append(frame_tensor)
        
        cap.release()
        
        if frames:
            return torch.stack(frames)
        else:
            return torch.zeros((1, 3, 256, 256))  # Tensor vacío

# Registro del nodo
NODE_CLASS_MAPPINGS = {
    "Video3DRigging": Video3DRigging
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Video3DRigging": "🎬 Video 3D Rigging"
}

"""
Webcam gesture -> meme detector (desktop version).

Opens two windows, side by side like the OBS/streamer setups:
  - "Camera": your webcam feed with hand landmarks drawn on top
  - "Meme": the cat meme matching whatever gesture you're making
"""

import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)
from mediapipe import Image, ImageFormat

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
MEMES = ROOT / "memes"

# --- Diccionario actualizado ---
# facu ojo
GESTURE_MEMES = {
    "default": ["default.jpg"],
    "spinCat": ["spin cat.mov"],
    "kiraPose": ["kira.jpg"],
    "josukePose": ["josuke.jpg"],
    "giornoPose": ["giorno.jpg"],
    "zeppeliPose": ["zeppeli.jpg"],
    "polnareffPose": ["polnareff.jpg"], 
    "jonathanPose": ["jonathan.webp"],  
    "crazyDiamondPose":["crazydiamond.jpg"], 
}

# gestures whose meme is a video, not a still image
VIDEO_GESTURES = {"spinCat"}

STABLE_FRAMES_REQUIRED = 5
DEFAULT_FALLBACK_MS = 600
FACE_STALE_MS = 1200

# side-eye and spin tuning parameters
SIDE_EYE_YAW_DEG = 15.0
SPIN_FLOW_WIDTH = 160
SPIN_FLOW_HEIGHT = 90
SPIN_FLOW_NOISE_FLOOR_PX = 0.4
SPIN_FLOW_MIN_MOVING_FRACTION = 0.15
SPIN_MAG_THRESHOLD = 0.8
SPIN_FRACTION_WINDOW_MS = 2200
SPIN_FRACTION_REQUIRED = 0.55
SPIN_FLOW_PEAK_HOLD_MS = 2000

HAND_COVER_FACE_DIST_FACE_LOST = 1.3
HAND_COVER_FACE_DIST_FACE_SEEN = 0.7

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# ---- geometry helpers -----------------------
def p3(lm):
    return np.array([lm.x, lm.y, lm.z])

def dist(a, b):
    return float(np.linalg.norm(a - b))

def angle_deg(v1, v2):
    m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if m1 < 1e-9 or m2 < 1e-9:
        return 180.0
    cos_a = np.clip(np.dot(v1, v2) / (m1 * m2), -1.0, 1.0)
    return math.degrees(math.acos(cos_a))

def finger_extended(pts, mcp, pip, tip):
    v1 = pts[pip] - pts[mcp]
    v2 = pts[tip] - pts[pip]
    return angle_deg(v1, v2) < 45

def yaw_from_transform_matrix(matrix):
    r = np.asarray(matrix)[:3, :3]
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    if sy < 1e-6:
        return 0.0
    yaw = math.atan2(-r[2, 0], sy)
    return math.degrees(yaw)

def classify_hand(landmarks):
    pts = [p3(lm) for lm in landmarks]
    hand_scale = dist(pts[0], pts[9]) or 1e-6

    index_up = finger_extended(pts, 5, 6, 8)
    middle_up = finger_extended(pts, 9, 10, 12)
    ring_up = finger_extended(pts, 13, 14, 16)
    pinky_up = finger_extended(pts, 17, 18, 20)

    thumb_pinky_spread = dist(pts[4], pts[17]) / hand_scale
    thumb_out = thumb_pinky_spread > 1.05

    curled_count = sum(1 for v in (index_up, middle_up, ring_up, pinky_up) if not v)

    return {
        "indexUp": index_up,
        "middleUp": middle_up,
        "ringUp": ring_up,
        "pinkyUp": pinky_up,
        "thumbOut": thumb_out,
        "curledCount": curled_count,
        "handScale": hand_scale,
        "indexTip": pts[8],
        "thumbTip": pts[4], 
        "wrist": pts[0],
        "palmCenter": pts[9],
    }

def frame_flow_signal(frame, prev_small_gray):
    small = cv2.resize(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (SPIN_FLOW_WIDTH, SPIN_FLOW_HEIGHT)
    )
    if prev_small_gray is None:
        return 0.0, 0.0, small

    flow = cv2.calcOpticalFlowFarneback(
        prev_small_gray, small, None, 0.5, 2, 15, 2, 5, 1.2, 0
    )
    flow_x = flow[..., 0]

    magnitude = float(np.abs(flow_x).mean())

    moving_mask = np.abs(flow_x) > SPIN_FLOW_NOISE_FLOOR_PX
    moving_count = int(moving_mask.sum())
    total = flow_x.size
    if moving_count / total < SPIN_FLOW_MIN_MOVING_FRACTION:
        coherence = 0.0
    else:
        mean_sign = np.sign(flow_x[moving_mask].mean())
        if mean_sign == 0:
            coherence = 0.0
        else:
            agree = int((np.sign(flow_x[moving_mask]) == mean_sign).sum())
            coherence = agree / moving_count

    return magnitude, coherence, small

class GestureState:
    def __init__(self):
        self.last_face = None
        self.face_seen_this_frame = False
        self.last_yaw_debug = 0.0
        self.flow_history = []
        self.flow_peak_history = []
        self.last_flow_magnitude_debug = 0.0
        self.last_flow_coherence_debug = 0.0
        self.last_flow_score_debug = 0.0
        self.last_flow_peak_debug = 0.0
        self.last_flow_fraction_debug = 0.0

    def update_flow(self, magnitude, coherence):
        now = time.time() * 1000
        score = magnitude * coherence 

        self.flow_history.append((now, magnitude))
        self.flow_history = [(t, m) for t, m in self.flow_history if now - t < SPIN_FRACTION_WINDOW_MS]

        self.flow_peak_history.append((now, score))
        self.flow_peak_history = [
            (t, s) for t, s in self.flow_peak_history if now - t < SPIN_FLOW_PEAK_HOLD_MS
        ]

        self.last_flow_magnitude_debug = magnitude
        self.last_flow_coherence_debug = coherence
        self.last_flow_score_debug = score
        self.last_flow_peak_debug = max((s for _, s in self.flow_peak_history), default=0.0)
        elevated = sum(1 for _, m in self.flow_history if m > SPIN_MAG_THRESHOLD)
        self.last_flow_fraction_debug = elevated / len(self.flow_history) if self.flow_history else 0.0

    def is_spinning(self, now):
        self.flow_history = [(t, m) for t, m in self.flow_history if now - t < SPIN_FRACTION_WINDOW_MS]
        if not self.flow_history:
            return False
        elevated = sum(1 for _, m in self.flow_history if m > SPIN_MAG_THRESHOLD)
        fraction = elevated / len(self.flow_history)
        return fraction > SPIN_FRACTION_REQUIRED

    def update_face(self, face_result):
        now = time.time() * 1000
        saw_face = bool(face_result.face_landmarks)

        if saw_face:
            f = face_result.face_landmarks[0]
            upper_lip, lower_lip = p3(f[13]), p3(f[14])
            right_cheek, left_cheek = p3(f[234]), p3(f[454])
            mouth_center = (upper_lip + lower_lip) / 2
            face_width = dist(right_cheek, left_cheek)
            mouth_open = dist(upper_lip, lower_lip) / face_width

            yaw_deg = 0.0
            if face_result.facial_transformation_matrixes:
                yaw_deg = yaw_from_transform_matrix(face_result.facial_transformation_matrixes[0])

            self.last_face = (mouth_center, face_width, mouth_open, yaw_deg, now)
            self.last_yaw_debug = yaw_deg
        self.face_seen_this_frame = saw_face

    def decide(self, hand_result):
        now = time.time() * 1000
        face_is_fresh = self.last_face is not None and now - self.last_face[4] < FACE_STALE_MS

        # 1. Recuperamos la detección de Spin
        if self.is_spinning(now):
            return "spinCat"

        if not hand_result.hand_landmarks:
            return "default"

        hands = [classify_hand(lm) for lm in hand_result.hand_landmarks]

        # INICIO DE GESTOS A 2 MANOS
        if len(hands) == 2:
            avg_scale = (hands[0]["handScale"] + hands[1]["handScale"]) / 2

            # facu ojo      

            # --- POSE DE POLNAREFF ---
            ambas_abiertas = hands[0]["curledCount"] <= 1 and hands[1]["curledCount"] <= 1
            distancia_palmas = dist(hands[0]["palmCenter"], hands[1]["palmCenter"]) / avg_scale
            manos_juntas = distancia_palmas < 1.8 

            def es_horizontal(h):
                return abs(h["indexTip"][0] - h["wrist"][0]) > abs(h["indexTip"][1] - h["wrist"][1])
            def es_vertical(h):
                return abs(h["indexTip"][1] - h["wrist"][1]) > abs(h["indexTip"][0] - h["wrist"][0])

            orientacion_cruzada = (es_horizontal(hands[0]) and es_vertical(hands[1])) or \
                                  (es_vertical(hands[0]) and es_horizontal(hands[1]))

            arriba_de_la_cabeza = True
            if face_is_fresh:
                mouth_center, face_width, _, _, _ = self.last_face
                head_top_y = mouth_center[1] - face_width * 0.8
                arriba_de_la_cabeza = hands[0]["palmCenter"][1] < head_top_y and hands[1]["palmCenter"][1] < head_top_y

            if ambas_abiertas and manos_juntas and orientacion_cruzada and arriba_de_la_cabeza:
                return "polnareffPose"

            # --- POSE DE ZEPPELI ---
            if hands[0]["palmCenter"][1] < hands[1]["palmCenter"][1]:
                mano_arriba, mano_abajo = hands[0], hands[1]
            else:
                mano_arriba, mano_abajo = hands[1], hands[0]

            dx_abajo = abs(mano_abajo["wrist"][0] - mano_abajo["palmCenter"][0])
            dy_abajo = abs(mano_abajo["wrist"][1] - mano_abajo["palmCenter"][1])
            abajo_horizontal = dx_abajo > dy_abajo

            abajo_abierta = mano_abajo["curledCount"] <= 1
            arriba_abierta_o_garra = mano_arriba["curledCount"] <= 2
            distancia_vertical = (mano_abajo["palmCenter"][1] - mano_arriba["palmCenter"][1]) / avg_scale
            separacion_notable = distancia_vertical > 1.2 

            if abajo_horizontal and abajo_abierta and arriba_abierta_o_garra and separacion_notable:
                return "zeppeliPose"

            # --- POSE DE GIORNO ---
            ambas_abiertas = hands[0]["curledCount"] == 0 and hands[1]["curledCount"] == 0
            pulgares_extendidos = hands[0]["thumbOut"] and hands[1]["thumbOut"]
            distancia_vertical = abs(hands[0]["palmCenter"][1] - hands[1]["palmCenter"][1]) / avg_scale
            desnivel_claro = distancia_vertical > 0.8 
            
            if ambas_abiertas and pulgares_extendidos and desnivel_claro:
                return "giornoPose"

            # --- POSE DE JOSUKE ---
            if hands[0]["palmCenter"][1] < hands[1]["palmCenter"][1]:
                mano_arriba, mano_abajo = hands[0], hands[1]
            else:
                mano_arriba, mano_abajo = hands[1], hands[0]

            arriba_cerrada = mano_arriba["curledCount"] >= 3
            abajo_abierta = mano_abajo["curledCount"] <= 1
            distancia_vertical = (mano_abajo["palmCenter"][1] - mano_arriba["palmCenter"][1]) / avg_scale
            separacion_suficiente = distancia_vertical > 1.5

            if arriba_cerrada and abajo_abierta and separacion_suficiente:
                return "josukePose"
            
            # --- POSE DE KIRA ---
            kira_hand_0_ready = hands[0]["indexUp"] and hands[0]["thumbOut"] and hands[0]["curledCount"] >= 2
            kira_hand_1_ready = hands[1]["indexUp"] and hands[1]["thumbOut"] and hands[1]["curledCount"] >= 2

            if kira_hand_0_ready and kira_hand_1_ready:
                dist_i0_t1 = dist(hands[0]["indexTip"], hands[1]["thumbTip"]) / avg_scale
                dist_t0_i1 = dist(hands[0]["thumbTip"], hands[1]["indexTip"]) / avg_scale

                if dist_i0_t1 < 1.5 and dist_t0_i1 < 1.5:
                    return "kiraPose"

# ==========================================
        # AQUi COMIENZAN LOS GESTOS DE UNA SOLA MANO
        # ==========================================
        
        h = hands[0]

        # --- NUEVA LoGICA: POSE DE JONATHAN (Una mano frente al rostro) ---
        # 1. La mano debe estar completamente abierta (0 dedos recogidos)
        mano_abierta = h["curledCount"] == 0
        
        # 2. Orientacion vertical (la muneca debe estar mas abajo que la palma)
        # Recordatorio: En pantalla, el eje Y aumenta hacia abajo.
        es_vertical = h["wrist"][1] > h["palmCenter"][1]

        # 3. Posicion cubriendo o frente al rostro
        frente_al_rostro = False
        if face_is_fresh:
            mouth_center, face_width, _, _, _ = self.last_face
            # Distancia normalizada entre la palma y el centro de la boca
            d_rostro = dist(h["palmCenter"], mouth_center) / face_width
            # Exigimos proximidad estricta al rostro
            frente_al_rostro = d_rostro < 0.7 

        if mano_abierta and es_vertical and frente_al_rostro:
            return "jonathanPose"

# --- NUEVA LÓGICA: POSE DE CRAZY DIAMOND (Mano horizontal sobre los ojos/frente) ---
        # 1. La mano debe estar completamente recta y abierta (0 dedos recogidos)
        mano_recta = h["curledCount"] == 0

        # 2. Orientación horizontal de la mano (como una visera)
        # Comparamos la distancia en el eje X frente al eje Y entre la muñeca y la punta del índice
        dx = abs(h["indexTip"][0] - h["wrist"][0])
        dy = abs(h["indexTip"][1] - h["wrist"][1])
        es_horizontal = dx > dy

        # 3. Posición cubriendo la zona de los ojos/frente
        frente_a_ojos = False
        if face_is_fresh:
            mouth_center, face_width, _, _, _ = self.last_face
            
            # La mano debe estar por encima de la boca (Y disminuye hacia arriba)
            arriba_de_boca = h["palmCenter"][1] < mouth_center[1]
            
            # La mano debe estar alineada horizontalmente con la cara
            alineada_x = abs(h["palmCenter"][0] - mouth_center[0]) < face_width * 0.8
            
            # La mano debe estar a la altura correcta (midiendo desde la boca hacia arriba)
            distancia_y_normalizada = abs(mouth_center[1] - h["palmCenter"][1]) / face_width
            # 0.4 a 1.1 suele abarcar desde el puente de la nariz hasta la frente alta
            altura_correcta = 0.4 < distancia_y_normalizada < 1.1

            frente_a_ojos = arriba_de_boca and alineada_x and altura_correcta

        if mano_recta and es_horizontal and frente_a_ojos:
            return "crazyDiamondPose"
        # --------------------------------------

        # ==========================================
        # FINAL DE LA FUNCIoN DECIDE
        # ==========================================


        return "default"

def load_memes():
    cache = {}
    for gesture, files in GESTURE_MEMES.items():
        if gesture in VIDEO_GESTURES:
            continue
        imgs = []
        for name in files:
            img = cv2.imread(str(MEMES / name))
            if img is None:
                raise FileNotFoundError(f"missing meme file: {MEMES / name}")
            imgs.append(img)
        cache[gesture] = imgs
    return cache

def draw_debug_hud(frame, state, gesture):
    lines = [
        f"gesture: {gesture}",
        f"yaw: {state.last_yaw_debug:+.1f} deg  (side-eye thr +/-{SIDE_EYE_YAW_DEG:.1f})",
        f"flow mag: {state.last_flow_magnitude_debug:.2f}  (thr {SPIN_MAG_THRESHOLD:.2f})",
        f"spin fraction (2.2s window): {state.last_flow_fraction_debug:.2f}  (thr {SPIN_FRACTION_REQUIRED:.2f})",
        f"peak score (last 2s): {state.last_flow_peak_debug:.2f}  <- read this AFTER you stop spinning",
    ]
    for i, line in enumerate(lines):
        y = 24 + i * 22
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 1, cv2.LINE_AA)

def draw_landmarks(frame, hand_result):
    h, w = frame.shape[:2]
    for hand in hand_result.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (80, 220, 120), 2)
        for x, y in pts:
            cv2.circle(frame, (x, y), 4, (60, 140, 255), -1)

def fit_to_height(img, height):
    h, w = img.shape[:2]
    scale = height / h
    return cv2.resize(img, (int(w * scale), height))

def main():
    hand_landmarker = HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODELS / "hand_landmarker.task")),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
        )
    )
    face_landmarker = FaceLandmarker.create_from_options(
        FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODELS / "face_landmarker.task")),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )
    )

    memes = load_memes()

    flow_log_path = ROOT / "flow_debug_log.csv"
    flow_log = open(flow_log_path, "w", buffering=1)
    flow_log.write("t_ms,magnitude,coherence,score,fraction,peak_2s,gesture\n")

    spin_video_cap = cv2.VideoCapture(str(MEMES / GESTURE_MEMES["spinCat"][0]))
    if not spin_video_cap.isOpened():
        raise FileNotFoundError(f"missing meme file: {MEMES / GESTURE_MEMES['spinCat'][0]}")

    def next_spin_frame():
        ok, vframe = spin_video_cap.read()
        if not ok:
            spin_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, vframe = spin_video_cap.read()
        return vframe

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0)")

    cv2.namedWindow("Camera")
    cv2.namedWindow("Meme")
    cv2.moveWindow("Camera", 40, 80)
    cv2.moveWindow("Meme", 720, 80)

    state = GestureState()
    current_gesture = "default"
    candidate_gesture = "default"
    candidate_streak = 0
    last_non_default_at = time.time() * 1000
    current_meme = random.choice(memes["default"])
    prev_flow_gray = None

    start_time = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            magnitude, coherence, prev_flow_gray = frame_flow_signal(frame, prev_flow_gray)
            state.update_flow(magnitude, coherence)
            flow_log.write(
                f"{time.time() * 1000:.0f},{magnitude:.4f},{coherence:.4f},"
                f"{state.last_flow_score_debug:.4f},{state.last_flow_fraction_debug:.4f},"
                f"{state.last_flow_peak_debug:.4f},{current_gesture}\n"
            )

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
            ts_ms = int((time.time() - start_time) * 1000)

            hand_result = hand_landmarker.detect_for_video(mp_image, ts_ms)
            face_result = face_landmarker.detect_for_video(mp_image, ts_ms)
            state.update_face(face_result)

            gesture = state.decide(hand_result)

            now = time.time() * 1000
            if gesture == candidate_gesture:
                candidate_streak += 1
            else:
                candidate_gesture = gesture
                candidate_streak = 1

            if candidate_streak >= STABLE_FRAMES_REQUIRED and gesture != current_gesture:
                current_gesture = gesture
                if gesture not in VIDEO_GESTURES:
                    current_meme = random.choice(memes[gesture])
                elif gesture == "spinCat":
                    spin_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            if gesture != "default":
                last_non_default_at = now
            elif now - last_non_default_at > DEFAULT_FALLBACK_MS and current_gesture != "default":
                current_gesture = "default"
                current_meme = random.choice(memes["default"])

            draw_landmarks(frame, hand_result)
            draw_debug_hud(frame, state, current_gesture)

            if current_gesture == "spinCat":
                vframe = next_spin_frame()
                meme_view = (
                    fit_to_height(vframe, frame.shape[0])
                    if vframe is not None
                    else fit_to_height(current_meme, frame.shape[0])
                )
            else:
                meme_view = fit_to_height(current_meme, frame.shape[0])
            cv2.imshow("Camera", frame)
            cv2.imshow("Meme", meme_view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        cap.release()
        spin_video_cap.release()
        flow_log.close()
        cv2.destroyAllWindows()
        hand_landmarker.close()
        face_landmarker.close()

if __name__ == "__main__":
    main()
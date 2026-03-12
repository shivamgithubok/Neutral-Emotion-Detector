"""
vision_engine.py — Elevatics AI Fall Detection v5
=============================================================
Multi-person tracking (up to 4) · Blendshape emotion (52 coeff)
Confidence from landmark visibility · Per-person ID + color
"""

import cv2, numpy as np, time, math, os, threading, urllib.request
from collections import deque, Counter
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

import mediapipe as mp

# Suppress MediaPipe library warnings (NORM_RECT width/height)
os.environ['GLOG_minloglevel'] = '2'

# ── MediaPipe aliases ─────────────────────────────────────────────────────────
BaseOptions            = mp.tasks.BaseOptions
PoseLandmarker         = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions  = mp.tasks.vision.PoseLandmarkerOptions
FaceLandmarker         = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions  = mp.tasks.vision.FaceLandmarkerOptions
RunningMode            = mp.tasks.vision.RunningMode
MPImage                = mp.Image
MPImageFormat          = mp.ImageFormat

# ── Model paths ───────────────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))

POSE_MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
                   "pose_landmarker/pose_landmarker_full/float16/latest/"
                   "pose_landmarker_full.task")
POSE_MODEL_PATH = os.path.join(_DIR, "pose_landmarker_full.task")

FACE_MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
                   "face_landmarker/face_landmarker/float16/latest/"
                   "face_landmarker.task")
FACE_MODEL_PATH = os.path.join(_DIR, "face_landmarker.task")

# ── Tracking constants ────────────────────────────────────────────────────────
MAX_PERSONS   = 4
IOU_THRESH    = 0.30   # min IoU to match existing track
TRACK_TTL     = 1.5    # seconds before dropping a track without detection
FACE_ASSOC_T  = 0.5    # max bbox overlap fraction to associate face→person

# ── BlazePose 33 indices ──────────────────────────────────────────────────────
NOSE       =  0
L_EYE      =  2;  R_EYE       =  5
L_EAR      =  7;  R_EAR       =  8
L_SHOULDER = 11;  R_SHOULDER  = 12
L_ELBOW    = 13;  R_ELBOW     = 14
L_WRIST    = 15;  R_WRIST     = 16
L_HIP      = 23;  R_HIP       = 24
L_KNEE     = 25;  R_KNEE      = 26
L_ANKLE    = 27;  R_ANKLE     = 28

KEY_LANDMARKS = [NOSE, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW,
                 L_WRIST, R_WRIST, L_HIP, R_HIP, L_KNEE, R_KNEE,
                 L_ANKLE, R_ANKLE]

SKELETON = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
    (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(27,29),(29,31),(27,31),
    (24,26),(26,28),(28,30),(30,32),(28,32),
]

# ── Per-track colour palette (BGR) ────────────────────────────────────────────
TRACK_COLORS = [
    (255, 200,   0),   # ID 1 — gold-cyan
    (  0, 140, 255),   # ID 2 — amber
    (  0, 255, 140),   # ID 3 — green
    (220,   0, 200),   # ID 4 — magenta
]

STATE_BGR = {
    "STANDING": (  0,255,140), "WALKING":  (255,230,  0),
    "RUNNING":  (  0,165,255), "SITTING":  (180,255,100),
    "LYING":    (200,  0,230), "SLEEPING": (150, 80,200),
    "FALLEN":   ( 30, 30,220), "UNKNOWN":  ( 70, 75, 85),
}
EXPR_BGR = {
    "HAPPY":     (  0,255,140), "LAUGHING":  (  0,200, 80),
    "SAD":       (180, 80, 40), "ANGRY":     ( 30, 30,220),
    "SURPRISED": (  0,220,255), "SCREAMING": ( 30, 30,255),
    "FEARFUL":   (  0,140,200), "DISGUSTED": ( 40,200,  0),
    "NEUTRAL":   (120,130,140), "UNKNOWN":   ( 70, 75, 85),
}
JOINT_COLORS = [
    (  0,255,200),(  0,240,255),(  0,200,255),(  0,160,255),
    (  0,120,255),(  0, 80,255),(  0, 40,255),(  0,255,160),
    (  0,255, 80),( 50,255,  0),(120,255,  0),(200,255,  0),
    (255,220,  0),(255,160,  0),(255, 80,  0),(255,  0, 80),
    (255,  0,160),(255,  0,240),(200,  0,255),(120,  0,255),
    ( 40,  0,255),(  0, 80,255),(  0,160,255),(  0,240,255),
    (  0,255,200),(  0,255,120),( 40,255,  0),(120,255,  0),
    (200,255,  0),(255,200,  0),(255,120,  0),(255, 40,  0),(200,0,200),
]
DIM_GREY   = ( 70, 75, 85)
ALERT_RED  = ( 30, 30,220)


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class FaceData:
    landmarks: list   = field(default_factory=list)
    blendshapes: dict = field(default_factory=dict)
    bbox:     tuple   = field(default_factory=tuple)
    valid:    bool    = False
    expression: str   = "UNKNOWN"
    expr_confidence: float = 0.0
    mouth_open:   float = 0.0
    brow_raise:   float = 0.0
    eye_open:     float = 0.0

@dataclass
class PersonStatus:
    track_id:     int   = 0
    state:        str   = "UNKNOWN"
    expression:   str   = "UNKNOWN"
    fall_score:   float = 0.0
    confidence:   float = 0.0      # 0–1 from visible landmarks
    lm_count:     int   = 0        # visible key landmarks out of 13
    lm_total:     int   = len(KEY_LANDMARKS)
    spine_angle:  float = 0.0
    aspect_ratio: float = 0.0
    mouth_open:   float = 0.0
    brow_raise:   float = 0.0
    eye_open:     float = 0.0
    expr_conf:    float = 0.0
    bbox:         tuple = field(default_factory=tuple)
    color:        tuple = field(default_factory=lambda: (200,200,200))


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────────────────
class Smoother:
    def __init__(self, alpha=0.25):
        self._v: Optional[float] = None; self.a = alpha
    def __call__(self, x):
        self._v = x if self._v is None else self.a*x + (1-self.a)*self._v
        return self._v


class ExpressionTemporalSmoother:
    """
    Confidence-weighted temporal smoother for expression predictions.

    Why this fixes SAD→HAPPY misclassification:
    ─────────────────────────────────────────────
    • Per-class score accumulation: instead of just majority-voting the label,
      we accumulate the raw confidence for every class across a rolling window.
      A single high-confidence HAPPY frame can't override 10 moderate SAD frames.
    • Minimum sustain gate: a new expression must hold for `min_sustain` frames
      before it is reported — eliminates single-frame flips during crying where
      the mouth momentarily curls upward (triggering HAPPY).
    • Recency bias: recent frames are weighted slightly more than old ones via
      a linear ramp so the smoother still reacts to genuine expression changes.
    • SAD/FEARFUL/ANGRY protection: negative emotions are penalised less during
      the accumulation so the bias toward HAPPY (which stems from class imbalance
      in FER training data) is counteracted.
    """
    # Emotions that need extra protection from being swamped by HAPPY
    _NEGATIVE  = {"SAD", "FEARFUL", "ANGRY", "DISGUSTED"}
    # Emotions that get a small prior boost to offset training-data imbalance
    _BOOST     = {"SAD": 0.08, "FEARFUL": 0.05, "ANGRY": 0.04, "DISGUSTED": 0.03}

    def __init__(self, window: int = 18, min_sustain: int = 4):
        """
        window      – number of frames to keep in the rolling window (~0.7s @ 25fps)
        min_sustain – frames the new winner must hold before being emitted
        """
        self._window      = window
        self._min_sustain = min_sustain
        # Rolling buffer: list of (label, confidence)
        self._buf: deque  = deque(maxlen=window)
        # Currently reported expression and how many frames it has been winning
        self._current_expr: str   = "NEUTRAL"
        self._current_conf: float = 0.0
        self._sustain_count: int  = 0
        self._candidate:    str   = "NEUTRAL"

    def update(self, expr: str, conf: float) -> Tuple[str, float]:
        """
        Feed one frame's prediction and return the smoothed (expression, confidence).
        """
        self._buf.append((expr, max(0.0, min(1.0, conf))))

        if not self._buf:
            return self._current_expr, self._current_conf

        n = len(self._buf)
        # Linear recency weights: oldest frame gets weight 1, newest gets weight n
        weights = np.arange(1, n + 1, dtype=np.float32)

        # Accumulate weighted confidence per class
        scores: Dict[str, float] = {}
        for i, (lbl, c) in enumerate(self._buf):
            w = float(weights[i])
            scores[lbl] = scores.get(lbl, 0.0) + c * w

        # Normalise by total weight so scores stay in [0,1] range
        total_w = float(weights.sum())
        scores  = {k: v / total_w for k, v in scores.items()}

        # Apply anti-HAPPY-bias boost for negative emotions
        for lbl, boost in self._BOOST.items():
            if lbl in scores:
                scores[lbl] = min(1.0, scores[lbl] + boost)

        best  = max(scores, key=lambda k: scores[k])
        bconf = scores[best]

        # Sustain gate: require the winner to hold for min_sustain frames
        if best == self._candidate:
            self._sustain_count += 1
        else:
            self._candidate     = best
            self._sustain_count = 1

        if self._sustain_count >= self._min_sustain:
            self._current_expr  = best
            self._current_conf  = round(bconf, 3)

        return self._current_expr, self._current_conf

    def reset(self):
        self._buf.clear()
        self._current_expr  = "NEUTRAL"
        self._current_conf  = 0.0
        self._sustain_count = 0
        self._candidate     = "NEUTRAL"


def _iou(a: tuple, b: tuple) -> float:
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    ua    = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / max(ua, 1)


def _bbox_from_lms(lms, w, h, pad=10):
    xs = [lm.x*w for lm in lms if lm.visibility > 0.25]
    ys = [lm.y*h for lm in lms if lm.visibility > 0.25]
    if not xs or not ys: return None
    return (max(0,int(min(xs))-pad), max(0,int(min(ys))-pad),
            min(w,int(max(xs))+pad), min(h,int(max(ys))+pad))


def _try_download(url, path, label) -> bool:
    if os.path.exists(path) and os.path.getsize(path) > 500_000:
        return True
    print(f"[Vision] Downloading {label}…")
    try:
        urllib.request.urlretrieve(url, path)
        if os.path.exists(path) and os.path.getsize(path) > 500_000:
            print(f"[Vision] {label} ready")
            return True
    except Exception as e:
        print(f"[Vision] Download failed ({label}): {e}")
        if os.path.exists(path): os.remove(path)
    return False


# ─────────────────────────────────────────────────────────────────────────────
# BLENDSHAPE EMOTION CLASSIFIER  (52-coefficient ARKit model)
# ─────────────────────────────────────────────────────────────────────────────
class BlendshapeEmotionClassifier:
    """
    Uses MediaPipe's 52 ARKit blendshape coefficients for emotion detection.
    Much more accurate than geometry ratios — each coefficient is trained
    by the MediaPipe model rather than hand-crafted.
    """
    _EMOTIONS = ["SCREAMING","LAUGHING","HAPPY","SURPRISED",
                 "FEARFUL","ANGRY","DISGUSTED","SAD","NEUTRAL"]

    def __init__(self):
        self._sm     = {e: Smoother(0.35) for e in self._EMOTIONS}
        self._tsm    = ExpressionTemporalSmoother(window=18, min_sustain=4)

    def classify(self, blendshapes: list) -> Tuple[str, float, dict]:
        """
        Args:
            blendshapes: list of Category objects from MediaPipe
        Returns:
            (expression, confidence, feature_dict)
        """
        if not blendshapes:
            return "UNKNOWN", 0.0, {}

        bs = {b.category_name: float(b.score) for b in blendshapes}

        # ── Aggregate named features ─────────────────────────────────────
        smile       = (bs.get("mouthSmileLeft",0) + bs.get("mouthSmileRight",0)) / 2
        frown       = (bs.get("mouthFrownLeft",0) + bs.get("mouthFrownRight",0)) / 2
        jaw_open    = bs.get("jawOpen", 0)
        eye_wide    = (bs.get("eyeWideLeft",0) + bs.get("eyeWideRight",0)) / 2
        eye_squint  = (bs.get("eyeSquintLeft",0) + bs.get("eyeSquintRight",0)) / 2
        eye_blink   = (bs.get("eyeBlinkLeft",0) + bs.get("eyeBlinkRight",0)) / 2
        brow_down   = (bs.get("browDownLeft",0) + bs.get("browDownRight",0)) / 2
        brow_in_up  = bs.get("browInnerUp", 0)
        brow_out_up = (bs.get("browOuterUpLeft",0) + bs.get("browOuterUpRight",0)) / 2
        nose_sneer  = (bs.get("noseSneerLeft",0) + bs.get("noseSneerRight",0)) / 2
        mouth_str   = (bs.get("mouthStretchLeft",0) + bs.get("mouthStretchRight",0)) / 2
        mouth_funnel= bs.get("mouthFunnel", 0)
        mouth_press = (bs.get("mouthPressLeft",0) + bs.get("mouthPressRight",0)) / 2
        cheek_sq    = (bs.get("cheekSquintLeft",0) + bs.get("cheekSquintRight",0)) / 2
        dimple      = (bs.get("mouthDimpleLeft",0) + bs.get("mouthDimpleRight",0)) / 2

        # ── Score each emotion (weighted sum, all 0–1) ───────────────────
        # SAD fix: crying activates cheekSquint (pushing cheeks up like a smile),
        # eyeBlink (from squeezing eyes), browInnerUp, and frown — but NOT dimple
        # or cheekSquint in the happy direction.  We add:
        #   • cheek_sq  as a SAD signal (raised wet cheeks)
        #   • mouth_press (lips pressed/trembling)
        #   • stronger eye_blink weight (screwed-shut eyes when crying)
        # HAPPY suppression: when strong SAD signals coexist with smile, penalise
        # HAPPY so the classifier doesn't fire on a "crying smile".
        sad_signals   = frown*0.35 + brow_in_up*0.28 + eye_blink*0.18 + \
                        cheek_sq*0.10 + mouth_press*0.09
        # Crying-smile guard: if SAD signals are strong AND smile is present,
        # penalise HAPPY score (smile during crying ≠ happiness)
        happy_penalty = max(0.0, sad_signals - 0.20) * 0.6

        raw = {
            "SCREAMING":  jaw_open*0.40 + mouth_str*0.30 + eye_wide*0.20 + brow_in_up*0.10,
            "LAUGHING":   smile*0.35   + eye_squint*0.30 + jaw_open*0.20 + cheek_sq*0.15,
            "HAPPY":      max(0.0, smile*0.45 + dimple*0.20 + cheek_sq*0.20 +
                              (1-frown)*0.15 - happy_penalty),
            "SURPRISED":  eye_wide*0.35+ brow_out_up*0.30+ jaw_open*0.25 + brow_in_up*0.10,
            "FEARFUL":    eye_wide*0.30 + brow_in_up*0.30+ mouth_str*0.25+ (1-smile)*0.15,
            "ANGRY":      brow_down*0.40+ nose_sneer*0.25+ mouth_press*0.20+(1-smile)*0.15,
            "DISGUSTED":  nose_sneer*0.45+mouth_funnel*0.25+brow_down*0.20+(1-smile)*0.10,
            "SAD":        sad_signals,
            "NEUTRAL":    0.20,
        }

        # Smooth scores per-emotion with EMA
        smoothed = {e: self._sm[e](raw[e]) for e in self._EMOTIONS}

        best_expr = max(smoothed, key=lambda k: smoothed[k])
        best_conf = smoothed[best_expr]

        # Require minimum confidence threshold; fall back to NEUTRAL
        if best_conf < 0.28:
            best_expr, best_conf = "NEUTRAL", 0.28

        # Temporal smoothing: confidence-weighted rolling window with sustain gate
        # This is the primary defence against SAD→HAPPY flips during crying
        best_expr, best_conf = self._tsm.update(best_expr, best_conf)

        features = {
            "mouth_open": round(jaw_open, 3),
            "brow_raise": round((brow_in_up + brow_out_up) / 2, 3),
            "eye_open":   round(max(0, 1 - eye_blink), 3),
        }
        return best_expr, round(best_conf, 3), features


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM PYTORCH AFFECTNET MODEL
# ─────────────────────────────────────────────────────────────────────────────
class AffectNetModel(threading.local().__class__): # Use a generic base or nn.Module if imported
    """
    Reverse-engineered architecture for Affectnet_model.pth.
    Matches the 11/9/9/7/5 kernel sequence and 7056 flatten size.
    """
    def __init__(self):
        # We'll import torch inside if needed, but since this class is used 
        # within DLBasedEmotionClassifier which already handles imports,
        # we assume torch and nn are available or will be.
        import torch.nn as nn
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=11), # 0
            nn.ReLU(inplace=True),            # 1
            nn.MaxPool2d(kernel_size=2),      # 2
            nn.Conv2d(32, 16, kernel_size=9), # 3
            nn.MaxPool2d(kernel_size=2),      # 4
            nn.Conv2d(16, 16, kernel_size=9, padding=4), # 5
            nn.ReLU(inplace=True),            # 6
            nn.Conv2d(16, 16, kernel_size=7, padding=3), # 7
            nn.ReLU(inplace=True),            # 8
            nn.Conv2d(16, 16, kernel_size=5, padding=2), # 9
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                     # 0
            nn.Linear(7056, 4096),            # 1
            nn.ReLU(inplace=True),            # 2
            nn.Dropout(0.5),                  # 3
            nn.Linear(4096, 7)                # 4
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# DEEP LEARNING EMOTION CLASSIFIER (Multi-Backend)
# ─────────────────────────────────────────────────────────────────────────────
class DLBasedEmotionClassifier:
    """
    Uses a MobileFaceNet ONNX model trained on FER datasets.
    Provides higher accuracy for primary emotion labels.
    """
    MAP = ["ANGRY", "DISGUSTED", "FEARFUL", "HAPPY", "NEUTRAL", "SAD", "SURPRISED"]

    def __init__(self, model_path: str):
        self.enabled = False
        if model_path.endswith(".onnx"):
            self.model_type = "ONNX"
        elif model_path.endswith(".pth"):
            self.model_type = "TORCH"
        else:
            self.model_type = "HDF5"
            
        self.model_path = model_path
        self.net = None
        self.tf_model = None
        self.torch_model = None
        self._tsm = ExpressionTemporalSmoother(window=15, min_sustain=3)

        if not os.path.exists(model_path):
            print(f"[Vision] Expression model not found: {model_path}")
            return
            
        try:
            if self.model_type == "ONNX":
                self.net = cv2.dnn.readNetFromONNX(model_path)
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                print(f"[Vision] ONNX Emotion model loaded: {model_path}")
            elif self.model_type == "TORCH":
                import torch
                from torchvision import transforms
                import torch.nn as nn
                
                # Assign the class to nn.Module if not already
                AffectNetModel.__bases__ = (nn.Module,)
                
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                # Load the state_dict into our reconstructed architecture
                try:
                    self.torch_model = AffectNetModel().to(self.device)
                    state_dict = torch.load(model_path, map_location=self.device)
                    
                    # If it's a full model save, extract state_dict
                    if not isinstance(state_dict, dict):
                        state_dict = state_dict.state_dict()
                        
                    self.torch_model.load_state_dict(state_dict)
                    self.torch_model.eval()
                    print(f"[Vision] PyTorch AffectNet model loaded: {model_path}")
                except Exception as load_err:
                    print(f"[Vision] Failed to load PyTorch state_dict: {load_err}")
                    # Fallback to direct load if it was actually a scripted model
                    try:
                        self.torch_model = torch.load(model_path, map_location=self.device)
                        if hasattr(self.torch_model, 'eval'):
                            self.torch_model.eval()
                        print(f"[Vision] PyTorch model loaded via direct load: {model_path}")
                    except:
                        self.enabled = False
                        raise load_err
                
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((112, 112)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                print(f"[Vision] PyTorch Emotion model loaded: {model_path}")
            else:
                # Lazy import tensorflow only when needed
                import tensorflow as tf
                self.tf_model = tf.keras.models.load_model(model_path)
                print(f"[Vision] HDF5 Emotion model loaded: {model_path}")
            self.enabled = True
        except Exception as e:
            print(f"[Vision] Failed to load {self.model_type} Emotion model: {e}")

    def classify(self, face_img: np.ndarray) -> Tuple[str, float]:
        if not self.enabled or face_img is None or face_img.size == 0:
            return "UNKNOWN", 0.0

        try:
            if self.model_type == "ONNX":
                # Preprocess: 112x112, subtract mean, scale
                blob = cv2.dnn.blobFromImage(face_img, 1.0/255.0, (112, 112), (0, 0, 0), swapRB=True, crop=False)
                self.net.setInput(blob)
                preds = self.net.forward()
            elif self.model_type == "TORCH":
                import torch
                img_t = self.transform(face_img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    output = self.torch_model(img_t)
                preds = output.cpu().numpy()
            else:
                # Preprocess for HDF5 (assuming 112x112 normalized [0,1])
                img_res = cv2.resize(face_img, (112, 112))
                img_res = img_res.astype(np.float32) / 255.0
                blob = np.expand_dims(img_res, axis=0) # (1, 112, 112, 3)
                preds = self.tf_model.predict(blob, verbose=0)
            
            # Softmax to get probabilities (if not already softmaxed)
            exp_preds = np.exp(preds - np.max(preds))
            probs = exp_preds / exp_preds.sum()
            
            idx = np.argmax(probs)
            raw_expr = self.MAP[idx]
            raw_conf = float(probs[0][idx])
            # Apply temporal smoother to DL output as well
            return self._tsm.update(raw_expr, raw_conf)
        except Exception as e:
            print(f"[Vision] {self.model_type} Inference error: {e}")
            return "UNKNOWN", 0.0

    # Fallback for when blendshapes are unavailable (geometry only)
    @staticmethod
    def classify_geometry(landmarks, w: int, h: int):
        """478-point geometry fallback if blendshapes are off."""
        if not landmarks or len(landmarks) < 468:
            return "UNKNOWN", 0.0, {}
        def pt(i): return landmarks[i].x*w, landmarks[i].y*h
        try:
            ml  = pt(61);  mr  = pt(291)
            mt  = pt(0);   mb  = pt(17)
            let = pt(159); leb = pt(145)
            ret = pt(386); reb = pt(374)
            lbr = pt(70);  rbr = pt(300)
            ns  = pt(4)
        except IndexError:
            return "NEUTRAL", 0.25, {}

        face_h   = max(abs(mb[1]-mt[1])*6, 50.0)
        mo       = abs(mb[1]-mt[1]) / face_h
        mw       = abs(mr[0]-ml[0]) / max(face_h,1)
        smile    = (ns[1] - (ml[1]+mr[1])/2) / max(face_h,1)
        eo       = ((abs(leb[1]-let[1])+abs(reb[1]-ret[1]))/2) / max(face_h,1)
        br       = (ns[1]-(lbr[1]+rbr[1])/2) / max(face_h,1)

        if mo > 0.22 and mw > 0.38 and eo > 0.06: expr = "SCREAMING"
        elif smile > 0.05 and mw > 0.36 and eo < 0.055 and mo > 0.10: expr = "LAUGHING"
        elif smile > 0.045 and mw > 0.30: expr = "HAPPY"
        elif br > 0.38 and eo > 0.065 and mo > 0.08: expr = "SURPRISED"
        elif smile < -0.01 and br < 0.28: expr = "SAD"
        elif br < 0.22 and mo < 0.08: expr = "ANGRY"
        else: expr = "NEUTRAL"

        return expr, 0.55, {"mouth_open": round(mo,3), "brow_raise": round(br,3), "eye_open": round(eo,3)}


# ─────────────────────────────────────────────────────────────────────────────
# POSE STATE CLASSIFIER  (one instance per tracked person)
# ─────────────────────────────────────────────────────────────────────────────
class StateClassifier:
    VIS_THRESH    = 0.45
    KEY_VIS_RATIO = 0.70   # relaxed for partial occlusion

    def __init__(self):
        self._cy_hist  = deque(maxlen=20)
        self._ts_hist  = deque(maxlen=20)
        self._state_h  = deque(maxlen=10)
        self._sm_fall  = Smoother(0.3)
        self._sm_ar    = Smoother(0.4)
        self._sm_vel   = Smoother(0.35)
        self._fallen_ts: Optional[float] = None

    def _vis_ratio(self, lms) -> Tuple[float, int]:
        vis = sum(1 for i in KEY_LANDMARKS
                  if i < len(lms) and lms[i].visibility > self.VIS_THRESH)
        return vis / len(KEY_LANDMARKS), vis

    def classify(self, lms, bbox, w, h, ts) -> PersonStatus:
        st = PersonStatus()
        if not lms: return st

        ratio, lm_count = self._vis_ratio(lms)
        st.confidence = ratio
        st.lm_count   = lm_count
        st.lm_total   = len(KEY_LANDMARKS)

        if ratio < self.KEY_VIS_RATIO:
            st.state = "UNKNOWN"
            return st

        def p(i): return lms[i].x*w, lms[i].y*h
        lhx,lhy = p(L_HIP)
        rhx,rhy = p(R_HIP)
        lax,lay = p(L_ANKLE)
        rax,ray = p(R_ANKLE)
        lsx,lsy = p(L_SHOULDER)
        rsx,rsy = p(R_SHOULDER)
        lkx,lky = p(L_KNEE)
        rkx,rky = p(R_KNEE)
        nx, ny  = p(NOSE)

        hip_y      = (lhy + rhy) / 2
        ankle_y    = (lay + ray) / 2
        shoulder_y = (lsy + rsy) / 2
        knee_y     = (lky + rky) / 2
        hip_cx     = (lhx + rhx) / 2
        sh_cx      = (lsx + rsx) / 2

        spine_ang = abs(math.degrees(math.atan2(
            abs(hip_cx-sh_cx), abs(hip_y-shoulder_y)+1e-9)))

        if bbox:
            x1,y1,x2,y2 = bbox
            ar   = max(1,x2-x1) / max(1,y2-y1)
        else:
            ar = 0.5

        ar_s = self._sm_ar(ar)
        self._cy_hist.append(hip_y)
        self._ts_hist.append(ts)

        hip_vel = 0.0
        if len(self._cy_hist) >= 4:
            ys  = np.array(list(self._cy_hist)[-4:])
            tss = np.array(list(self._ts_hist)[-4:])
            dts = np.diff(tss) + 1e-9
            hip_vel = float(np.mean(np.diff(ys)/dts))
        hip_vel_s = self._sm_vel(hip_vel)

        hip_ankle_n = abs(ankle_y - hip_y) / max(h, 1)

        # Fall score — weighted combination of 4 signals
        fs  = min(ar_s/1.6, 1.0)           * 0.35
        fs += min(spine_ang/65.0, 1.0)     * 0.30
        fs += min(max(hip_vel_s,0)/280, 1) * 0.20
        fs += (1 - min(hip_ankle_n/0.35,1))* 0.15
        fall_score = self._sm_fall(min(fs, 1.0))

        # Confirm fall after 1s sustained
        if fall_score > 0.58:
            if self._fallen_ts is None: self._fallen_ts = ts
            elif (ts - self._fallen_ts) >= 1.0:
                st.state = "FALLEN"; st.fall_score = fall_score
                st.spine_angle = spine_ang; st.aspect_ratio = ar_s
                return st
        else:
            self._fallen_ts = None

        if ar_s > 1.4 and spine_ang > 55:
            state = "SLEEPING" if hip_y/h > 0.65 else "LYING"
        elif knee_y < hip_y+10 and hip_ankle_n < 0.28:
            state = "SITTING"
        elif abs(hip_vel_s) > 120 and spine_ang < 30:
            state = "RUNNING"
        elif abs(hip_vel_s) > 40 and spine_ang < 35:
            state = "WALKING"
        else:
            state = "STANDING"

        self._state_h.append(state)
        if len(self._state_h) >= 4:
            state = Counter(list(self._state_h)[-6:]).most_common(1)[0][0]

        st.state = state; st.fall_score = fall_score
        st.spine_angle = spine_ang; st.aspect_ratio = ar_s
        return st


# ─────────────────────────────────────────────────────────────────────────────
# PERSON TRACK — one per tracked individual
# ─────────────────────────────────────────────────────────────────────────────
class PersonTrack:
    _next_id = 1

    def __init__(self):
        self.id    = PersonTrack._next_id
        PersonTrack._next_id += 1
        self.color = TRACK_COLORS[(self.id - 1) % len(TRACK_COLORS)]
        self.clf   = StateClassifier()
        self.bbox  : Optional[tuple] = None
        self.last_seen: float = time.time()
        self.status: PersonStatus = PersonStatus(track_id=self.id, color=self.color)

    def update(self, lms, bbox, face: FaceData, w, h, ts):
        self.bbox = bbox
        self.last_seen = ts
        st = self.clf.classify(lms, bbox, w, h, ts)
        st.track_id   = self.id
        st.color      = self.color
        st.bbox       = bbox or (0,0,0,0)
        st.expression = face.expression
        st.mouth_open = face.mouth_open
        st.brow_raise = face.brow_raise
        st.eye_open   = face.eye_open
        st.expr_conf  = face.expr_confidence
        self.status   = st
        return st


# ─────────────────────────────────────────────────────────────────────────────
# TRACKER POOL — IoU-based multi-person assignment
# ─────────────────────────────────────────────────────────────────────────────
class TrackerPool:
    def __init__(self, max_persons=MAX_PERSONS):
        self._tracks: List[PersonTrack] = []
        self._max = max_persons

    def update(self, detections: list, face_pool: list,
               w: int, h: int, ts: float) -> List[PersonStatus]:
        """
        detections: list of (lms, bbox) from MediaPipe
        face_pool:  list of FaceData (one per detected face)
        Returns:    list of PersonStatus (one per active track)
        """
        # ── Drop stale tracks ─────────────────────────────────────────────
        self._tracks = [t for t in self._tracks
                        if ts - t.last_seen < TRACK_TTL]

        # ── IoU matching: detection → existing track ──────────────────────
        unmatched_det = list(range(len(detections)))
        matched_tracks = set()

        if self._tracks and detections:
            # Build IoU matrix
            iou_mat = np.zeros((len(detections), len(self._tracks)))
            for di, (_, db) in enumerate(detections):
                for ti, trk in enumerate(self._tracks):
                    if db and trk.bbox:
                        iou_mat[di, ti] = _iou(db, trk.bbox)

            # Greedy assignment (best IoU first)
            while True:
                best = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
                if iou_mat[best] < IOU_THRESH:
                    break
                di, ti = best
                if di in unmatched_det and ti not in matched_tracks:
                    lms, bbox = detections[di]
                    face = _best_face(bbox, face_pool)
                    self._tracks[ti].update(lms, bbox, face, w, h, ts)
                    unmatched_det.remove(di)
                    matched_tracks.add(ti)
                iou_mat[best[0], :] = 0
                iou_mat[:, best[1]] = 0

        # ── Create new tracks for unmatched detections ────────────────────
        for di in unmatched_det:
            if len(self._tracks) < self._max:
                trk = PersonTrack()
                lms, bbox = detections[di]
                face = _best_face(bbox, face_pool)
                trk.update(lms, bbox, face, w, h, ts)
                self._tracks.append(trk)

        # ── Return statuses sorted by confidence desc ─────────────────────
        statuses = [t.status for t in self._tracks]
        statuses.sort(key=lambda s: (
            1 if s.state == "FALLEN" else 0,   # fallen first
            s.confidence
        ), reverse=True)
        return statuses


def _best_face(person_bbox: Optional[tuple], face_pool: List[FaceData]) -> FaceData:
    """Associate the best-matching face to a person bbox."""
    empty = FaceData()
    if not face_pool or not person_bbox:
        return face_pool[0] if face_pool else empty

    px1,py1,px2,py2 = person_bbox
    best_face, best_iou = empty, -1.0
    for fd in face_pool:
        if not fd.bbox: continue
        iou = _iou(person_bbox, fd.bbox)
        if iou > best_iou:
            best_iou = iou
            best_face = fd
    return best_face


# ─────────────────────────────────────────────────────────────────────────────
# HOG FALLBACK
# ─────────────────────────────────────────────────────────────────────────────
class HOGDetector:
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect_multi(self, frame) -> list:
        h, w = frame.shape[:2]
        scale = min(1.0, 640 / max(w, h))
        small = cv2.resize(frame, (int(w*scale), int(h*scale)))
        gray  = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        rects, _ = self.hog.detectMultiScale(gray, winStride=(8,8),
                                             padding=(8,8), scale=1.05)
        inv = 1.0 / scale
        results = []
        for (x,y,bw,bh) in rects:
            bbox = (int(x*inv), int(y*inv),
                    int((x+bw)*inv), int((y+bh)*inv))
            results.append((None, bbox))
        return results[:MAX_PERSONS]


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-PERSON HUD RENDERER
# ─────────────────────────────────────────────────────────────────────────────
class HUDRenderer:
    BRACKET_LEN = 22
    BRACKET_W   = 2

    def __init__(self):
        self._phase = 0.0

    def render(self, frame, detections_lms: list,
               statuses: List[PersonStatus], fps, frame_n) -> np.ndarray:
        self._phase = time.time() % (2*math.pi)
        img = self._vignette(frame)
        img = self._scanlines(img)
        h, w = img.shape[:2]

        # ── Per-person rendering ──────────────────────────────────────────
        has_fallen = any(s.state == "FALLEN" for s in statuses)
        for lms, bbox in detections_lms:
            if lms: img = self._skeleton(img, lms, w, h)

        for st in statuses:
            if st.bbox and any(st.bbox):
                img = self._bracket_box(img, st.bbox, st.color, st.track_id)
            if st.bbox and any(st.bbox) and st.state != "UNKNOWN":
                img = self._status_card(img, st, w, h)
            # Confidence badge in corner of bbox
            if st.bbox and any(st.bbox):
                img = self._confidence_badge(img, st)

        img = self._corner_hud(img, statuses, fps, frame_n)

        if has_fallen:
            img = self._alert_flash(img)

        return img

    # ── Atmosphere ────────────────────────────────────────────────────────
    @staticmethod
    def _vignette(frame):
        h, w = frame.shape[:2]
        Y,X  = np.ogrid[:h,:w]
        dist = np.sqrt(((X-w//2)/(w//2))**2 + ((Y-h//2)/(h//2))**2)
        mask = (1.0 - np.clip(dist*0.52, 0, 0.42))[...,np.newaxis]
        return (frame.astype(np.float32)*mask).astype(np.uint8)

    @staticmethod
    def _scanlines(frame):
        img = frame.copy()
        img[::2] = (img[::2].astype(np.uint16)*88//100).astype(np.uint8)
        return img

    # ── Skeleton ─────────────────────────────────────────────────────────
    def _skeleton(self, img, lms, w, h):
        pulse = 0.65 + 0.35*math.sin(self._phase*2.5)
        pts   = [(int(lm.x*w), int(lm.y*h)) for lm in lms]
        for s, e in SKELETON:
            if s>=len(pts) or e>=len(pts): continue
            if lms[s].visibility>0.3 and lms[e].visibility>0.3:
                col = tuple(int(c*0.55) for c in JOINT_COLORS[s%len(JOINT_COLORS)])
                cv2.line(img, pts[s], pts[e], col, 1, cv2.LINE_AA)
        for i,(px,py) in enumerate(pts):
            if i>=len(lms) or lms[i].visibility<0.35: continue
            base = JOINT_COLORS[i%len(JOINT_COLORS)]
            brt  = tuple(min(255,int(c*pulse)) for c in base)
            r    = 5 if i in (NOSE,L_HIP,R_HIP,L_SHOULDER,R_SHOULDER) else 3
            cv2.circle(img,(px,py),r+1,(0,0,0),-1)
            cv2.circle(img,(px,py),r,brt,-1,cv2.LINE_AA)
            cv2.circle(img,(px,py),r+3,tuple(int(c*0.3) for c in base),1,cv2.LINE_AA)
        return img

    # ── Bracket box (per-person color + ID) ──────────────────────────────
    def _bracket_box(self, img, bbox, col, track_id):
        x1,y1,x2,y2 = bbox
        pulse  = 0.7 + 0.3*math.sin(self._phase*3.0)
        bright = tuple(min(255,int(c*(0.8+0.2*pulse))) for c in col)
        dim    = tuple(int(c*0.35) for c in col)
        bl, bw = self.BRACKET_LEN, self.BRACKET_W

        # Thin side lines
        fade = tuple(int(c*0.4*pulse) for c in col)
        cv2.line(img,(x1,y1+bl),(x1,y2-bl),fade,1)
        cv2.line(img,(x2,y1+bl),(x2,y2-bl),fade,1)
        cv2.line(img,(x1+bl,y1),(x2-bl,y1),fade,1)
        cv2.line(img,(x1+bl,y2),(x2-bl,y2),fade,1)

        # Corner brackets
        def corner(ox,oy,dx,dy):
            cv2.line(img,(ox,oy),(ox+dx*bl,oy),dim,bw+3,cv2.LINE_AA)
            cv2.line(img,(ox,oy),(ox,oy+dy*bl),dim,bw+3,cv2.LINE_AA)
            cv2.line(img,(ox,oy),(ox+dx*bl,oy),bright,bw+1,cv2.LINE_AA)
            cv2.line(img,(ox,oy),(ox,oy+dy*bl),bright,bw+1,cv2.LINE_AA)
        corner(x1,y1,1,1); corner(x2,y1,-1,1)
        corner(x1,y2,1,-1); corner(x2,y2,-1,-1)

        # Track ID pill top-left
        font = cv2.FONT_HERSHEY_SIMPLEX
        id_txt = f"T{track_id}"
        tw = cv2.getTextSize(id_txt, font, 0.4, 1)[0][0]
        pill_x, pill_y = x1, y1-16
        cv2.rectangle(img,(pill_x,pill_y),(pill_x+tw+10,pill_y+14),col,-1)
        cv2.putText(img,id_txt,(pill_x+5,pill_y+11),font,0.38,(0,0,0),1,cv2.LINE_AA)

        # Scan line
        sy = y1 + int((time.time()%1.0)*(y2-y1))
        if 0<=sy<img.shape[0] and x2>x1:
            band = img[sy:sy+2,x1:x2].astype(np.float32)
            ov   = np.full_like(band, col, dtype=np.float32)
            img[sy:sy+2,x1:x2] = cv2.addWeighted(band,0.82,ov,0.18,0).astype(np.uint8)
        return img

    # ── Confidence badge (bottom of bbox) ────────────────────────────────
    def _confidence_badge(self, img, st: PersonStatus):
        if not st.bbox or not any(st.bbox): return img
        x1,y1,x2,y2 = st.bbox
        pct  = int(st.confidence * 100)
        text = f"{pct}% ({st.lm_count}/{st.lm_total}lm)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        tw, th = cv2.getTextSize(text, font, 0.32, 1)[0]
        bx, by = x1, y2+2
        # Background pill
        col  = st.color
        col_d = tuple(int(c*0.25) for c in col)
        cv2.rectangle(img,(bx,by),(bx+tw+8,by+th+5),col_d,-1)
        cv2.rectangle(img,(bx,by),(bx+tw+8,by+th+5),tuple(int(c*0.6) for c in col),1)
        # Confidence fill bar (full width = tw+8)
        bar_w = int((tw+8) * st.confidence)
        cv2.rectangle(img,(bx,by+th+2),(bx+bar_w,by+th+5),col,-1)
        cv2.putText(img,text,(bx+4,by+th+1),font,0.32,(220,228,235),1,cv2.LINE_AA)
        return img

    # ── Floating status card (per person) ────────────────────────────────
    def _status_card(self, img, st: PersonStatus, w, h):
        if not st.bbox or not any(st.bbox): return img
        x1,y1,x2,y2 = st.bbox
        cx = (x1+x2)//2; ny = y1

        col  = STATE_BGR.get(st.state, DIM_GREY)
        ecol = EXPR_BGR.get(st.expression, DIM_GREY)
        pulse = 0.88 + 0.12*math.sin(self._phase*2.0)

        cw, ch = 200, 76
        cardx = max(cw//2+6, min(w-cw//2-6, cx))
        cardy = max(ch+12, ny-45)
        x1c, y1c = cardx-cw//2, cardy-ch
        x2c, y2c = x1c+cw,      y1c+ch

        # Background
        ov = img.copy()
        cv2.rectangle(ov,(x1c,y1c),(x2c,y2c),(8,12,18),-1)
        cv2.addWeighted(ov,0.88,img,0.12,0,img)

        bright_c = tuple(min(255,int(c*pulse)) for c in col)
        cv2.rectangle(img,(x1c,y1c),(x2c,y1c+3),bright_c,-1)
        cv2.rectangle(img,(x1c,y1c),(x2c,y2c),tuple(int(c*0.5) for c in col),1,cv2.LINE_AA)

        font = cv2.FONT_HERSHEY_SIMPLEX
        W    = (228,232,238)

        # Header: "T1  BODY STATE"
        cv2.putText(img,f"T{st.track_id}",(x1c+8,y1c+15),font,0.35,col,1,cv2.LINE_AA)
        cv2.putText(img,"BODY STATE",(x1c+30,y1c+15),font,0.30,(70,85,95),1,cv2.LINE_AA)

        sv = st.state
        tw = cv2.getTextSize(sv,font,0.58,2)[0][0]
        cv2.putText(img,sv,(x1c+(cw-tw)//2+1,y1c+35),font,0.58,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(img,sv,(x1c+(cw-tw)//2,  y1c+35),font,0.58,W,      2,cv2.LINE_AA)
        cv2.putText(img,sv,(x1c+(cw-tw)//2,  y1c+35),font,0.58,bright_c,1,cv2.LINE_AA)

        cv2.line(img,(x1c+8,y1c+42),(x2c-8,y1c+42),(35,42,50),1)

        # Expression + confidence
        eb   = tuple(min(255,int(c*pulse)) for c in ecol)
        expr_txt = st.expression
        if st.expr_conf > 0:
            expr_txt += f" {int(st.expr_conf*100)}%"
        cv2.putText(img,"EXPRESSION",(x1c+8,y1c+55),font,0.28,(70,85,95),1,cv2.LINE_AA)
        etw = cv2.getTextSize(expr_txt,font,0.38,1)[0][0]
        cv2.putText(img,expr_txt,(x2c-etw-8+1,y1c+55),font,0.38,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(img,expr_txt,(x2c-etw-8,  y1c+55),font,0.38,eb,      1,cv2.LINE_AA)

        # Risk bar
        bx,by_,bw_,bh_ = x1c+8, y1c+62, cw-16, 5
        cv2.rectangle(img,(bx,by_),(bx+bw_,by_+bh_),(22,28,35),-1)
        fw = int(bw_*st.fall_score)
        r  = min(255,int(st.fall_score*2*255))
        g  = min(255,int((1-st.fall_score)*2*255))
        if fw>0: cv2.rectangle(img,(bx,by_),(bx+fw,by_+bh_),(20,g,r),-1)
        cv2.rectangle(img,(bx,by_),(bx+bw_,by_+bh_),(55,65,75),1)
        cv2.putText(img,f"RISK {int(st.fall_score*100)}%",
                    (x1c+8,y1c+75),font,0.26,(130,145,158),1,cv2.LINE_AA)

        # Connector
        cv2.line(img,(cardx,y2c),(cx,ny),tuple(int(c*0.35) for c in col),1,cv2.LINE_AA)
        return img

    # ── Corner HUD ────────────────────────────────────────────────────────
    @staticmethod
    def _corner_hud(img, statuses: List[PersonStatus], fps, fn):
        h, w  = img.shape[:2]
        font  = cv2.FONT_HERSHEY_SIMPLEX
        WHITE = (220,228,235); DIM = (80,95,108); ACCENT=(0,190,115)

        # Base rows
        rows: list = [
            ("FPS",   f"{fps:.1f}"),
            ("FRAME", f"{fn:06d}"),
            ("TRACK", f"{len(statuses)} person(s)"),
        ]
        for st in statuses[:4]:
            conf_pct = int(st.confidence*100)
            rows.append((f"T{st.track_id}",
                         f"{st.state[:4]}  {conf_pct}%lm"))

        pw, ph = 170, len(rows)*17+10
        px = w - pw - 4
        ov = img.copy()
        cv2.rectangle(ov,(px-2,2),(w-2,ph+2),(8,12,18),-1)
        cv2.addWeighted(ov,0.82,img,0.18,0,img)
        cv2.line(img,(px-2,2),(px-2,ph+2),(0,100,65),1)

        for i,(lbl,val) in enumerate(rows):
            y = 17+i*17
            cv2.putText(img,lbl,(px+2,y),font,0.32,DIM,1,cv2.LINE_AA)
            tw = cv2.getTextSize(val,font,0.34,1)[0][0]
            cv2.putText(img,val,(w-tw-5,y),font,0.34,WHITE,1,cv2.LINE_AA)

        # Brand bar
        bh_ = 20
        cv2.rectangle(img,(0,h-bh_),(w,h),(6,10,16),-1)
        cv2.putText(img,"ELEVATICS AI  //  MULTI-PERSON SAFETY v5",
                    (10,h-6),font,0.30,ACCENT,1,cv2.LINE_AA)
        return img

    @staticmethod
    def _alert_flash(img):
        h,w = img.shape[:2]
        p = int(85+65*math.sin(time.time()*7))
        ov = img.copy()
        cv2.rectangle(ov,(0,0),(w,h),(0,0,min(255,p+80)),8)
        cv2.addWeighted(ov,0.5,img,0.5,0,img)
        msg  = "  FALL DETECTED  "; font = cv2.FONT_HERSHEY_DUPLEX
        tw   = cv2.getTextSize(msg,font,0.95,2)[0][0]
        cv2.putText(img,msg,((w-tw)//2+2,52),font,0.95,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(img,msg,((w-tw)//2,50),  font,0.95,(50,50,255),2,cv2.LINE_AA)
        return img


# ─────────────────────────────────────────────────────────────────────────────
# VISION ENGINE  — main orchestrator
# ─────────────────────────────────────────────────────────────────────────────
class VisionEngine:
    def __init__(self):
        self._pose_lm: Optional[PoseLandmarker]  = None
        self._face_lm: Optional[FaceLandmarker]  = None
        self._dl_expr: Optional[DLBasedEmotionClassifier] = None
        
        # Default to VIT AffectNet PyTorch model
        dl_path = os.path.join(_DIR, "VIT_affectnet.pth")
        if not os.path.exists(dl_path):
            # Fallback to older AffectNet if VIT doesn't exist
            dl_path = os.path.join(_DIR, "Affectnet_model.pth")
            if not os.path.exists(dl_path):
                # Ultimate fallback to ONNX
                dl_path = os.path.join(_DIR, "facial_expression_recognition_mobilefacenet.onnx")
            
        self._dl_expr = DLBasedEmotionClassifier(dl_path)
        
        self.bs_clf   = BlendshapeEmotionClassifier()
        self.tracker  = TrackerPool(MAX_PERSONS)
        self.hud      = HUDRenderer()
        self.hog      = HOGDetector()
        
        self._use_pose  = False
        self._use_face  = False
        self._use_blend = False
        self._lock      = threading.Lock()

        self.frame_count = 0
        self.fps         = 0.0
        self._fps_ts: deque = deque(maxlen=30)

        self.state_counts: Dict[str,int] = {
            s:0 for s in ["STANDING","WALKING","RUNNING",
                           "SITTING","LYING","SLEEPING","FALLEN"]}
        self.expr_counts: Dict[str,int]  = {
            e:0 for e in ["HAPPY","LAUGHING","SAD","ANGRY",
                           "SURPRISED","SCREAMING","FEARFUL","DISGUSTED","NEUTRAL"]}
        self.event_log: list = []
        self._prev_states: Dict[int,str] = {}
        self._prev_exprs:  Dict[int,str] = {}

        self._init_models()

    def _init_models(self):
        # Pose (up to 4 persons)
        if _try_download(POSE_MODEL_URL, POSE_MODEL_PATH, "PoseLandmarker"):
            try:
                opts = PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
                    running_mode=RunningMode.VIDEO,
                    num_poses=MAX_PERSONS,
                    min_pose_detection_confidence=0.50,
                    min_pose_presence_confidence=0.50,
                    min_tracking_confidence=0.45,
                )
                self._pose_lm = PoseLandmarker.create_from_options(opts)
                self._use_pose = True
                print(f"[Vision] PoseLandmarker ✓  (max {MAX_PERSONS} persons)")
            except Exception as e:
                print(f"[Vision] Pose init failed: {e}")

        # Face + blendshapes (up to 4 faces)
        if _try_download(FACE_MODEL_URL, FACE_MODEL_PATH, "FaceLandmarker"):
            try:
                opts = FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
                    running_mode=RunningMode.VIDEO,
                    num_faces=MAX_PERSONS,
                    min_face_detection_confidence=0.45,
                    min_face_presence_confidence=0.45,
                    min_tracking_confidence=0.45,
                    output_face_blendshapes=True,
                )
                self._face_lm = FaceLandmarker.create_from_options(opts)
                self._use_face  = True
                self._use_blend = True
                print("[Vision] FaceLandmarker ✓  (blendshapes enabled)")
            except Exception as e:
                print(f"[Vision] Face init failed: {e}")

        if not self._use_pose: print("[Vision] HOG fallback active.")
        if not self._use_face: print("[Vision] Haar cascade fallback active.")

    def process_frame(self, frame: np.ndarray):
        now = time.time()
        self._fps_ts.append(now)
        if len(self._fps_ts) >= 2:
            elapsed = self._fps_ts[-1] - self._fps_ts[0]
            self.fps = (len(self._fps_ts)-1) / max(elapsed,1e-9)
        self.frame_count += 1
        h, w = frame.shape[:2]
        ts_ms = int(now*1000)

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = MPImage(image_format=MPImageFormat.SRGB, data=rgb)

        # ── Pose detection ────────────────────────────────────────────────
        detections = []
        if self._use_pose and self._pose_lm:
            with self._lock:
                try: res = self._pose_lm.detect_for_video(mp_img, ts_ms)
                except: res = None
            if res and res.pose_landmarks:
                for lms in res.pose_landmarks:
                    bbox = _bbox_from_lms(lms, w, h)
                    detections.append((lms, bbox))
        else:
            detections = self.hog.detect_multi(frame)

        # ── Face detection + emotion ──────────────────────────────────────
        face_pool: List[FaceData] = []
        if self._use_face and self._face_lm:
            with self._lock:
                try: fres = self._face_lm.detect_for_video(mp_img, ts_ms)
                except: fres = None
            if fres and fres.face_landmarks:
                for fi, flms in enumerate(fres.face_landmarks):
                    fxs = [lm.x*w for lm in flms]
                    fys = [lm.y*h for lm in flms]
                    fbbox = (max(0,int(min(fxs))-6), max(0,int(min(fys))-6),
                             min(w,int(max(fxs))+6), min(h,int(max(fys))+6))

                    # ── Emotion: ML + Blendshapes ────────────────────────
                    expr, econf, feats = "UNKNOWN", 0.0, {}
                    
                    # 1. Blendshapes (for features and specific states like LAUGHING)
                    if self._use_blend and fres.face_blendshapes and fi < len(fres.face_blendshapes):
                        expr_bs, conf_bs, feats = self.bs_clf.classify(fres.face_blendshapes[fi])
                    else:
                        expr_bs, conf_bs, feats = DLBasedEmotionClassifier.classify_geometry(flms, w, h)
                    
                    # 2. DL Model (High-accuracy primary emotions)
                    if self._dl_expr and self._dl_expr.enabled:
                        fx1, fy1, fx2, fy2 = fbbox
                        if fx2 > fx1 and fy2 > fy1:
                            crop = frame[fy1:fy2, fx1:fx2]
                            expr_dl, conf_dl = self._dl_expr.classify(crop)
                            
                            # ── Confidence-weighted fusion ───────────────
                            # Problem: DL model is biased toward HAPPY due to
                            # FER training imbalance.  Fix: when blendshapes
                            # indicate a negative emotion with reasonable conf,
                            # we trust blendshapes over DL for that specific
                            # conflict.  Otherwise use the higher-confidence source.
                            _NEGATIVE_EXPRS = {"SAD","FEARFUL","ANGRY","DISGUSTED"}
                            bs_is_negative  = expr_bs in _NEGATIVE_EXPRS
                            dl_is_happy     = expr_dl in ("HAPPY","LAUGHING")

                            if expr_bs in ("SCREAMING","LAUGHING") and conf_bs > 0.45:
                                # Blendshapes best for extreme mouth-open states
                                expr, econf = expr_bs, conf_bs
                            elif bs_is_negative and dl_is_happy and conf_bs > 0.30:
                                # Crying-smile guard: blendshapes caught a negative
                                # emotion but DL says HAPPY — trust blendshapes
                                expr, econf = expr_bs, conf_bs
                            elif conf_dl > conf_bs and conf_dl > 0.40:
                                expr, econf = expr_dl, conf_dl
                            else:
                                expr, econf = expr_bs, conf_bs
                    else:
                        expr, econf = expr_bs, conf_bs

                    fd = FaceData(
                        landmarks=flms, bbox=fbbox, valid=True,
                        expression=expr, expr_confidence=econf,
                        mouth_open=feats.get("mouth_open",0.0),
                        brow_raise=feats.get("brow_raise",0.0),
                        eye_open  =feats.get("eye_open",0.0),
                    )
                    if self._use_blend and fres.face_blendshapes and fi < len(fres.face_blendshapes):
                        fd.blendshapes = {b.category_name: float(b.score)
                                          for b in fres.face_blendshapes[fi]}
                    face_pool.append(fd)

        # ── Multi-person tracking ─────────────────────────────────────────
        statuses = self.tracker.update(detections, face_pool, w, h, now)

        # ── Log state/expression changes ──────────────────────────────────
        for st in statuses:
            prev_s = self._prev_states.get(st.track_id)
            if st.state != prev_s and st.state not in ("UNKNOWN",):
                self.state_counts[st.state] = self.state_counts.get(st.state,0)+1
                self.event_log.append({
                    "ts": time.strftime("%H:%M:%S"),
                    "type": "pose", "track_id": st.track_id,
                    "state": st.state, "expr": st.expression,
                    "score": round(st.fall_score,3),
                    "conf":  round(st.confidence,2),
                })
                self._prev_states[st.track_id] = st.state

            prev_e = self._prev_exprs.get(st.track_id)
            if st.expression != prev_e and st.expression not in ("UNKNOWN","NEUTRAL"):
                self.expr_counts[st.expression] = self.expr_counts.get(st.expression,0)+1
                self.event_log.append({
                    "ts": time.strftime("%H:%M:%S"),
                    "type": "expr", "track_id": st.track_id,
                    "state": st.state, "expr": st.expression,
                    "score": round(st.expr_conf,3),
                })
                self._prev_exprs[st.track_id] = st.expression

        if len(self.event_log) > 300:
            self.event_log = self.event_log[-200:]

        # ── Render ────────────────────────────────────────────────────────
        annotated = self.hud.render(
            frame, detections, statuses, self.fps, self.frame_count)

        # Primary status = highest priority (fallen > highest confidence)
        primary = statuses[0] if statuses else PersonStatus()
        return annotated, primary

    def set_expression_model(self, model_filename: str):
        """Switch the DL expression model at runtime."""
        path = os.path.join(_DIR, model_filename)
        if os.path.exists(path):
            self._dl_expr = DLBasedEmotionClassifier(path)
            print(f"[Vision] Switched to expression model: {model_filename}")
            return True
        print(f"[Vision] Model file not found: {path}")
        return False

    def get_persons_raw(self):
        """Return live PersonStatus objects for alert engine."""
        return [t.status for t in self.tracker._tracks]

    def get_stats(self) -> Dict:
        tracks = self.tracker._tracks
        persons_info = []
        for t in tracks:
            s = t.status
            persons_info.append({
                "id": s.track_id,
                "state": s.state,
                "expression": s.expression,
                "confidence": round(s.confidence,3),
                "lm_count": s.lm_count,
                "lm_total": s.lm_total,
                "fall_score": round(s.fall_score,3),
                "expr_conf": round(s.expr_conf,3),
            })
        return {
            "fps":          round(self.fps,1),
            "frame_count":  self.frame_count,
            "state_counts": self.state_counts,
            "expr_counts":  self.expr_counts,
            "event_log":    self.event_log[-50:],
            "persons":      persons_info,
            "num_tracked":  len(tracks),
            "mode":         ("MediaPipe" if self._use_pose else "HOG") +
                            (" + Blendshape" if self._use_blend else
                             " + Face" if self._use_face else ""),
        }

    def release(self):
        for lm in (self._pose_lm, self._face_lm):
            if lm:
                try: lm.close()
                except: pass
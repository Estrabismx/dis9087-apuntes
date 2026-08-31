import {
  HandLandmarker,
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";


// gestos
// facu ojo
const GESTURE_MEMES = {
  default: ["poses/default.jpg"],
  kiraPose: ["poses/kira.jpg"], 
  josukePose: ["poses/josuke.jpg"],
  giornoPose: ["poses/giorno.jpg"],
  zeppeliPose: ["poses/zeppeli.jpg"],
  polnareffPose: ["poses/polnareff.jpg"],
  jonathanPose: ["poses/jonathan.webp"],
  sideEyeDio: ["poses/dio.jpg"], // <-- AÑADIDO: Evita error si giras la cabeza y no hay manos
  crazyDiamondPose: ["poses/crazydiamond.jpg"],

};

const STABLE_FRAMES_REQUIRED = 5;
const DEFAULT_FALLBACK_MS = 600;
const FACE_STALE_MS = 1200;
const SIDE_EYE_YAW_DEG = 15.0;
const HAND_COVER_FACE_DIST_FACE_LOST = 1.3;
const HAND_COVER_FACE_DIST_FACE_SEEN = 0.7;

const video = document.getElementById("video");
const memeImg = document.getElementById("memeImg");
const debugHud = document.getElementById("debugHud");

let handLandmarker, faceLandmarker;
let lastVideoTime = -1;
let currentGesture = "default";
let candidateGesture = "default";
let candidateStreak = 0;
let lastNonDefaultAt = performance.now();
let lastFace = null; 
let lastFaceSeenThisFrame = false;
let lastYawDebug = 0;

async function init() {
  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  handLandmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
  });

  faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFacialTransformationMatrixes: true,
  });

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();

  requestAnimationFrame(loop);
}

function vec(a, b) {
  return { x: b.x - a.x, y: b.y - a.y, z: (b.z || 0) - (a.z || 0) };
}

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z || 0) - (b.z || 0));
}

function angleDeg(v1, v2) {
  const dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
  const m1 = Math.hypot(v1.x, v1.y, v1.z);
  const m2 = Math.hypot(v2.x, v2.y, v2.z);
  if (m1 < 1e-9 || m2 < 1e-9) return 180;
  return (Math.acos(Math.min(1, Math.max(-1, dot / (m1 * m2)))) * 180) / Math.PI;
}

function fingerExtended(lm, mcp, pip, tip) {
  const angle = angleDeg(vec(lm[mcp], lm[pip]), vec(lm[pip], lm[tip]));
  return angle < 45;
}

function yawFromTransformMatrix(matrixData) {
  const r00 = matrixData[0];
  const r10 = matrixData[4];
  const r20 = matrixData[8];
  const sy = Math.hypot(r00, r10);
  if (sy < 1e-6) return 0;
  return (Math.atan2(-r20, sy) * 180) / Math.PI;
}

function classifyHand(lm) {
  const handScale = dist(lm[0], lm[9]) || 1e-6; 

  const indexUp = fingerExtended(lm, 5, 6, 8);
  const middleUp = fingerExtended(lm, 9, 10, 12);
  const ringUp = fingerExtended(lm, 13, 14, 16);
  const pinkyUp = fingerExtended(lm, 17, 18, 20);

  const thumbPinkySpread = dist(lm[4], lm[17]) / handScale;
  const thumbOut = thumbPinkySpread > 1.05;

  const curledCount = [indexUp, middleUp, ringUp, pinkyUp].filter((v) => !v).length;

  return {
    indexUp,
    middleUp,
    ringUp,
    pinkyUp,
    thumbOut,
    curledCount,
    handScale,
    indexTip: lm[8],
    thumbTip: lm[4], 
    wrist: lm[0],
    palmCenter: lm[9],
  };
}

function updateFace(faceResult) {
  const now = performance.now();
  const sawFace = !!(faceResult.faceLandmarks && faceResult.faceLandmarks.length > 0);

  if (sawFace) {
    const f = faceResult.faceLandmarks[0];
    const upperLip = f[13];
    const lowerLip = f[14];
    const rightCheek = f[234];
    const leftCheek = f[454];
    const mouthCenter = {
      x: (upperLip.x + lowerLip.x) / 2,
      y: (upperLip.y + lowerLip.y) / 2,
      z: ((upperLip.z || 0) + (lowerLip.z || 0)) / 2,
    };
    const faceWidth = dist(rightCheek, leftCheek);
    const mouthOpen = dist(upperLip, lowerLip) / faceWidth;

    let yawDeg = 0;
    if (faceResult.facialTransformationMatrixes && faceResult.facialTransformationMatrixes.length > 0) {
      yawDeg = yawFromTransformMatrix(faceResult.facialTransformationMatrixes[0].data);
    }

    lastFace = { mouthCenter, faceWidth, mouthOpen, yawDeg, t: now };
    lastYawDebug = yawDeg;
  }
  lastFaceSeenThisFrame = sawFace;
}

function isPointing(h) {
  return h.indexUp && !h.middleUp && !h.ringUp && !h.pinkyUp;
}

function decideGesture(handResult) {
  const now = performance.now();
  const faceIsFresh = !!lastFace && now - lastFace.t < FACE_STALE_MS;

  if (!handResult.landmarks || handResult.landmarks.length === 0) {
    if (faceIsFresh && Math.abs(lastFace.yawDeg) > SIDE_EYE_YAW_DEG) {
      return "sideEyeDio";
    }
    return "default";
  }

  const hands = handResult.landmarks.map(classifyHand);

  if (hands.length === 2) {
    const avgScale = (hands[0].handScale + hands[1].handScale) / 2;

    if (faceIsFresh) {
      const { mouthCenter, faceWidth } = lastFace;
      const h0 = hands[0];
      const h1 = hands[1];

      // ojo facu
      // agregar poses de 2 manos

      // Pose de Polnareff
      if (h0.curledCount <= 1 && h1.curledCount <= 1) {
        const headTopY = mouthCenter.y - (faceWidth * 1.1);
        
        if (h0.palmCenter.y < headTopY && h1.palmCenter.y < headTopY) {
          const palmDist = dist(h0.palmCenter, h1.palmCenter) / faceWidth;
          
          const dir0 = vec(h0.wrist, h0.palmCenter);
          const dir1 = vec(h1.wrist, h1.palmCenter);
          
          const angle = angleDeg(dir0, dir1);

          if (palmDist < 1.5 && angle > 50 && angle < 130) {
            return "polnareffPose";
          }
        }
      }

      // Pose de Zeppeli
      if (h0.curledCount <= 1 && h1.curledCount <= 1) {
        const headTopY = mouthCenter.y - (faceWidth * 1.0); 
        const chestY = mouthCenter.y + (faceWidth * 0.5);   

        const handAbove = h0.palmCenter.y < headTopY ? h0 : (h1.palmCenter.y < headTopY ? h1 : null);
        const handBelow = h0.palmCenter.y > chestY ? h0 : (h1.palmCenter.y > chestY ? h1 : null);

        if (handAbove && handBelow && handAbove !== handBelow) {
          return "zeppeliPose";
        }
      }

      // Pose de Giorno
      if (h0.curledCount === 0 && h1.curledCount === 0) {
        const y0 = h0.palmCenter.y;
        const y1 = h1.palmCenter.y;

        const unaArribaUnaAbajo = (y0 < mouthCenter.y && y1 > mouthCenter.y) || 
                                  (y1 < mouthCenter.y && y0 > mouthCenter.y);
        
        const separacionVertical = Math.abs(y0 - y1) / faceWidth;

        if (unaArribaUnaAbajo && separacionVertical > 1.2) {
          return "giornoPose";
        }
      }

      // Pose de Josuke
      const checkJosuke = (hA, hB) => hA.curledCount === 0 && hB.curledCount >= 3;

      let openHand = null;
      let curledHand = null;

      if (checkJosuke(h0, h1)) {
        openHand = h0;
        curledHand = h1;
      } else if (checkJosuke(h1, h0)) {
        openHand = h1;
        curledHand = h0;
      }

      if (openHand && curledHand) {
        const curledDist = dist(curledHand.palmCenter, mouthCenter) / faceWidth;
        const openDist = dist(openHand.palmCenter, mouthCenter) / faceWidth;

        if (curledDist < 2.5 && openDist > 3.0 && openHand.palmCenter.y > mouthCenter.y) {
          return "josukePose";
        }
      }
    }

    // Pose de Kira
    const kiraHand0Ready = hands[0].indexUp && hands[0].thumbOut && hands[0].curledCount >= 2;
    const kiraHand1Ready = hands[1].indexUp && hands[1].thumbOut && hands[1].curledCount >= 2;

    if (kiraHand0Ready && kiraHand1Ready) {
      const distI0T1 = dist(hands[0].indexTip, hands[1].thumbTip) / avgScale;
      const distT0I1 = dist(hands[0].thumbTip, hands[1].indexTip) / avgScale;

      if (distI0T1 < 1.5 && distT0I1 < 1.5) {
        return "kiraPose";
      }
    }
  }

// Extraemos la primera mano para evaluar gestos de una sola mano
  const h = hands[0];

  // 1. PRIMERO evaluamos Crazy Diamond (porque sus reglas son más estrictas)
  if (h.curledCount === 0 && faceIsFresh) {
    const { mouthCenter, faceWidth } = lastFace;

    const d = dist(h.palmCenter, mouthCenter) / faceWidth;
    const isAboveMouth = h.palmCenter.y < mouthCenter.y;

    const deltaX = Math.abs(h.palmCenter.x - h.wrist.x);
    const deltaY = Math.abs(h.palmCenter.y - h.wrist.y);
    const isHorizontal = deltaX > (deltaY * 1.5);

    if (d < 1.5 && isAboveMouth && isHorizontal) {
      return "crazyDiamondPose";
    }
  }

  // 2. LUEGO evaluamos Jonathan (si la mano abierta no era horizontal, caerá aquí)
  if (h.curledCount === 0 && faceIsFresh) {
    const d = dist(h.palmCenter, lastFace.mouthCenter) / lastFace.faceWidth;
    
    const threshold = lastFaceSeenThisFrame
      ? HAND_COVER_FACE_DIST_FACE_SEEN
      : HAND_COVER_FACE_DIST_FACE_LOST;
    
    if (d < threshold) {
      return "jonathanPose";
    }
  }

  // <-- AÑADIDO: Retorno por defecto si hay 1 mano en pantalla o si 2 manos no coinciden con nada
  return "default"; 
}

function pickImage(gesture) {
  const images = GESTURE_MEMES[gesture];
  return images[Math.floor(Math.random() * images.length)];
}

function applyGesture(gesture) {
  if (gesture === currentGesture) return;
  currentGesture = gesture;
  memeImg.src = pickImage(gesture);
}

function loop() {
  const now = performance.now();
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const ts = performance.now();

    const handResult = handLandmarker.detectForVideo(video, ts);
    const faceResult = faceLandmarker.detectForVideo(video, ts);
    updateFace(faceResult);

    const gesture = decideGesture(handResult);

    if (gesture === candidateGesture) {
      candidateStreak++;
    } else {
      candidateGesture = gesture;
      candidateStreak = 1;
    }

    if (candidateStreak >= STABLE_FRAMES_REQUIRED) {
      applyGesture(gesture);
    }

    if (gesture !== "default") lastNonDefaultAt = now;
    if (now - lastNonDefaultAt > DEFAULT_FALLBACK_MS && currentGesture !== "default") {
      applyGesture("default");
    }

    updateDebugHud();
  }
  requestAnimationFrame(loop);
}

function updateDebugHud() {
  if (!debugHud) return;
  debugHud.textContent =
    `gesture: ${currentGesture}\n` +
    `yaw: ${lastYawDebug >= 0 ? "+" : ""}${lastYawDebug.toFixed(1)} deg  (side-eye thr +/-${SIDE_EYE_YAW_DEG.toFixed(1)})`;
}

init().catch((err) => console.error(err));
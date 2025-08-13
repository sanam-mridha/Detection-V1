import React, { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Camera, Play, Pause, RefreshCcw, ThumbsUp } from "lucide-react";
import { motion } from "framer-motion";

const FACE_TASK_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const HAND_TASK_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

let TasksVision;
let FaceLandmarker;
let HandLandmarker;
let FilesetResolver;

async function ensureTasksVision() {
  if (!TasksVision) {
    TasksVision = await import("@mediapipe/tasks-vision");
    FaceLandmarker = TasksVision.FaceLandmarker;
    HandLandmarker = TasksVision.HandLandmarker;
    FilesetResolver = TasksVision.FilesetResolver;
  }
}

function dist(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

function eyeAspectRatio(landmarks, isLeft) {
  const idx = isLeft ? { top: 159, bottom: 145, left: 33, right: 133 } : { top: 386, bottom: 374, left: 263, right: 362 };
  const top = landmarks[idx.top];
  const bottom = landmarks[idx.bottom];
  const left = landmarks[idx.left];
  const right = landmarks[idx.right];
  if (!top || !bottom || !left || !right) return 0;
  const v = dist(top, bottom);
  const h = dist(left, right);
  return h > 0 ? v / h : 0;
}

function mouthOpenRatio(landmarks) {
  const up = landmarks[13];
  const low = landmarks[14];
  const lc = landmarks[78];
  const rc = landmarks[308];
  if (!up || !low || !lc || !rc) return 0;
  const open = dist(up, low);
  const width = dist(lc, rc);
  return width > 0 ? open / width : 0;
}

function pinchDistance(hand) {
  const thumb = hand[4];
  const index = hand[8];
  if (!thumb || !index) return 0;
  return dist(thumb, index);
}

function headPose(landmarks) {
  const leftEye = landmarks[33];
  const rightEye = landmarks[263];
  const nose = landmarks[1];
  if (!leftEye || !rightEye || !nose) return { yaw: 0, pitch: 0, roll: 0 };
  const dx = rightEye.x - leftEye.x;
  const dy = rightEye.y - leftEye.y;
  const roll = Math.atan2(dy, dx);
  const midEyeX = (leftEye.x + rightEye.x) / 2;
  const yaw = (nose.x - midEyeX) * 2;
  const pitch = (nose.y - (leftEye.y + rightEye.y) / 2) * 2;
  return { yaw, pitch, roll };
}

function isThumbsUp(hand) {
  const thumbTip = hand[4];
  const indexTip = hand[8];
  const wrist = hand[0];
  if (!thumbTip || !indexTip || !wrist) return false;
  return thumbTip.y < wrist.y && pinchDistance(hand) > 0.1;
}

const smooth = (prev, next, k = 0.35) => prev * (1 - k) + next * k;

export default function VisionTrackerPro() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [running, setRunning] = useState(false);
  const [ready, setReady] = useState(false);
  const [fps, setFps] = useState(0);
  const [faceInfo, setFaceInfo] = useState({ leftEAR: 0, rightEAR: 0, blinkLeft: 0, blinkRight: 0, mouthOpen: 0, velocity: 0, faces: 0, yaw: 0, pitch: 0, roll: 0 });
  const [handInfo, setHandInfo] = useState({ hands: 0, pinch: 0, thumbsUp: false });
  const [confidence, setConfidence] = useState(0.5);
  const lastTimeRef = useRef(0);
  const lastCenterRef = useRef(null);
  const rafRef = useRef(0);
  const faceLandmarkerRef = useRef(null);
  const handLandmarkerRef = useRef(null);

  const initModels = async () => {
    await ensureTasksVision();
    const fileset = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm");
    const face = await FaceLandmarker.createFromOptions(fileset, { baseOptions: { modelAssetPath: FACE_TASK_URL }, runningMode: "VIDEO", numFaces: 2, outputFaceBlendshapes: true, minFaceDetectionConfidence: confidence, minTrackingConfidence: confidence });
    const hands = await HandLandmarker.createFromOptions(fileset, { baseOptions: { modelAssetPath: HAND_TASK_URL }, runningMode: "VIDEO", numHands: 2, minHandDetectionConfidence: confidence, minTrackingConfidence: confidence });
    faceLandmarkerRef.current = face;
    handLandmarkerRef.current = hands;
    setReady(true);
  };

  const startCamera = async () => {
    if (!ready) await initModels();
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    const video = videoRef.current;
    video.srcObject = stream;
    await video.play();
    setRunning(true);
    lastTimeRef.current = performance.now();
    loop();
  };

  const stopCamera = () => {
    setRunning(false);
    cancelAnimationFrame(rafRef.current);
    const stream = videoRef.current?.srcObject;
    if (stream) stream.getTracks().forEach((t) => t.stop());
  };

  const draw = (pred) => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const video = videoRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.save();
    ctx.scale(-1, 1);
    ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
    ctx.restore();
    const { face, hands } = pred;
    ctx.lineWidth = 1.5;
    ctx.font = "14px ui-sans-serif, system-ui";
    if (face?.landmarks?.length) {
      face.landmarks.forEach((lm) => {
        ctx.beginPath();
        for (const p of lm) {
          const x = canvas.width - p.x * canvas.width;
          const y = p.y * canvas.height;
          ctx.moveTo(x + 1, y);
          ctx.arc(x, y, 1.2, 0, Math.PI * 2);
        }
        ctx.stroke();
      });
    }
    if (hands?.length) {
      hands.forEach((hand) => {
        ctx.beginPath();
        hand.forEach((p) => {
          const x = canvas.width - p.x * canvas.width;
          const y = p.y * canvas.height;
          ctx.moveTo(x + 1, y);
          ctx.arc(x, y, 2, 0, Math.PI * 2);
        });
        ctx.stroke();
      });
    }
  };

  const loop = () => {
    const video = videoRef.current;
    if (!video || video.readyState < 2) {
      rafRef.current = requestAnimationFrame(loop);
      return;
    }
    const faceLmk = faceLandmarkerRef.current;
    const handLmk = handLandmarkerRef.current;
    if (!faceLmk || !handLmk) return;
    const now = performance.now();
    const dt = (now - lastTimeRef.current) / 1000;
    lastTimeRef.current = now;
    const faceResult = faceLmk.detectForVideo(video, now);
    const handResult = handLmk.detectForVideo(video, now);
    const faceLandmarks = faceResult?.faceLandmarks || [];
    let leftEAR = 0, rightEAR = 0, blinkL = 0, blinkR = 0, mouth = 0, vel = 0, yaw = 0, pitch = 0, roll = 0;
    if (faceLandmarks.length) {
      const lm = faceLandmarks[0];
      leftEAR = eyeAspectRatio(lm, true);
      rightEAR = eyeAspectRatio(lm, false);
      mouth = mouthOpenRatio(lm);
      const blends = faceResult?.faceBlendshapes?.[0]?.categories || [];
      const getBlend = (name) => blends.find((b) => b.categoryName === name)?.score ?? 0;
      blinkL = getBlend("eyeBlinkLeft");
      blinkR = getBlend("eyeBlinkRight");
      const cx = lm.reduce((s, p) => s + p.x, 0) / lm.length;
      const cy = lm.reduce((s, p) => s + p.y, 0) / lm.length;
      const center = { x: cx, y: cy };
      if (lastCenterRef.current) {
        const d = Math.hypot(center.x - lastCenterRef.current.x, center.y - lastCenterRef.current.y);
        vel = d / Math.max(dt, 1e-3);
      }
      lastCenterRef.current = center;
      const pose = headPose(lm);
      yaw = pose.yaw;
      pitch = pose.pitch;
      roll = pose.roll;
    }
    const hands = handResult?.landmarks || [];
    let pinch = 0;
    let thumbs = false;
    if (hands.length) {
      pinch = Math.max(...hands.map((h) => pinchDistance(h)));
      thumbs = hands.some((h) => isThumbsUp(h));
    }
    setFaceInfo((prev) => ({ faces: faceLandmarks.length, leftEAR: smooth(prev.leftEAR, leftEAR), rightEAR: smooth(prev.rightEAR, rightEAR), blinkLeft: smooth(prev.blinkLeft, blinkL), blinkRight: smooth(prev.blinkRight, blinkR), mouthOpen: smooth(prev.mouthOpen, mouth), velocity: smooth(prev.velocity, vel), yaw: smooth(prev.yaw, yaw), pitch: smooth(prev.pitch, pitch), roll: smooth(prev.roll, roll) }));
    setHandInfo((prev) => ({ hands: hands.length, pinch: smooth(prev.pinch, pinch), thumbsUp: thumbs }));
    draw({ face: { landmarks: faceLandmarks }, hands });
    setFps((prev) => smooth(prev, 1 / Math.max(dt, 1e-3), 0.25));
    if (running) rafRef.current = requestAnimationFrame(loop);
  };

  useEffect(() => {
    return () => {
      cancelAnimationFrame(rafRef.current);
      const stream = videoRef.current?.srcObject;
      if (stream) stream.getTracks().forEach((t) => t.stop());
    };
  }, []);

  const Metric = ({ label, value, fmt = (v) => v.toFixed(3) }) => (
    <div className="flex items-center justify-between py-1">
      <span className="text-sm text-muted-foreground">{label}</span>
      <span className="font-mono text-sm">{fmt(value)}</span>
    </div>
  );

  return (
    <div className="min-h-screen w-full bg-gradient-to-b from-zinc-50 to-white p-4 md:p-8">
      <div className="mx-auto max-w-6xl">
        <motion.h1 initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} className="text-2xl md:text-4xl font-bold tracking-tight mb-4">Vision Tracker Pro</motion.h1>
        <p className="text-muted-foreground mb-6 max-w-2xl">Real‑time face & hand recognition with blink, mouth, head pose, movement metrics, pinch detection and thumbs‑up detection.</p>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 lg:gap-6">
          <Card className="lg:col-span-2 overflow-hidden">
            <CardHeader className="flex flex-row items-center justify-between space-y-0">
              <CardTitle className="text-lg md:text-xl flex items-center gap-2"><Camera className="w-5 h-5"/> Camera</CardTitle>
              <div className="flex items-center gap-2">
                {!running ? (<Button onClick={startCamera} className="rounded-2xl"><Play className="mr-2 h-4 w-4"/>Start</Button>) : (<Button variant="secondary" onClick={stopCamera} className="rounded-2xl"><Pause className="mr-2 h-4 w-4"/>Pause</Button>)}
                <Button variant="ghost" onClick={() => window.location.reload()} className="rounded-2xl"><RefreshCcw className="mr-2 h-4 w-4"/>Reset</Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="relative aspect-video w-full bg-black/5 rounded-2xl overflow-hidden">
                <video ref={videoRef} className="absolute inset-0 h-full w-full object-cover" playsInline muted />
                <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />
                {!running && (<div className="absolute inset-0 grid place-items-center text-sm text-muted-foreground">Click Start to enable your camera</div>)}
                <div className="absolute left-3 bottom-3 text-xs bg-white/80 backdrop-blur px-2 py-1 rounded-full shadow">FPS: {fps.toFixed(1)}</div>
              </div>
            </CardContent>
          </Card>
          <div className="grid grid-cols-1 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Detection Settings</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-muted-foreground">Confidence</span>
                      <span className="text-xs font-mono">{confidence.toFixed(2)}</span>
                    </div>
                    <Slider min={0.1} max={0.9} step={0.05} defaultValue={[confidence]} onValueChange={(v) => setConfidence(v[0])} />
                    <p className="text-xs text-muted-foreground mt-2">(Takes effect next start)</p>
                  </div>
                </div>
              </CardContent>
            </Card>
            <Card className="rounded-2xl">
              <CardHeader>
                <CardTitle className="text-base">Face Metrics</CardTitle>
              </CardHeader>
              <CardContent>
                <Metric label="Faces" value={faceInfo.faces} fmt={(v) => v.toFixed(0)} />
                <Metric label="Left EAR" value={faceInfo.leftEAR} />
                <Metric label="Right EAR" value={faceInfo.rightEAR} />
                <Metric label="Blink Left" value={faceInfo.blinkLeft} />
                <Metric label="Blink Right" value={faceInfo.blinkRight} />
                <Metric label="Mouth Open Ratio" value={faceInfo.mouthOpen} />
                <Metric label="Head Velocity" value={faceInfo.velocity} />
                <Metric label="Yaw" value={faceInfo.yaw} />
                <Metric label="Pitch" value={faceInfo.pitch} />
                <Metric label="Roll" value={faceInfo.roll} />
              </CardContent>
            </Card>
            <Card className="rounded-2xl">
              <CardHeader>
                <CardTitle className="text-base">Hand Metrics</CardTitle>
              </CardHeader>
              <CardContent>
                <Metric label="Hands" value={handInfo.hands} fmt={(v) => v.toFixed(0)} />
                <Metric label="Max Pinch Distance" value={handInfo.pinch} />
                <div className="flex items-center justify-between py-1">

      

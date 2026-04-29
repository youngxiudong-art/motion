// 🛑 这里没有任何 import，没有任何依赖，代码绝对能跑起来！

const video = document.getElementById('webcam');
const status = document.getElementById('status');
const cameraSelect = document.getElementById('camera-select');
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const logPanel = document.getElementById('log-panel');
const faceOverlay = document.getElementById('face-overlay');
const entryPage = document.getElementById('entry-page');
const classroomPanel = document.getElementById('classroom-panel');
const studentNameInput = document.getElementById('student-name-input');
const enterClassBtn = document.getElementById('enter-class-btn');
const entryError = document.getElementById('entry-error');
const studentNameDisplay = document.getElementById('student-name-display');
const faceWarning = document.getElementById('face-warning');

let session = null;
let yunetSession = null;
let stream = null;
let analysisInterval = null;
let emotionHistory = [];
let currentStudentName = "";
let lastUploadTime = 0;
let faceLostTime = null;
const FACE_DETECTION_TIMEOUT_MS = 2000;
const MIN_UPLOAD_INTERVAL_MS = 2000;
const emotions = ["生气", "厌恶", "恐惧", "开心", "难过", "惊讶", "中性"];
const faceOverlayCtx = faceOverlay ? faceOverlay.getContext("2d") : null; // 动态初始化，在MediaPipe加载后更新

function softmax(logits) {
    const logitsArray = Array.from(logits);
    const max = Math.max(...logitsArray);
    const exps = logitsArray.map(v => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(v => (v / sum) * 100); // 返回百分比形式
}

function findMaxIndex(arr) {
    let maxIdx = 0, maxVal = arr[0];
    for (let i = 1; i < arr.length; i++) {
        if (arr[i] > maxVal) { maxVal = arr[i]; maxIdx = i; }
    }
    return maxIdx;
}

async function loadYunetModel() {
    let attempts = 0;
    while (!window.ort && attempts < 20) {
        await new Promise(r => setTimeout(r, 100));
        attempts++;
    }

    if (!window.ort) {
        throw new Error("ONNX Runtime 加载超时");
    }

    if (!yunetSession) {
        try {
            addLog("正在加载人脸检测模型...");
            yunetSession = await window.ort.InferenceSession.create('./face_detection_yunet_2023mar.onnx');
            addLog("✅ 人脸检测模型已加载", 'success');
        } catch (err) {
            addLog("❌ 人脸检测模型加载失败: " + err.message, 'error');
            throw err;
        }
    }
}

async function detectFaceYunet(canvas) {
    if (!yunetSession) return { detected: false };

    try {
        const inputSize = 640;
        const resizedCanvas = document.createElement('canvas');
        resizedCanvas.width = inputSize;
        resizedCanvas.height = inputSize;
        const resizedCtx = resizedCanvas.getContext('2d', { willReadFrequently: true });
        resizedCtx.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, inputSize, inputSize);

        const resizedData = resizedCtx.getImageData(0, 0, inputSize, inputSize).data;
        const input = new Float32Array(3 * inputSize * inputSize);

        // NCHW format: convert RGBA to RGB channels (keep 0-255 range)
        for (let i = 0; i < inputSize * inputSize; i++) {
            const r = resizedData[i * 4];
            const g = resizedData[i * 4 + 1];
            const b = resizedData[i * 4 + 2];

            input[i] = r;
            input[inputSize * inputSize + i] = g;
            input[2 * inputSize * inputSize + i] = b;
        }


        const tensor = new window.ort.Tensor('float32', input, [1, 3, inputSize, inputSize]);
        const output = await yunetSession.run({ input: tensor });

        // Parse multi-scale YOLO outputs
        const candidates = [];
        const scales = [8, 16, 32];

        for (const scale of scales) {
            const objKey = `obj_${scale}`;
            const bboxKey = `bbox_${scale}`;

            if (!output[objKey] || !output[bboxKey]) continue;

            const objData = output[objKey].data;
            const bboxData = output[bboxKey].data;
            const gridSize = inputSize / scale;

            for (let y = 0; y < gridSize; y++) {
                for (let x = 0; x < gridSize; x++) {
                    const idx = y * gridSize + x;
                    const confidence = objData[idx];

                    if (confidence < 0.5) continue;

                    // Extract bbox - offset from grid cell center
                    const bboxIdx = idx * 4;
                    const gridCellSize = scale;
                    const cellCenterX = (x + 0.5) * gridCellSize;
                    const cellCenterY = (y + 0.5) * gridCellSize;

                    let cx = cellCenterX + bboxData[bboxIdx] * gridCellSize;
                    let cy = cellCenterY + bboxData[bboxIdx + 1] * gridCellSize;
                    let w = bboxData[bboxIdx + 2] * gridCellSize;
                    let h = bboxData[bboxIdx + 3] * gridCellSize;

                    candidates.push({
                        x: Math.max(0, Math.min(1, (cx - w / 2) / inputSize)),
                        y: Math.max(0, Math.min(1, (cy - h / 2) / inputSize)),
                        w: Math.max(0, Math.min(1, w / inputSize)),
                        h: Math.max(0, Math.min(1, h / inputSize)),
                        confidence: confidence
                    });
                }
            }
        }

        if (candidates.length === 0) {
            return { detected: false };
        }

        // Sort by confidence and pick the best
        candidates.sort((a, b) => b.confidence - a.confidence);
        const bestFace = candidates[0];

        // Expand bbox by 40% to ensure complete face is captured
        const margin = 0.4;
        let left = Math.max(0, bestFace.x - margin / 2);
        let top = Math.max(0, bestFace.y - margin / 2);
        let right = Math.min(1, bestFace.x + bestFace.w + margin / 2);
        let bottom = Math.min(1, bestFace.y + bestFace.h + margin / 2);

        return {
            detected: true,
            box: {
                x: left,
                y: top,
                w: right - left,
                h: bottom - top
            },
            confidence: bestFace.confidence
        };
    } catch (err) {
        addLog("YuNet检测出错: " + err.message, 'error');
        console.error("YuNet error details:", err);
        return { detected: false };
    }
}

function averageEmotionHistory(frames) {
    if (frames.length === 0) return new Array(7).fill(0);
    const avg = new Array(7).fill(0);
    frames.forEach(frame => {
        for (let i = 0; i < 7; i++) avg[i] += frame[i];
    });
    for (let i = 0; i < 7; i++) avg[i] /= frames.length;
    return avg;
}

function addLog(message, type = 'info') {
    if (!logPanel) return;
    const time = new Date().toLocaleTimeString();
    const div = document.createElement('div');
    let colorClass = '';
    if (type === 'error') colorClass = 'log-error';
    if (type === 'success') colorClass = 'log-success';
    div.innerHTML = `<span style="color: #888;">[${time}]</span> <span class="${colorClass}">${message}</span>`;
    logPanel.appendChild(div);
    logPanel.scrollTop = logPanel.scrollHeight;
}

function updateEmotionUI(probs) {
    if (!probs || probs.length !== 7) return;
    emotions.forEach((emotion, idx) => {
        const bar = document.getElementById(`emotion-bar-${idx}`);
        const pct = document.getElementById(`emotion-pct-${idx}`);
        if (bar && pct) {
            const percent = Math.round(probs[idx]);
            bar.style.width = percent + '%';
            pct.textContent = percent + '%';
        }
    });
}

function updateFaceWarning(show) {
    if (!faceWarning) return;
    if (show) {
        faceWarning.classList.add('hidden');
    } else {
        faceWarning.classList.remove('hidden');
    }
}

// 探针
addLog("✅ 核心识别脚本已成功加载，准备唤醒摄像头...", "success");

async function getCameras() {
    try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error("浏览器不支持摄像头，请用手机自带浏览器打开！");
        }
        addLog("正在请求摄像头权限...", "info");
        await navigator.mediaDevices.getUserMedia({ video: true });
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');
        
        if (cameraSelect) {
            cameraSelect.innerHTML = '';
            videoDevices.forEach(device => {
                const option = document.createElement('option');
                option.value = device.deviceId;
                option.text = device.label || `摄像头 ${cameraSelect.length + 1}`;
                cameraSelect.appendChild(option);
            });
        }
        if (status) status.innerText = "设备检测完成，请点击开启。";
        addLog("🎉 摄像头就绪！", 'success');
    } catch (e) {
        if (status) status.innerText = "⚠️ 无法获取摄像头权限";
        addLog("❌ 权限错误: " + e.message, 'error');
    }
}

async function startSystem() {
    try {
        addLog("正在启动摄像头...");
        const deviceId = cameraSelect ? cameraSelect.value : undefined;
        stream = await navigator.mediaDevices.getUserMedia({
            video: { deviceId: deviceId ? { exact: deviceId } : undefined }
        });
        if (video) video.srcObject = stream;

        if (!session) {
            try {
                addLog("载入 emotion.onnx 模型...");
                session = await window.ort.InferenceSession.create('./emotion.onnx');
                addLog("✅ 情绪识别模型已加载", 'success');
            } catch (err) {
                addLog("❌ 情绪识别模型加载失败: " + err.message, 'error');
                throw err;
            }
        }

        await loadYunetModel();

        if (status) status.innerText = "✅ 正在实时监测中...";
        addLog("系统启动成功，开始监测！", 'success');
        startAnalysis();
    } catch (e) {
        addLog("启动失败: " + e.message, 'error');
        console.error("Detailed error:", e);
    }
}

function startAnalysis() {
    const croppedCanvas = document.createElement('canvas');
    croppedCanvas.width = 48;
    croppedCanvas.height = 48;
    const croppedCtx = croppedCanvas.getContext('2d');

    const detectionCanvas = document.createElement('canvas');
    detectionCanvas.width = 640;
    detectionCanvas.height = 480;

    emotionHistory = [];
    lastUploadTime = Date.now();
    faceLostTime = null;

    if (analysisInterval) clearInterval(analysisInterval);

    analysisInterval = setInterval(async () => {
        if (!session || !stream || !video || !yunetSession) return;

        try {
            const detectionCtx = detectionCanvas.getContext('2d');
            detectionCtx.drawImage(video, 0, 0, 640, 480);

            const yunetResult = await detectFaceYunet(detectionCanvas);

            if (!yunetResult.detected) {
                if (!faceLostTime) {
                    faceLostTime = Date.now();
                    updateFaceWarning(false);
                    addLog("⏳ 等待人脸...", 'info');
                }
                const now = Date.now();
                if (now - faceLostTime > FACE_DETECTION_TIMEOUT_MS) {
                    emotionHistory = [];
                    addLog("⚠️ 2秒内未检测到人脸，暂停分析", 'error');
                    if (status) status.innerText = "未检测到人脸，等待中...";
                }
                return;
            }

            // 人脸重新出现时，重置丢失时间
            if (faceLostTime !== null) {
                faceLostTime = null;
                addLog("✅ 检测到人脸，继续识别", 'success');
                emotionHistory = [];
                lastUploadTime = Date.now();
            }

            faceLostTime = null;
            updateFaceWarning(true);

            if (syncFaceOverlaySize()) {
                drawFaceBox(yunetResult.box, faceOverlay.width, faceOverlay.height);
            }

            const box = yunetResult.box;

            const cropX = Math.max(0, Math.floor(box.x * video.videoWidth));
            const cropY = Math.max(0, Math.floor(box.y * video.videoHeight));
            const cropW = Math.min(video.videoWidth - cropX, Math.floor(box.w * video.videoWidth));
            const cropH = Math.min(video.videoHeight - cropY, Math.floor(box.h * video.videoHeight));

            croppedCtx.clearRect(0, 0, 48, 48);
            croppedCtx.drawImage(video, cropX, cropY, cropW, cropH, 0, 0, 48, 48);

            const imgData = croppedCtx.getImageData(0, 0, 48, 48).data;
            const contrast = 1.3;
            const intercept = 128 * (1 - contrast);
            const input = new Float32Array(3 * 48 * 48);

            for (let i = 0; i < 48 * 48; i++) {
                for (let c = 0; c < 3; c++) {
                    let val = imgData[i * 4 + c];
                    val = val * contrast + intercept;
                    input[i + c * 2304] = Math.max(0, Math.min(255, val)) / 255.0;
                }
            }

            const tensor = new window.ort.Tensor('float32', input, [1, 3, 48, 48]);
            const output = await session.run({ images: tensor });
            const results = output[Object.keys(output)[0]].data;

            const probs = softmax(results);
            const maxIdx = findMaxIndex(probs);
            const confidence = probs[maxIdx] / 100;

            updateEmotionUI(probs);

            emotionHistory.push(probs);
            if (emotionHistory.length > 4) emotionHistory.shift();

            let requiredFrames = confidence > 0.5 ? 1 : confidence > 0.3 ? 2 : 4;
            const now = Date.now();

            if (emotionHistory.length >= requiredFrames && (now - lastUploadTime) >= MIN_UPLOAD_INTERVAL_MS) {
                const recentFrames = emotionHistory.slice(-requiredFrames);
                const avgProbs = averageEmotionHistory(recentFrames);
                const avgMaxIdx = findMaxIndex(avgProbs);
                const resultLabel = emotions[avgMaxIdx];
                const avgConfidence = avgProbs[avgMaxIdx];

                addLog(`📤 上传: [${resultLabel}] 置信度: ${(avgConfidence).toFixed(1)}%`);

                if (window.uploadFocusData) {
                    await window.uploadFocusData(currentStudentName || "Student_A", avgProbs);
                }

                lastUploadTime = now;
            }

        } catch (err) {
            addLog("分析出错: " + err.message, 'error');
        }
    }, 1500);
}

function stopSystem() {
    if (analysisInterval) clearInterval(analysisInterval);
    analysisInterval = null;
    if (stream) stream.getTracks().forEach(t => t.stop());
    if (video) video.srcObject = null;
    emotionHistory = [];
    faceLostTime = null;
    clearFaceOverlay();
    if (faceWarning) faceWarning.classList.add('hidden');
    if (status) status.innerText = "⏹ 已停止";
    addLog("系统已手动关闭");
}

function clearFaceOverlay() {
    if (!faceOverlayCtx || !faceOverlay) return;
    faceOverlayCtx.clearRect(0, 0, faceOverlay.width, faceOverlay.height);
}

function syncFaceOverlaySize() {
    if (!faceOverlay || !video) return false;
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh) return false;
    if (faceOverlay.width !== vw || faceOverlay.height !== vh) {
        faceOverlay.width = vw;
        faceOverlay.height = vh;
    }
    return true;
}

function drawFaceBox(box, canvasWidth, canvasHeight) {
    if (!faceOverlayCtx || !faceOverlay) return;

    clearFaceOverlay();
    if (!box) return;

    faceOverlayCtx.strokeStyle = '#00ff00';
    faceOverlayCtx.lineWidth = 2;
    faceOverlayCtx.strokeRect(
        box.x * canvasWidth,
        box.y * canvasHeight,
        box.w * canvasWidth,
        box.h * canvasHeight
    );
}


function normalizeStudentName(name) {
    return (name || "").trim();
}

async function enterClassroom() {
    const inputName = normalizeStudentName(studentNameInput ? studentNameInput.value : "");
    if (!inputName) {
        if (entryError) entryError.innerText = "请输入姓名后再进入课堂。";
        if (studentNameInput) studentNameInput.focus();
        return;
    }

    currentStudentName = inputName;
    if (entryError) entryError.innerText = "";
    if (studentNameDisplay) studentNameDisplay.innerText = `当前学生：${currentStudentName}`;
    if (entryPage) entryPage.classList.add("hidden");
    if (classroomPanel) classroomPanel.classList.remove("hidden");

    addLog(`👋 欢迎 ${currentStudentName}，正在初始化课堂设备...`, "success");
    await getCameras();
    await startSystem();
}

if (startBtn) startBtn.onclick = startSystem;
if (stopBtn) stopBtn.onclick = stopSystem;
if (enterClassBtn) {
    enterClassBtn.onclick = enterClassroom;
}
if (studentNameInput) {
    studentNameInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
            enterClassroom();
        }
    });
}

// 进入页默认激活输入框，不提前触发摄像头流程
if (studentNameInput) {
    setTimeout(() => studentNameInput.focus(), 200);
}
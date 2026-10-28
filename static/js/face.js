/* Shared camera helpers for enrolment, login verification and check-in/out */
const CameraKit = {
    csrf() { return document.querySelector('meta[name="csrf-token"]').content; },

    async start(video) {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error("Camera is blocked. Open this site with HTTPS (or http://localhost) and allow camera access.");
        }
        let stream;
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } }, audio: false
            });
        } catch (e) {
            throw new Error("Camera permission denied or no camera found. Please allow camera access and reload.");
        }
        video.srcObject = stream;
        await video.play();
        return stream;
    },

    stop(video) {
        if (video && video.srcObject) video.srcObject.getTracks().forEach(t => t.stop());
    },

    frame(video, work, maxW = 480, quality = 0.8) {
        const w = Math.min(maxW, video.videoWidth), h = Math.round(video.videoHeight * w / video.videoWidth);
        work.width = w; work.height = h;
        work.getContext("2d").drawImage(video, 0, 0, w, h);
        return work.toDataURL("image/jpeg", quality);
    },

    async post(url, body) {
        const r = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json", "X-CSRFToken": this.csrf() },
            body: JSON.stringify(body), credentials: "same-origin"
        });
        let data = {};
        try { data = await r.json(); } catch (e) {}
        data._status = r.status;
        return data;
    },

    /* draws the detected face box on the (mirrored) overlay canvas */
    box(overlay, video, res, colour) {
        overlay.width = video.clientWidth; overlay.height = video.clientHeight;
        const ctx = overlay.getContext("2d");
        ctx.clearRect(0, 0, overlay.width, overlay.height);
        if (!res || !res.box || !res.frame) return;
        // video uses object-fit: cover -> map frame coords to displayed coords
        const [fw, fh] = res.frame, cw = overlay.width, ch = overlay.height;
        const scale = Math.max(cw / fw, ch / fh);
        const ox = (cw - fw * scale) / 2, oy = (ch - fh * scale) / 2;
        const [x, y, w, h] = res.box;
        ctx.strokeStyle = colour; ctx.lineWidth = 3; ctx.shadowColor = colour; ctx.shadowBlur = 12;
        ctx.strokeRect(x * scale + ox, y * scale + oy, w * scale, h * scale);
    },

    sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
};

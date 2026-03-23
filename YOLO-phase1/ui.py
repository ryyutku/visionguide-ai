# ui.py  —  run with: python ui.py
# Requires: pip install pillow

import cv2
import tkinter as tk
import logging
import time
from PIL import Image, ImageTk

for _n in ["comtypes", "comtypes.client", "comtypes.server",
           "PIL", "ultralytics", "torch", "urllib3",
           "pyttsx3", "pyttsx3.driver", "pyttsx3.drivers"]:
    logging.getLogger(_n).setLevel(logging.CRITICAL)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-10s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ui")

from detector import DetectorTracker
from scene    import SceneAnalyzer
from guidance import GuidanceEngine
from speech   import SpeechEngine, PRIORITY_HIGH

# ── Palette ───────────────────────────────────────────────────────────────────
BG        = "#0f0f0f"
SURFACE   = "#1a1a1a"
BORDER    = "#2a2a2a"
TXT       = "#e8e8e8"
TXT_MUTED = "#555"

CLR_CLEAR    = "#22c55e"
CLR_OCCUPIED = "#f59e0b"
CLR_CROWDED  = "#ef4444"
CLR_URGENT   = "#f97316"
CLR_WARNING  = "#eab308"
CLR_CLEAR_A  = "#4ade80"
CLR_INFO     = "#444"

ZONE_COLORS     = {"clear": CLR_CLEAR, "occupied": CLR_OCCUPIED, "crowded": CLR_CROWDED}
PRIORITY_COLORS = {3: CLR_URGENT, 2: CLR_WARNING, 1: CLR_CLEAR_A}

# Banner stays visible for this many milliseconds after a message fires
BANNER_LINGER_MS = 4000


class VisionGuideUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("VisionGuide")
        root.configure(bg=BG)
        root.resizable(False, False)

        self._detector    = DetectorTracker("yolov8n.pt")
        self._scene       = SceneAnalyzer()
        self._guidance    = GuidanceEngine()
        self._speech      = SpeechEngine()

        self._cap = cv2.VideoCapture(0)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self._running       = True
        self._alert_count   = 0

        # Banner linger state
        self._banner_text   = "Initialising..."
        self._banner_color  = TXT_MUTED
        self._banner_until  = 0.0   # timestamp when banner should fade to "Monitoring"

        self._build_ui()
        self._loop()

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Top bar
        top = tk.Frame(self.root, bg=BG)
        top.pack(fill="x", padx=12, pady=(10, 4))
        tk.Label(top, text="VisionGuide", bg=BG, fg=TXT,
                 font=("Helvetica", 14, "bold")).pack(side="left")
        self._status_lbl = tk.Label(top, text="● Running", bg=BG,
                                    fg=CLR_CLEAR, font=("Helvetica", 11))
        self._status_lbl.pack(side="right")

        # Main row
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True, padx=12)

        # Camera card
        cam_card = tk.Frame(main, bg=SURFACE,
                            highlightbackground=BORDER, highlightthickness=1)
        cam_card.pack(side="left", fill="both", expand=True)

        self._cam_label = tk.Label(cam_card, bg="#000")
        self._cam_label.pack()

        # Alert banner — stays visible for BANNER_LINGER_MS after each alert
        self._banner = tk.Label(
            cam_card,
            text="Initialising...",
            bg="#111", fg=TXT_MUTED,
            font=("Helvetica", 13, "bold"),
            anchor="w", padx=12, pady=8,
        )
        self._banner.pack(fill="x")

        # Sidebar
        sidebar = tk.Frame(main, bg=BG, width=230)
        sidebar.pack(side="left", fill="y", padx=(10, 0))
        sidebar.pack_propagate(False)

        # Zone status
        zf = tk.LabelFrame(sidebar, text="Zones", bg=BG, fg=TXT_MUTED,
                           font=("Helvetica", 10),
                           highlightbackground=BORDER, bd=1)
        zf.pack(fill="x", pady=(0, 8))

        self._zone_labels = {}
        for zone in ("left", "center", "right"):
            row = tk.Frame(zf, bg=BG)
            row.pack(fill="x", padx=8, pady=4)
            tk.Label(row, text=zone.upper(), bg=BG, fg=TXT_MUTED,
                     font=("Helvetica", 10), width=8, anchor="w").pack(side="left")
            dot = tk.Label(row, text="●", bg=BG, fg=CLR_CLEAR,
                           font=("Helvetica", 10))
            dot.pack(side="left", padx=(0, 4))
            lbl = tk.Label(row, text="clear", bg=BG, fg=CLR_CLEAR,
                           font=("Helvetica", 10, "bold"))
            lbl.pack(side="left")
            self._zone_labels[zone] = (dot, lbl)

        # Metrics grid
        mf = tk.Frame(sidebar, bg=BG)
        mf.pack(fill="x", pady=(0, 8))
        mf.columnconfigure((0, 1), weight=1)

        self._metric_labels = {}
        for i, (label, val) in enumerate([
            ("Objects", "0"), ("Confirmed", "0"),
            ("Closest",  "—"), ("Alerts",    "0"),
        ]):
            card = tk.Frame(mf, bg=SURFACE,
                            highlightbackground=BORDER, highlightthickness=1)
            card.grid(row=i // 2, column=i % 2, padx=3, pady=3, sticky="nsew")
            tk.Label(card, text=label, bg=SURFACE, fg=TXT_MUTED,
                     font=("Helvetica", 9)).pack(anchor="w", padx=8, pady=(6, 0))
            v = tk.Label(card, text=val, bg=SURFACE, fg=TXT,
                         font=("Helvetica", 18, "bold"))
            v.pack(anchor="w", padx=8, pady=(0, 6))
            self._metric_labels[label] = v

        # Alert log
        lf = tk.LabelFrame(sidebar, text="Alert log", bg=BG, fg=TXT_MUTED,
                           font=("Helvetica", 10),
                           highlightbackground=BORDER, bd=1)
        lf.pack(fill="both", expand=True)

        self._log_text = tk.Text(
            lf, bg=SURFACE, fg=TXT,
            font=("Courier", 9),
            state="disabled", wrap="word",
            relief="flat", bd=0,
            highlightthickness=0,
            height=16,
        )
        self._log_text.pack(fill="both", expand=True, padx=4, pady=4)
        self._log_text.tag_config("urgent",  foreground=CLR_URGENT)
        self._log_text.tag_config("warning", foreground=CLR_WARNING)
        self._log_text.tag_config("clear",   foreground=CLR_CLEAR_A)
        self._log_text.tag_config("info",    foreground=CLR_INFO)
        self._log_text.tag_config("time",    foreground="#444")

        # Zone occupancy bars (bottom row)
        bf = tk.Frame(self.root, bg=BG)
        bf.pack(fill="x", padx=12, pady=(8, 10))
        bf.columnconfigure((0, 1, 2), weight=1)

        self._zone_bars = {}
        for i, zone in enumerate(("left", "center", "right")):
            card = tk.Frame(bf, bg=SURFACE,
                            highlightbackground=BORDER, highlightthickness=1)
            card.grid(row=0, column=i, padx=4, sticky="nsew")
            tk.Label(card, text=zone.upper(), bg=SURFACE, fg=TXT_MUTED,
                     font=("Helvetica", 9)).pack(anchor="w", padx=8, pady=(6, 2))
            bar_bg = tk.Frame(card, bg=BORDER, height=5)
            bar_bg.pack(fill="x", padx=8, pady=(0, 6))
            bar_bg.pack_propagate(False)
            bar_fill = tk.Frame(bar_bg, bg=CLR_CLEAR, height=5, width=4)
            bar_fill.place(x=0, y=0, height=5)
            self._zone_bars[zone] = (bar_bg, bar_fill)

        self._add_log("System started", tag="info")

    # ── Main Loop ─────────────────────────────────────────────────────────────

    def _loop(self):
        if not self._running:
            return

        ret, frame = self._cap.read()
        if ret:
            processed, detections = self._detector.get_detections(frame)
            scene_state            = self._scene.analyze(detections, frame.shape[1])
            message, priority           = self._guidance.decide(scene_state, self._speech)

            if message:
                log.info("[ALERT p%d] %s", priority, message)
                if priority < PRIORITY_HIGH: self._speech.say(message, priority)
                self._alert_count += 1

                tag = {3: "urgent", 2: "warning", 1: "clear"}.get(priority, "info")
                self._add_log(message, tag=tag)

                # Set banner and record when it should expire
                self._banner_text  = message
                self._banner_color = PRIORITY_COLORS.get(priority, TXT)
                self._banner_until = time.time() + BANNER_LINGER_MS / 1000

            self._update_ui(processed, scene_state, detections)

        self.root.after(33, self._loop)

    # ── UI Updates ────────────────────────────────────────────────────────────

    def _update_ui(self, frame, state, detections):
        # Camera frame
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._cam_label.configure(image=photo)
        self._cam_label.image = photo

        # Banner linger — show last message until it expires, then fade to muted
        now = time.time()
        if now < self._banner_until:
            self._banner.configure(text=self._banner_text,
                                   fg=self._banner_color)
        else:
            self._banner.configure(text="Monitoring...", fg=TXT_MUTED)

        # Zone labels + dots
        for zone, (dot, lbl) in self._zone_labels.items():
            status = state.zones[zone]
            color  = ZONE_COLORS.get(status, TXT_MUTED)
            dot.configure(fg=color)
            lbl.configure(text=status, fg=color)

        # Zone bars
        for zone, (bg_frame, fill) in self._zone_bars.items():
            status = state.zones[zone]
            count  = state.zone_counts[zone]
            color  = ZONE_COLORS.get(status, TXT_MUTED)
            fill.configure(bg=color)
            w = bg_frame.winfo_width()
            if w > 1:
                pct    = min(count / 3, 1.0)
                fill_w = max(4, int(w * pct)) if count > 0 else 4
                fill.place(x=0, y=0, width=fill_w, height=5)

        # Metrics
        confirmed = sum(1 for d in detections if d["confirmed"])
        prox      = state.closest_proximity
        prox_color = {"close": CLR_CROWDED, "medium": CLR_OCCUPIED,
                      "far": CLR_CLEAR, "none": TXT_MUTED}.get(prox, TXT)

        self._metric_labels["Objects"].configure(text=str(len(detections)))
        self._metric_labels["Confirmed"].configure(
            text=str(confirmed),
            fg=TXT if confirmed == 0 else CLR_OCCUPIED)
        self._metric_labels["Closest"].configure(
            text=prox if prox != "none" else "—", fg=prox_color)
        self._metric_labels["Alerts"].configure(text=str(self._alert_count))

    def _add_log(self, msg: str, tag: str = "info"):
        t = time.strftime("%H:%M:%S")
        self._log_text.configure(state="normal")
        self._log_text.insert("1.0", f"{msg}\n", tag)
        self._log_text.insert("1.0", f"{t}  ", "time")
        lines = int(self._log_text.index("end-1c").split(".")[0])
        if lines > 80:
            self._log_text.delete(f"{lines - 10}.0", "end")
        self._log_text.configure(state="disabled")

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def cleanup(self):
        self._running = False
        self._speech.shutdown()
        self._cap.release()


def main():
    root = tk.Tk()
    app  = VisionGuideUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.cleanup(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
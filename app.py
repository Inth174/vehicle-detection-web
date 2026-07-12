"""
==============================================================
app.py
Streamlit Web App - Vehicle Detection (YOLOv8)
--------------------------------------------------------------
Upload an image OR a video, run it through the trained YOLOv8
vehicle detection model (+ tracking/direction for video), and
display the annotated result plus a summary.

Run with:
    streamlit run app.py
==============================================================
"""

import tempfile
import time
from pathlib import Path

import cv2
import gdown
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.detect import VehicleDetector
from src.visualize import draw_vehicle, draw_dashboard

# --------------------------------------------------------------
# Page config
# --------------------------------------------------------------

st.set_page_config(
    page_title="Vehicle Detection - YOLOv8",
    page_icon="🚗",
    layout="wide",
)

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"

# Temp folder INSIDE the project (cross-platform, avoids hardcoded
# Linux-only "/tmp/" which does not exist on Windows).
TEMP_DIR = ROOT_DIR / "temp_outputs"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------
# Google Drive model config
# --------------------------------------------------------------
# Model .pt nặng hơn 100MB nên không đưa lên GitHub được -> lưu trên
# Google Drive (chế độ chia sẻ "Anyone with the link") và tự động tải
# về khi app khởi chạy nếu máy chưa có sẵn file.
#
# Cách lấy FILE ID: mở link chia sẻ Drive, nó có dạng:
#   https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrStUvWxYz/view?usp=sharing
# Phần ID chính là đoạn nằm giữa "/d/" và "/view":
#   1AbCdEfGhIjKlMnOpQrStUvWxYz
#
# Điền tên file model (đúng như tên sẽ dùng trong models/) và ID
# tương ứng vào dict bên dưới. Có thể khai báo nhiều model.
GDRIVE_MODELS = {
    "TN1_best.pt": "1_KDzlkU0Xk8VRvZXEQERWYay8OiFDvd1",
    "TN3_best.pt": "1wrQxiesfCSnJGQh-RF3CzizY5vdIADxi",
}


def ensure_model_downloaded(filename: str, file_id: str) -> Path:
    """
    Kiểm tra file model đã tồn tại trong thư mục models/ chưa.
    Nếu chưa có, dùng gdown để tải về từ Google Drive.
    """

    dest_path = MODELS_DIR / filename

    if dest_path.exists():
        return dest_path

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    url = f"https://drive.google.com/uc?id={file_id}"

    with st.spinner(f"Đang tải model '{filename}' từ Google Drive (chỉ tải 1 lần)..."):
        gdown.download(url, str(dest_path), quiet=False)

    if not dest_path.exists():
        st.error(
            f"Tải model '{filename}' thất bại. Kiểm tra lại File ID hoặc "
            "quyền chia sẻ (phải là 'Anyone with the link')."
        )
        st.stop()

    return dest_path


def sync_gdrive_models():
    """Đảm bảo mọi model khai báo trong GDRIVE_MODELS đã có sẵn cục bộ."""

    for filename, file_id in GDRIVE_MODELS.items():

        if not file_id or "Dán_Mã_ID" in file_id:
            # Chưa điền ID thật -> bỏ qua, không cố tải.
            continue

        ensure_model_downloaded(filename, file_id)


sync_gdrive_models()


# --------------------------------------------------------------
# Helpers
# --------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_detector(model_path: str, conf: float) -> VehicleDetector:
    """Load (and cache) the YOLOv8 detector for a given model + conf."""
    return VehicleDetector(model_path=model_path, conf=conf)


@st.cache_resource(show_spinner=False)
def load_predictor(model_path: str, conf: float, tracker_type: str, allowed_direction: str):
    """Load (and cache) the full Predictor (detector + tracker + direction) for video."""
    # Imported lazily so the app still works for images even if
    # deep_sort_realtime / supervision are not installed.
    from src.predict import Predictor

    return Predictor(
        model_path=model_path,
        tracker_type=tracker_type,
        conf=conf,
        allowed_direction=allowed_direction,
    )


def list_available_models():
    if not MODELS_DIR.exists():
        return []
    return sorted([p.name for p in MODELS_DIR.glob("*.pt")])


def run_image_detection(detector: VehicleDetector, image_bgr: np.ndarray):
    """Run detection and draw annotations on a copy of the image."""
    result = detector.detect_image(image_bgr)
    annotated = image_bgr.copy()

    for det in result["detections"]:
        draw_vehicle(
            annotated,
            box=det["bbox"],
            class_name=det["class_name"],
            confidence=det["confidence"],
        )

    return result, annotated


# --------------------------------------------------------------
# Sidebar - settings
# --------------------------------------------------------------

st.sidebar.title("⚙️ Cấu hình")

available_models = list_available_models()

if not available_models:
    st.sidebar.error(
        f"Không tìm thấy file model (.pt) trong thư mục '{MODELS_DIR.name}/'."
    )
    st.stop()

model_name = st.sidebar.selectbox("Chọn model", available_models, index=0)
conf_threshold = st.sidebar.slider(
    "Ngưỡng tin cậy (confidence)", min_value=0.05, max_value=0.95, value=0.25, step=0.05
)

model_path = str(MODELS_DIR / model_name)

mode = st.sidebar.radio("Loại đầu vào", ["Ảnh", "Video"])

tracker_type = "deepsort"
allowed_direction = "DOWN"

if mode == "Video":
    tracker_type = st.sidebar.selectbox(
        "Thuật toán tracking", ["deepsort", "bytetrack"], index=0
    )
    allowed_direction = st.sidebar.selectbox(
        "Hướng di chuyển hợp lệ (phát hiện đi ngược chiều)",
        ["UP", "DOWN", "LEFT", "RIGHT"],
        index=1,
    )
    frame_skip = st.sidebar.slider(
        "Xử lý 1 frame mỗi N frame (tăng tốc video dài)", 1, 5, 1
    )

st.sidebar.markdown("---")
st.sidebar.caption(
    "Model YOLOv8 dùng để phát hiện phương tiện giao thông "
    "(xe máy, ô tô, xe tải, xe buýt, ...)."
)

# --------------------------------------------------------------
# Main UI
# --------------------------------------------------------------

st.title("🚗 Vehicle Detection Demo (YOLOv8)")
st.write(
    "Tải ảnh hoặc video lên để chạy qua mô hình phát hiện phương tiện "
    "giao thông và xem kết quả nhận diện."
)

# ================================================================
# IMAGE MODE
# ================================================================

if mode == "Ảnh":

    uploaded_file = st.file_uploader(
        "Chọn một ảnh (jpg, jpeg, png)",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    if uploaded_file is not None:

        pil_image = Image.open(uploaded_file).convert("RGB")
        image_rgb = np.array(pil_image)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        with st.spinner("Đang tải model và chạy nhận diện..."):
            detector = load_detector(model_path, conf_threshold)
            start = time.time()
            result, annotated_bgr = run_image_detection(detector, image_bgr)
            elapsed = time.time() - start

        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Ảnh gốc")
            st.image(image_rgb, use_container_width=True)
        with col2:
            st.subheader("Kết quả nhận diện")
            st.image(annotated_rgb, use_container_width=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Số phương tiện phát hiện", result["vehicle_count"])
        m2.metric("Thời gian xử lý", f"{result['processing_time']:.3f}s")
        m3.metric("FPS ước tính", f"{result['fps']:.1f}")
        m4.metric("Thời gian tổng", f"{elapsed:.3f}s")

        st.subheader("📋 Chi tiết các đối tượng phát hiện")

        if result["detections"]:
            # ==========================
                    # DataFrame hiển thị
                    # ==========================
                    df = pd.DataFrame([
                        {
                            "STT": i + 1,
                            "Loại phương tiện": det["class_name"],
                            "Độ tin cậy": f"{det['confidence'] * 100:.1f}%",
                        }
                        for i, det in enumerate(result["detections"])
                    ])

                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True,
                    )

                    # ==========================
                    # Tổng số phương tiện
                    # ==========================
                    st.metric(
                        label="🚗 Tổng số phương tiện",
                        value=len(df)
                    )

                    # ==========================
                    # Thống kê theo class UA-DETRAC
                    # ==========================
                    counts = df["Loại phương tiện"].value_counts()

                    st.write("Class phát hiện được:")
                    st.write(df["Loại phương tiện"].unique())

                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(8,4))

                    ax.bar(counts.index, counts.values)

                    ax.set_ylim(0, max(counts.values) + 1)

                    ax.set_xlabel("Loại phương tiện")
                    ax.set_ylabel("Số lượng")
                    ax.set_title("Số lượng phương tiện theo loại")

                    for i, v in enumerate(counts.values):
                        ax.text(i, v + 0.05, str(v), ha="center")

                    st.pyplot(fig)
        else:
            st.info("Không phát hiện được phương tiện nào trong ảnh này.")

        out_path = TEMP_DIR / "annotated_result.jpg"
        cv2.imwrite(str(out_path), annotated_bgr)
        with open(out_path, "rb") as f:
            st.download_button(
                "⬇️ Tải ảnh kết quả",
                data=f,
                file_name="ket_qua_nhan_dien.jpg",
                mime="image/jpeg",
            )

    else:
        st.info("👆 Vui lòng tải ảnh lên để bắt đầu.")

# ================================================================
# VIDEO MODE
# ================================================================

else:

    uploaded_video = st.file_uploader(
        "Chọn một video (mp4, avi, mov, mkv)",
        type=["mp4", "avi", "mov", "mkv"],
    )

    if uploaded_video is not None:

        # Save upload to a temp file (inside the project's temp_outputs
        # folder, not the OS "/tmp" which doesn't exist on Windows) so
        # cv2.VideoCapture can read it.
        in_tmp = tempfile.NamedTemporaryFile(
            delete=False,
            dir=str(TEMP_DIR),
            suffix=Path(uploaded_video.name).suffix,
        )
        in_tmp.write(uploaded_video.read())
        in_tmp.close()

        st.video(in_tmp.name)

        run_button = st.button("▶️ Chạy nhận diện trên video")

        if run_button:

            try:
                with st.spinner("Đang tải model (detector + tracker)..."):
                    predictor = load_predictor(
                        model_path, conf_threshold, tracker_type, allowed_direction
                    )
            except ImportError as e:
                st.error(
                    "Thiếu thư viện cho tracking video (deep_sort_realtime / "
                    f"supervision). Hãy cài `pip install -r requirements.txt`.\n\n{e}"
                )
                st.stop()

            cap = cv2.VideoCapture(in_tmp.name)
            fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            out_path = TEMP_DIR / "annotated_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps_in, (width, height))

            progress_bar = st.progress(0)
            status_text = st.empty()
            preview_placeholder = st.empty()

            seen_ids = set()
            wrong_way_ids = set()
            class_counter = {}

            frame_idx = 0
            start_time = time.time()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1

                if frame_idx % frame_skip != 0:
                    writer.write(frame)
                    continue

                pred = predictor.predict_frame(frame)

                annotated = frame.copy()

                for track in pred["tracks"]:
                    draw_vehicle(
                        annotated,
                        box=track["bbox"],
                        track_id=track["track_id"],
                        class_name=track["class_name"],
                        confidence=track["confidence"],
                        direction=track["direction"],
                        wrong_way=track["wrong_way"],
                    )

                    seen_ids.add(track["track_id"])
                    if track["wrong_way"]:
                        wrong_way_ids.add(track["track_id"])
                    class_counter[track["class_name"]] = class_counter.get(track["class_name"], 0) + 1

                draw_dashboard(
                    annotated,
                    fps=pred["fps"],
                    processing_time=pred["processing_time"],
                    vehicle_count=len(seen_ids),
                    wrong_way_count=len(wrong_way_ids),
                )

                writer.write(annotated)

                if frame_idx % 5 == 0 or frame_idx == total_frames:
                    preview_placeholder.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        caption=f"Frame {frame_idx}/{total_frames}",
                        use_container_width=True,
                    )

                if total_frames > 0:
                    progress_bar.progress(min(frame_idx / total_frames, 1.0))
                status_text.text(f"Đang xử lý frame {frame_idx}/{total_frames}...")

            cap.release()
            writer.release()

            elapsed = time.time() - start_time
            status_text.text(f"Hoàn tất! Tổng thời gian xử lý: {elapsed:.1f}s")
            progress_bar.progress(1.0)

            st.subheader("🎬 Kết quả video")
            st.video(out_path)

            m1, m2, m3, m4 = st.columns(4)

            m1.metric("🚗 Tổng phương tiện", len(seen_ids))
            m2.metric("🚫 Xe đi ngược chiều", len(wrong_way_ids))
            m3.metric("⚡ FPS", f"{pred['fps']:.1f}")
            m4.metric("⏱ Thời gian", f"{elapsed:.1f}s")

            if class_counter:

                st.subheader("📊 Thống kê số lượt phát hiện")

                import matplotlib.pyplot as plt

                counts = pd.Series(class_counter)

                fig, ax = plt.subplots(figsize=(8,4))

                ax.bar(counts.index, counts.values)

                ax.set_ylim(bottom=0)

                ax.set_xlabel("Loại phương tiện")
                ax.set_ylabel("Số lượt")

                for i,v in enumerate(counts.values):
                    ax.text(i,v+0.2,str(v),ha="center")

                st.pyplot(fig)

            with open(out_path, "rb") as f:
                st.download_button(
                    "⬇️ Tải video kết quả",
                    data=f,
                    file_name="ket_qua_nhan_dien.mp4",
                    mime="video/mp4",
                )

    else:
        st.info("👆 Vui lòng tải video lên để bắt đầu.")

import app.streamlit as st
import requests
from PIL import Image

st.set_page_config(layout="wide")
st.title("🔍 Deepfake Detection")

img_col, ctrl_col = st.columns([6, 4])

with ctrl_col:
    uploaded_file = st.file_uploader(
        "Upload a JPG image", type=["jpg", "jpeg"],
        help="Maximum 200 MB per file"
    )

if uploaded_file:
    # show the image on the left
    with img_col:
        st.image(
            Image.open(uploaded_file),
            caption=uploaded_file.name,
            use_container_width=True
        )

    with ctrl_col:
        if st.button("Analyse Image"):
            uploaded_file.seek(0)
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.read(),
                    uploaded_file.type,
                )
            }

            with st.spinner("Running preprocessing and prediction…"):
                try:
                    r = requests.post(
                        "http://localhost:8000/upload-image/",
                        files=files,
                        timeout=600,
                    )
                    if r.ok:
                        data = r.json()
                        label = "Real" if data["image_is_real"] else "Deepfake"
                        conf  = data["confidence"] * 100
                        st.success(f"**{label}** ({conf:.1f}% confidence)")
                    else:
                        try:
                            err = r.json().get("error", "")
                        except ValueError:
                            err = r.text
                        if "No face detected" in err:
                            st.warning(
                                "No face detected in the image :(\n"
                                "Please upload a photo where the face is clearly visible."
                            )
                        else:
                            st.error(f"ERROR {r.status_code}: {err}")

                except Exception as e:
                    st.error(f"Unexpected ERROR: {e}")
else:
    st.info("Upload an image to begin.")

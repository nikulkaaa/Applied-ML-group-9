import streamlit as st
import requests
from PIL import Image
from pathlib import Path

st.set_page_config(layout="wide")
st.title("🔍 Deepfake Detection")

# layout
img_col, ctrl_col = st.columns([6, 4])

with ctrl_col:
    uploaded_file = st.file_uploader(
        "Upload a JPG image", type=["jpg", "jpeg"],
        help="Maximum 200 MB per file"
    )

if uploaded_file:
    # Show the uploaded image on the left for a quick visual check
    with img_col:
        st.image(
            Image.open(uploaded_file),
            caption=uploaded_file.name,
            use_container_width=True,
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

                    # Handle prediction response
                    if r.ok:
                        data = r.json()
                        label = "Real" if data.get("image_is_real") else "Deepfake"
                        conf = data.get("confidence", 0) * 100
                        st.success(f"**{label}** ({conf:.1f}% confidence)")

                        # Locate and display saliency map
                        try:
                            stem = Path(uploaded_file.name).stem
                            suffix = Path(uploaded_file.name).suffix

                            # Directory: uploads/<stem>_preprocessed/
                            saliency_dir = Path("uploads") / f"{stem}_preprocessed"
                            candidate = saliency_dir / f"{stem}_preprocessed_saliency{suffix}"

                            # Fallback: any saliency file in the dir if default not found
                            if not candidate.is_file() and saliency_dir.is_dir():
                                for p in saliency_dir.glob("*saliency*.*"):
                                    candidate = p
                                    break

                            if candidate.is_file():
                                st.image(
                                    Image.open(candidate),
                                    caption="Saliency Map",
                                    use_container_width=True,
                                )
                            else:
                                st.info(
                                    "Saliency map not found in “uploads/" \
                                    f"{stem}_preprocessed/”. Make sure the preprocessing "
                                    "step saves the map with ‘_saliency’ in its name."
                                )
                        except Exception as e:
                            st.warning(f"Could not load saliency map: {e}")

                    # Handle API errors
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

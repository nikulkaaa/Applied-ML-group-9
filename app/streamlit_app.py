import streamlit as st
import requests
import json
from PIL import Image
from pathlib import Path
import traceback

st.set_page_config(layout="wide")
st.title("🔍 Deepfake Detection")

# Layout
img_col, ctrl_col = st.columns([6, 4])

with ctrl_col:
    #  Model selector
    model_choice = st.radio(
        "Choose model version",
        ["Baseline model", "Full model"],
        index=0, # Default to Baseline model
        help=(
            "*Baseline model* is fast and lightweight.  \n"
            "*Full model* runs a 3D reconstruction pass for higher accuracy and deepfake detection."
        ),
    )

    # File upload
    uploaded_file = st.file_uploader(
        "Upload a JPG image",
        type=["jpg", "jpeg"],
        help="Maximum 200 MB per file",
    )

# Display image
if uploaded_file:
    with img_col:
        st.image(
            Image.open(uploaded_file),
            caption=uploaded_file.name,
            use_container_width=True,
        )

    # Analyse button
    with ctrl_col:
        if st.button("Analyse Image"):
            uploaded_file.seek(0) # Reset file pointer
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.read(),
                    uploaded_file.type,
                )
            }

            # Choose endpoint & timeout:
            if model_choice == "Baseline model":
                endpoint = "http://localhost:8000/upload-image/"

                progress_stages = {
                    "PREPROC_START": {"text": "Preprocessing started…", "value": 0},
                    "PREPROC_DONE": {"text": "Preprocessing done.", "value": 50},
                    "MODEL_START": {"text": "Running model prediction…", "value": 50},
                    "MODEL_DONE": {"text": "Model prediction done.", "value": 90},
                }
            else: # Full model
                endpoint = "http://localhost:8000/upload-image-full/"
                progress_stages = {
                    "PREPROC_START": {"text": "Preprocessing started…", "value": 0},
                    "PREPROC_DONE": {"text": "Preprocessing done.", "value": 25},
                    "3D_START": {"text": "Reconstructing your image in 3D…", "value": 25},
                    "3D_DONE": {"text": "3D reconstruction done.", "value": 65},
                    "MODEL_START": {"text": "Running model prediction…", "value": 65},
                    "MODEL_DONE": {"text": "Model prediction done.", "value": 90},
                }
            
            timeout_secs = 1800 # Max timeout for full model

            # Streaming Inference
            st.info(f"Requesting analysis from: {endpoint}") # For debugging
            try:
                with st.spinner("Running pipeline…"):
                    response = requests.post(
                        endpoint, files=files, timeout=timeout_secs, stream=True
                    )
                    response.raise_for_status() # Raise HTTPError for bad responses (400 or 500)

                    status_placeholder = st.empty()
                    progress_bar = st.progress(0)

                    current_progress = 0

                    final_data = None
                    error_lines = []

                    for raw_line in response.iter_lines(decode_unicode=True):
                        if not raw_line or raw_line.strip() == "":
                            continue

                        line = raw_line.strip()
                        # st.text(f"(debug) API raw: {line}") # For debugging API output

                        if line.startswith("STATUS:"):
                            code = line.split("STATUS:", 1)[1].strip()
                            if code in progress_stages:
                                status_placeholder.text(progress_stages[code]["text"])
                                current_progress = progress_stages[code]["value"]
                                progress_bar.progress(current_progress)
                            elif "ERROR" in code or "NO_FACE" in code:
                                error_lines.append(f"Pipeline status: {code}")
                            else:
                                status_placeholder.text(f"Status: {code}")
                        
                        elif line.startswith("ERROR:"):
                            error_lines.append(line.split("ERROR:", 1)[1].strip())
                        
                        else:
                            try:
                                final_data = json.loads(line)
                                current_progress = 100 # Mark as complete
                                progress_bar.progress(current_progress)
                                status_placeholder.text("Pipeline complete. Results below.")
                                break 
                            except json.JSONDecodeError:
                                error_lines.append(f"Unexpected output from API: {line}")

                # After the loop
                if error_lines:
                    st.error("Errors occurred during processing:\n" + "\n".join(error_lines))
                    st.stop()

                if not final_data:
                    st.error("Pipeline completed, but no valid result data was received.")
                    st.stop()

                # Display main prediction results (label, confidence, saliency) in ctrl_col
                with ctrl_col:
                    # Determine if real or fake
                    is_real = final_data.get("image_is_real")
                    if is_real is None:
                        is_real = (final_data.get("label") == "real")
                    
                    label_text = "Real!" if is_real else "Deepfake!"
                    confidence_value = final_data.get("confidence", 0) * 100 # Convert to percentage
                    
                    st.success(f"**Prediction: {label_text}** ({confidence_value:.1f}% confidence)")

                    saliency_path_str = final_data.get("saliency")
                    if saliency_path_str:
                        try:
                            # FastAPI serves content from UPLOAD_DIR at /uploads/
                            saliency_url = f"http://localhost:8000/uploads/{saliency_path_str}"
                            
                            st.image(
                                saliency_url, # Pass the URL directly to st.image
                                caption="Saliency Map",
                                use_container_width=True,
                            )
                        except Exception as e:
                            #  URL related errors or if image not found via URL
                            st.warning(f"Could not load saliency map from URL '{saliency_url}': {e}")
                    else:
                        if model_choice == "Full model":
                             st.info("Saliency map not generated or not available.")


            except requests.exceptions.HTTPError as e:
                st.error(f"API request failed with HTTP status {e.response.status_code}:\n{e.response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"API request failed: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred in Streamlit app: {e}")
                st.error(traceback.format_exc()) # for debugging

            # -Display 3D Reconstruction Outputs
            if model_choice == "Full model" and final_data and not error_lines:
                st.markdown("---") 
                st.subheader("🖼️ 3D Reconstruction Outputs")
                
                col1_3d, col2_3d, col3_3d = st.columns(3)

                def display_st_image(column, image_path_str, caption_text):
                    if image_path_str:
                        try:
                            # Construct the full URL to the image served by FastAPI's StaticFiles
                            # FastAPI serves content from UPLOAD_DIR at /uploads/
                            image_url = f"http://localhost:8000/uploads/{image_path_str}"
                            
                            with column:
                                # Streamlit's st.image can accept a URL
                                st.image(image_url, caption=caption_text, use_container_width=True)
                        except Exception as e:
                            with column:
                                st.warning(f"Could not load {caption_text.lower()} from URL '{image_url}': {e}")
                    else:
                        with column:
                            st.info(f"{caption_text} not available.")

                display_st_image(col1_3d, final_data.get("rendered_3d_image"), "Rendered 3D Face")
                display_st_image(col2_3d, final_data.get("depth_map_image"), "Depth Map")
                display_st_image(col3_3d, final_data.get("normals_map_image"), "Normals Map")


else: # No file uploaded
    with ctrl_col:
        st.info("Upload an image to begin analysis.")
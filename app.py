"""Launch: python -m streamlit run app.py"""
import csv
import json
from pathlib import Path

from PIL import Image, UnidentifiedImageError
import streamlit as st

from inference import load_predictor, predict_image
from train import BASE, CLASSES

st.set_page_config(page_title="SmartSort AI", page_icon="♻️", layout="wide")
st.markdown("""<style>
.stApp {background: #f5f8f5;}
.block-container {max-width: 1150px; padding-top: 2.2rem;}
h1,h2,h3 {color:#143c30;}
[data-testid="stMetric"] {background:#e7f0e9;border-radius:12px;padding:14px;}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def cached_predictor(path, model_mtime, policy_mtime, policy_filename):
    return load_predictor(path, policy_filename)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


st.caption("SMARTSORT AI  /  WASTE CLASSIFICATION")
st.title("A clearer decision for every item.")
st.write("Upload a waste photo to identify its material and see whether it needs manual review.")
st.caption("Plastic · Paper · Glass · Metal · Cardboard")

models = sorted(path for path in (BASE / "models").glob("*/best.pt")
                if path.parent.name.startswith(("baseline_", "augmented_")))
models = [path for path in models if path.with_name("uncertainty.json").exists()]
if not models:
    st.error("No calibrated model found. Complete training and run calibrate.py first.")
    st.stop()
with st.sidebar:
    st.header("SmartSort AI")
    selected_run = st.selectbox("Saved model", [path.parent.name for path in models])
    st.info("Use one primary item, centered in the photo. Mixed objects and unknown materials are outside this prototype's scope.")
    st.caption("Photos are processed by this local app and are not saved by the app.")
    st.caption("A high confidence score can still be wrong. Follow your local recycling rules.")
checkpoint = next(path for path in models if path.parent.name == selected_run)
result_dir = BASE / "results" / selected_run
full_frame_policy = checkpoint.with_name("uncertainty_full_frame.json")
active_policy = checkpoint.with_name("uncertainty.json")

classify_tab, results_tab, about_tab = st.tabs(["Classify an item", "Experiment results", "How it works"])
with classify_tab:
    if full_frame_policy.exists():
        framing = st.radio("Image framing", ["Center crop (original model input)", "Whole photo (experimental)"],
                           horizontal=True)
        if framing == "Whole photo (experimental)":
            active_policy = full_frame_policy
            st.info("Preserves the whole object. This mode has its own review cutoff and can send more photos for review. It is not an overall accuracy upgrade.")
    mode = st.radio("Choose an image source", ["Upload my photo", "Try a dataset example"], horizontal=True)
    source_image = None
    source_name = None
    try:
        if mode == "Upload my photo":
            uploaded = st.file_uploader("Choose a JPG, PNG or WebP photo", type=["jpg", "jpeg", "png", "webp"])
            if uploaded is not None:
                if uploaded.size > 10 * 1024 * 1024:
                    st.error("Please use a photo smaller than 10 MB.")
                else:
                    with Image.open(uploaded) as image:
                        if image.width * image.height > 25_000_000:
                            st.error("Please resize this photo below 25 megapixels.")
                        else:
                            image.load()
                            source_image = image.copy()
                            source_name = uploaded.name
        else:
            category = st.selectbox("Example category", CLASSES, format_func=str.title)
            with (BASE / "data" / "splits" / "test.csv").open(encoding="utf-8", newline="") as handle:
                examples = [row for row in csv.DictReader(handle) if row["label"] == category]
            example = st.selectbox("Example photo", [row["path"] for row in examples], format_func=lambda p: Path(p).name)
            with Image.open(BASE / example) as image:
                source_image = image.copy()
            source_name = Path(example).name
            st.caption("Demo from the existing test dataset; this is not a new real-world test.")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        st.error(f"Could not read this photo. Try a valid JPG or PNG. Details: {exc}")

    condition_labels = {
        "Original (no simulation)": "clean",
        "Blur — mild": "blur_mild", "Blur — medium": "blur_medium", "Blur — strong": "blur_strong",
        "Low light — mild": "dark_mild", "Low light — medium": "dark_medium", "Low light — strong": "dark_strong",
        "Partial covering — mild": "occlusion_mild", "Partial covering — medium": "occlusion_medium",
        "Partial covering — strong": "occlusion_strong"}
    chosen_condition = st.selectbox("Optional image-quality experiment", list(condition_labels))
    if source_image is None:
        st.info("Upload a photo above, or select 'Try a dataset example' to explore the demo.")
    else:
        try:
            with st.spinner("Checking the image..."):
                predictor = cached_predictor(str(checkpoint), checkpoint.stat().st_mtime_ns,
                                             active_policy.stat().st_mtime_ns, active_policy.name)
                crop, prediction = predict_image(source_image, predictor, condition_labels[chosen_condition])
            photo_col, prediction_col = st.columns([1, 1.2], gap="large")
            with photo_col:
                st.image(crop, caption="Image analyzed by the model (224 × 224)", width=336)
                st.caption("Whole photo preserved; white padding keeps the original proportions."
                           if prediction["preprocessing"] == "full_frame" else
                           "This model uses a center crop. Keep the main object near the center.")
                if prediction["condition"] != "clean":
                    st.caption("Simulated image change for this prediction only; the original file stays unchanged.")
            with prediction_col:
                st.subheader(prediction["category"].title() if prediction["decision"] == "Accepted" else "Uncertain material")
                if prediction["decision"] != "Accepted":
                    st.write(f"Tentative guess: **{prediction['category'].title()}** — not confirmed.")
                a, b = st.columns(2)
                a.metric("Calibrated confidence", f"{prediction['calibrated_confidence']:.1%}")
                b.metric("Review cutoff", f"{prediction['threshold']:.1%}")
                if prediction["decision"] == "Accepted":
                    st.success("Accepted — confidence meets the review cutoff.")
                else:
                    st.warning("Manual review needed — confidence is below the review cutoff.")
                st.caption("Confidence is an estimate of reliability, not a guarantee of correct classification.")
                with st.expander("Compare all category scores"):
                    for label, score in sorted(prediction["class_scores"].items(), key=lambda item: item[1], reverse=True):
                        st.progress(score, text=f"{label.title()}: {score:.1%}")
                st.download_button("Download prediction", json.dumps(dict(filename=source_name, **prediction), indent=2),
                                   file_name="smartsort_prediction.json", mime="application/json")
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            st.error(f"Prediction could not run: {exc}")

with results_tab:
    full_frame_report = result_dir / "uncertainty_test_full_frame" / "metrics.json"
    if active_policy == full_frame_policy:
        st.subheader("Current full-image pipeline")
        if full_frame_report.exists():
            current = read_json(full_frame_report)
            st.dataframe([{"Condition": r["condition"], "Overall accuracy": f"{r['overall_accuracy']:.2%}",
                           "Coverage": f"{r['coverage']:.1%}",
                           "Accepted accuracy": f"{r['accepted_accuracy']:.1%}" if r["accepted_accuracy"] is not None else "N/A",
                           "Wrong but accepted": r["accepted_errors"]} for r in current["conditions"]], hide_index=True)
        else:
            st.info("Full-image calibration is ready; evaluation results are pending.")
        st.subheader("Earlier center-crop experiment")
        st.caption("The historical results below use the old center-crop pipeline, not the current full-image input.")
    clean_path = result_dir / "clean_test" / "metrics.json"
    if clean_path.exists():
        clean = read_json(clean_path)
        a, b, c = st.columns(3)
        a.metric("Clean test accuracy", f"{clean['accuracy']:.2%}")
        b.metric("Macro F1", f"{clean['macro_f1']:.4f}")
        c.metric("Correct test predictions", f"{clean['correct']} / {clean['total']}")
        st.caption("Held-out TrashNet images; these scores do not establish performance on your own photos.")
    robustness_path = result_dir / "robustness" / "metrics.json"
    if robustness_path.exists():
        robust = read_json(robustness_path)
        st.subheader("When image quality changes")
        st.dataframe([{"Condition": r["condition"], "Accuracy": f"{r['accuracy']:.2%}",
                       "Macro F1": round(r["macro_f1"], 4)} for r in robust["conditions"]], hide_index=True)
        st.caption("Same 357 source images in each condition. Gray masking simulates partial occlusion, not multiple-object detection.")
    policy_path = result_dir / "uncertainty_test" / "metrics.json"
    if policy_path.exists():
        review = read_json(policy_path)
        st.subheader("What happens to uncertain predictions?")
        st.dataframe([{"Condition": r["condition"], "Accepted": r["accepted"], "Sent for review": r["review"],
                       "Accepted accuracy": f"{r['accepted_accuracy']:.1%}" if r["accepted_accuracy"] is not None else "N/A",
                       "Wrong but accepted": r["accepted_errors"]} for r in review["conditions"]], hide_index=True)
        st.caption("Accepted accuracy excludes review cases. Some mistakes still pass the cutoff.")

with about_tab:
    st.subheader("Photo → material prediction → review decision")
    st.markdown("""1. **Prepare the photo:** use the selected model's image preparation (full-image padding or the earlier center crop).
2. **Classify the material:** a fine-tuned ResNet18 predicts one of five categories.
3. **Adjust confidence:** temperature scaling uses settings learned from validation images.
4. **Apply the review rule:** predictions below the fixed cutoff need manual checking.""")
    st.write("This student prototype studies how blur, reduced brightness and partial covering affect waste classification.")
    st.write("It does not identify every object in a mixed-waste photo, detect contamination, or reliably reject all unknown materials.")
    st.markdown("Dataset: [TrashNet — Gary Thung and Mindy Yang](https://github.com/garythung/trashnet). "
                "Confidence method: [temperature scaling](https://proceedings.mlr.press/v70/guo17a.html).")

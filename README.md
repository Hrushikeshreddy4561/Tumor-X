# Brain Tumor Model — Streamlit Demo

This small Streamlit app attempts to load a PyTorch model file named `best_brain_tumor_model.pth` and run simple inference on uploaded images.

Quick start

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/Scripts/activate  
python -m pip install -r requirements.txt
```

2. Place `best_brain_tumor_model.pth` in this folder (next to `streamlit_app.py`).

3. Run the app:


streamlit run streamlit_app.py


Notes

- If the `.pth` contains only a `state_dict`, the app will show keys and cannot run inference until the model class is provided and used to rehydrate the state dict.
- The app uses default preprocessing (resize to 224x224 and ImageNet normalization). Change `preprocess` in `streamlit_app.py` if your model expects different preprocessing.

If you want help adapting the app to your model class (so the state_dict can be rehydrated), paste the model class code here or point me to the file and I can integrate it.

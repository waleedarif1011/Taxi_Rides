# Hugging Face Space deploy

The Gradio app is Space-ready. A live Space was **not** created from this agent run because no Hugging Face write token (`HF_TOKEN`) was available in the environment.

## Exact steps (Waleed)

1. Create a token at https://huggingface.co/settings/tokens with **Write** + Spaces access.
2. Locally (or in CI):

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login   # paste the token
huggingface-cli repo create taxi-rides --type space --space_sdk gradio -y
git remote add space https://huggingface.co/spaces/<YOUR_HF_USER>/taxi-rides
git push space main
```

3. Confirm the Space builds and both tabs work:
   - **KPI Explorer** — filters update ride volume / average speed / tip %
   - **Fare & Tip Calculator** — presets return positive totals (e.g. JFK → Manhattan evening)
4. Update the Live Demo badge URL in `README.md` if the Space slug differs from `waleedarif1011/taxi-rides`.

## Files the Space needs

- `app.py`, `requirements.txt`, `README.md` (YAML frontmatter)
- `src/`
- `data/nyc_taxis.csv` (or ensure the Space can load it)
- `artifacts/` (`fare_model.joblib`, `metrics.json`)
- `assets/` (KPI charts shown in the explorer)

## Local smoke check before pushing

```bash
python -m pip install -r requirements.txt
python -m src.fare_model
python scripts/generate_figures.py
python app.py
```

# Streamlit-1

A small Streamlit web app for calculating percentage change between two values.

## Why this change?

The repository previously had no runnable app entrypoint, so `streamlit run ...` could not start anything.
This update adds a working app and includes input validation so the app does not crash when the original
value is `0` (percentage change is undefined).

## Run locally

```bash
streamlit run app.py
```

## Run tests

```bash
python -m unittest discover -s tests
```

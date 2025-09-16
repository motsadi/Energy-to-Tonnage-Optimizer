# Energy‑to‑Tonnage Optimiser Demo

This repository contains a demonstration of how machine learning can be
used to improve grinding‑circuit performance in mineral processing. It
includes:

* `synthetic_data.csv` – a synthetic dataset with 500 samples of grinding
  circuit data including mill power, feed rate, % solids, sump level,
  cyclone pressure, pump speed, pulp density, throughput and specific
  energy consumption.
* `generate_synthetic_data.py` – script used to generate the dataset.
* `app.py` – a Streamlit web app that allows users to explore the
  synthetic data, train regression models, input hypothetical
  operating conditions and receive predictions, and view recommended
  operating ranges based on high performance rows.

The goal of this demo is to show how AI can support more efficient and
stable operation of grinding circuits – a major consumer of energy at
mine sites. While the data are synthetic, they are constructed with
reasonable ranges and relationships inspired by industry experience.

## Running the app

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Launch the Streamlit app:

   ```bash
   streamlit run app.py
   ```

3. Interact with the app via the browser. Use the sidebar to navigate
   between data exploration, model training, prediction and
   operating‑window analysis.

## Dependencies

See `requirements.txt` for a list of Python packages needed to run
the application.
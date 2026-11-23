# Rotation dashboard

A Streamlit app for spotting sector, theme and country rotations. It uses daily closes from Yahoo Finance (via yfinance), so you don't need an API key.

- **Themes & sectors** are measured against the S&P 500 (SPY): 11 sectors, semis, software, cyber, biotech, defense, uranium, power grid, gold miners, oil services, copper miners, bitcoin miners, US momentum, and a custom US TMT momentum basket.
- **Countries** are measured against MSCI ACWI: 24 country ETFs across developed markets, Asia EM, LatAm and EM Europe.

Each view has a rotation map, a "what changed this week" panel and a heatmap. The "How to read this" tab explains the method.

## Files

```
app.py
requirements.txt
.streamlit/config.toml   (keeps the app in light mode)
```

## Deploy (same flow as the crypto dashboard)

1. On GitHub, create a new private repo, e.g. `rotasjon-dashboard`.
2. Upload the files: **Add file → Upload files**, then drag in `app.py`, `requirements.txt` and `README.md`. Click **Commit changes**.
3. Add the config: **Add file → Create new file**, name it `.streamlit/config.toml`, paste in the contents and commit.
4. On share.streamlit.io: **Create app → Deploy a public app from GitHub → Paste GitHub URL**, then paste
   `https://github.com/alphaconosseur/rotasjon-dashboard/blob/main/app.py` and click **Deploy**.

The app stays private as long as the GitHub repo is private.

## Changing the universe

Lines are listed at the top of `app.py` in `THEMES` and `COUNTRIES` as `(ticker, name, group)`. To add or remove a line, add or delete a row. The TMT basket's stock list is `TMT_UNIVERSE`.

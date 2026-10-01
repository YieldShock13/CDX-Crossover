# CDX-Crossover

Streamlit dashboard for European CDS credit conditions using DTCC/CFTC SDR transaction data retrieved through OpenBB.

## Series
- iTraxx Europe Xover 5Y
- iTraxx Europe Main 5Y
- Xover minus Main relative value

## Update
GitHub Actions rebuilds the rolling public-data history each weekday at 21:30 UTC. The workflow can also be run manually from the Actions tab.

## Streamlit
Entry point: `app.py`

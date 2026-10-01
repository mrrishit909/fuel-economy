"""Step 1: EPA "data on cars used for testing fuel economy", model years 2012-2026.

    ./venv/bin/python download.py

One row per emissions/fuel-economy test of a test vehicle, with its test weight, rated horsepower, transmission and
measured fuel economy. Source: https://www.epa.gov/compliance-and-fuel-economy-data/data-cars-used-testing-fuel-economy
Files go to data/raw/ (not committed).
"""
import ssl
import urllib.request
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"
NEW, OLD = "https://www.epa.gov/system/files/documents/", "https://www.epa.gov/sites/default/files/"
FILES = {
    2012: OLD + "2016-07/12tstcar.csv", 2013: OLD + "2016-07/13tstcar.csv", 2014: OLD + "2016-07/14tstcar.csv",
    2015: OLD + "2016-07/15tstcar.csv", 2016: OLD + "2016-07/16tstcar.csv", 2017: OLD + "2018-06/17tstcar-2018-05-30.xlsx",
    2018: OLD + "2018-10/18tstcar-2018-10-24.xlsx", 2019: OLD + "2020-10/19tstcar-2020-10-02.xlsx",
    2020: OLD + "2021-03/20tstcar-2021-03-02.xlsx", 2021: NEW + "2022-04/21-tstcar-2022-04-15.xlsx",
    2022: NEW + "2023-06/22-testcar-2023-06-13.xlsx", 2023: NEW + "2024-05/23-testcar-2024-05-17_0.xlsx",
    2024: NEW + "2025-05/24-testcar-2025-05.xlsx", 2025: NEW + "2026-01/25-testcar-2026-01-21.xlsx",
    2026: NEW + "2026-01/26-testcar-2026-01-21.xlsx",
}

if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    ctx = ssl.create_default_context(cafile="/etc/ssl/cert.pem")
    for year, url in FILES.items():
        out = RAW / f"{year}{Path(url).suffix}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        out.write_bytes(urllib.request.urlopen(req, context=ctx, timeout=120).read())
        print(year, out.stat().st_size // 1024, "KB")

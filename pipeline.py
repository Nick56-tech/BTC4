import requests
import pandas as pd
import datetime as dt
import time

BASE = "https://api.bybit.com"
SYMBOL = "BTCUSDT"
CATEGORY = "linear"
INTERVAL = "5"

START = "2021-01-01 00:00"
END = "2025-07-02 00:00"
WINDOW_HOURS = 48
OUTPUT_CSV = "btc_bybit_5m_filled.csv"


def ts_ms(iso: str) -> int:
    t = dt.datetime.fromisoformat(iso).replace(tzinfo=dt.timezone.utc)
    return int(t.timestamp() * 1000)


def fetch(endpoint: str, params: dict, path: str = "list"):
    url, rows, cursor = f"{BASE}{endpoint}", [], None
    while True:
        if cursor:
            params["cursor"] = cursor
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code} – {url}")
        j = r.json()
        if j["retCode"] != 0:
            raise RuntimeError(f"retCode {j['retCode']} – {j['retMsg']}")
        batch = j["result"][path]
        rows.extend(batch)
        cursor = j["result"].get("nextPageCursor")
        if not cursor or not batch:
            break
        time.sleep(0.05)
    return rows


def date_range_windows(start_iso: str, end_iso: str, hours: int = 48):
    start = dt.datetime.fromisoformat(start_iso)
    end = dt.datetime.fromisoformat(end_iso)
    while start < end:
        window_end = min(start + dt.timedelta(hours=hours), end)
        yield start.strftime("%Y-%m-%d %H:%M"), window_end.strftime("%Y-%m-%d %H:%M")
        start = window_end


def main():
    oi_all, kl_all, fd_all = [], [], []

    for win_start, win_end in date_range_windows(START, END, WINDOW_HOURS):
        start_ms, end_ms = ts_ms(win_start), ts_ms(win_end)

        oi_all.extend(
            fetch(
                "/v5/market/open-interest",
                dict(
                    category=CATEGORY,
                    symbol=SYMBOL,
                    intervalTime="5min",
                    startTime=start_ms,
                    endTime=end_ms,
                    limit=200,
                ),
                "list",
            )
        )

        kl_all.extend(
            fetch(
                "/v5/market/kline",
                dict(
                    category=CATEGORY,
                    symbol=SYMBOL,
                    interval=INTERVAL,
                    start=start_ms,
                    end=end_ms,
                    limit=1000,
                ),
                "list",
            )
        )

        fd_all.extend(
            fetch(
                "/v5/market/funding/history",
                dict(
                    category=CATEGORY,
                    symbol=SYMBOL,
                    startTime=start_ms,
                    endTime=end_ms,
                    limit=200,
                ),
                "list",
            )
        )

    df_oi = (
        pd.DataFrame(oi_all)
        .assign(
            timestamp=lambda d: pd.to_datetime(d["timestamp"].astype(int), unit="ms", utc=True),
            openInterest=lambda d: d["openInterest"].astype(float),
        )
        .set_index("timestamp")
    )

    cols_kline = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]
    df_kl = (
        pd.DataFrame(kl_all, columns=cols_kline)
        .assign(timestamp=lambda d: pd.to_datetime(d["timestamp"].astype(int), unit="ms", utc=True))
        .set_index("timestamp")
        .astype(float)
    )

    df_fd = (
        pd.DataFrame(fd_all)
        .assign(
            timestamp=lambda d: pd.to_datetime(d["fundingRateTimestamp"].astype(int), unit="ms", utc=True),
            fundingRate=lambda d: d["fundingRate"].astype(float),
        )
        .set_index("timestamp")
    )

    df = df_oi.join(df_kl, how="outer").join(df_fd, how="outer").sort_index()

    cols = ["symbol", "fundingRate", "fundingRateTimestamp"]
    for c in cols:
        if c in df.columns:
            df[c] = df[c].ffill()

    df.to_csv(OUTPUT_CSV)
    print(f"✅ Bybit data downloaded → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

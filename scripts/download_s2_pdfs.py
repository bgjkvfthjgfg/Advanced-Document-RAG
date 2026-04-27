import os
import re
import time
import requests
import pandas as pd
from tqdm import tqdm

SAVE_DIR = "data/raw_pdfs_bulk"
META_PATH = "data/raw_pdfs_bulk_metadata.csv"
FAILED_PATH = "data/raw_pdfs_bulk_failed.csv"

os.makedirs(SAVE_DIR, exist_ok=True)

BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"

# 扩宽后的查询词：仍然聚焦自动驾驶 / 车辆轨迹预测 / 交互预测
QUERIES = [
    '"trajectory prediction" "autonomous driving"',
    '"motion forecasting" "autonomous driving"',
    '"interactive prediction" "autonomous driving"',
    '"vehicle trajectory prediction"',
    '"multimodal trajectory prediction"',
    '"behavior prediction" "autonomous driving"',
    '"trajectory forecasting" "autonomous driving"',
    '"interactive motion prediction" "autonomous driving"',
    '"future trajectory prediction" vehicle',
    '"vehicle behavior prediction"',
    '"intention prediction" "autonomous driving"',
    '"multi-agent prediction" "autonomous driving"',
    '"traffic participant prediction"',
    '"scene prediction" "autonomous driving"',
    '"motion prediction" vehicle',
    '"trajectory forecasting" vehicle',
    '"interactive motion forecasting"',
    '"trajectory prediction" vehicle',
    '"autonomous driving" "trajectory forecasting"',
    '"autonomous driving" "behavior prediction"',
]

FIELDS = ",".join([
    "title",
    "year",
    "venue",
    "authors",
    "url",
    "openAccessPdf",
    "paperId",
    "citationCount"
])

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    # 如果你后面申请了 Semantic Scholar API Key，可以取消注释：
    # "x-api-key": "YOUR_API_KEY"
}

# 候选先多抓一点，给去重和失败留余量
TARGET_COUNT = 1800


def safe_filename(text: str, max_len: int = 180) -> str:
    text = re.sub(r'[\\/:*?"<>|]', "_", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len]


def normalize_title(title: str) -> str:
    title = title.lower().strip()
    title = re.sub(r"\s+", " ", title)
    return title


def get_with_retry(url, params=None, headers=None, timeout=60, max_retry=5, stream=False):
    last_err = None
    for attempt in range(max_retry):
        try:
            resp = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout,
                stream=stream,
                allow_redirects=True
            )
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_err = e
            wait_time = min(2 ** attempt, 8)
            print(f"[WARN] request failed (attempt {attempt + 1}/{max_retry}): {e}")
            time.sleep(wait_time)
    raise last_err


def collect_metadata(target_count=1800):
    all_records = []
    seen_titles = set()

    for query in QUERIES:
        token = None
        while True:
            params = {
                "query": query,
                "fields": FIELDS,
                "year": "2018-2026",
                "sort": "citationCount:desc"
            }
            if token:
                params["token"] = token

            resp = get_with_retry(
                BASE_URL,
                params=params,
                headers=HEADERS,
                timeout=60,
                max_retry=5,
                stream=False
            )
            data = resp.json()

            batch = data.get("data", [])
            if not batch:
                print(f"[INFO] query empty or finished: {query}")
                break

            for item in batch:
                title = (item.get("title") or "").strip()
                if not title:
                    continue

                title_key = normalize_title(title)
                if title_key in seen_titles:
                    continue

                oa = item.get("openAccessPdf") or {}
                pdf_url = oa.get("url")

                if not pdf_url:
                    continue

                seen_titles.add(title_key)
                all_records.append({
                    "paperId": item.get("paperId"),
                    "title": title,
                    "year": item.get("year"),
                    "venue": item.get("venue"),
                    "citationCount": item.get("citationCount"),
                    "url": item.get("url"),
                    "pdf_url": pdf_url,
                    "authors": "; ".join([a.get("name", "") for a in (item.get("authors") or [])]),
                    "query_source": query,
                })

                if len(all_records) >= target_count:
                    print(f"[INFO] reached target_count={target_count}")
                    return all_records

            print(f"[INFO] current query: {query}")
            print(f"[INFO] collected so far: {len(all_records)}")

            token = data.get("token")
            if not token:
                break

            time.sleep(0.5)

    return all_records


def download_pdfs(records):
    success_rows = []
    failed_rows = []

    for rec in tqdm(records, desc="Downloading PDFs"):
        title = rec["title"]
        pdf_url = rec["pdf_url"]
        filename = safe_filename(title) + ".pdf"
        filepath = os.path.join(SAVE_DIR, filename)

        if os.path.exists(filepath):
            rec["local_path"] = filepath
            success_rows.append(rec)
            continue

        try:
            resp = get_with_retry(
                pdf_url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=30,
                max_retry=1,
                stream=True
            )

            final_url = resp.url.lower()
            content_type = resp.headers.get("Content-Type", "").lower()

            if "pdf" not in content_type and not final_url.endswith(".pdf"):
                raise ValueError(
                    f"Not a direct PDF: content_type={content_type}, final_url={resp.url}"
                )

            with open(filepath, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            # 太小一般不是有效 PDF
            if os.path.getsize(filepath) < 10 * 1024:
                raise ValueError("Downloaded file too small, may be invalid PDF")

            rec["local_path"] = filepath
            success_rows.append(rec)

        except Exception as e:
            failed_rows.append({
                "title": title,
                "pdf_url": pdf_url,
                "error": str(e)
            })

            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception:
                    pass

        time.sleep(0.2)

    return success_rows, failed_rows


if __name__ == "__main__":
    print("[INFO] collecting metadata from Semantic Scholar...")
    records = collect_metadata(target_count=TARGET_COUNT)
    print(f"[INFO] collected records with open PDFs: {len(records)}")

    print("[INFO] start downloading PDFs...")
    success_rows, failed_rows = download_pdfs(records)

    pd.DataFrame(success_rows).to_csv(META_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(failed_rows).to_csv(FAILED_PATH, index=False, encoding="utf-8-sig")

    print(f"[OK] downloaded: {len(success_rows)}")
    print(f"[OK] failed: {len(failed_rows)}")
    print(f"[OK] saved metadata to: {META_PATH}")
    print(f"[OK] saved failed logs to: {FAILED_PATH}")
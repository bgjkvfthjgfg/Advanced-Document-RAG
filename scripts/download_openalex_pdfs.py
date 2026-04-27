import os
import re
import time
import requests
import pandas as pd
from tqdm import tqdm

SAVE_DIR = "data/raw_pdfs_openalex"
META_PATH = "data/raw_pdfs_openalex_metadata.csv"
FAILED_PATH = "data/raw_pdfs_openalex_failed.csv"

os.makedirs(SAVE_DIR, exist_ok=True)

BASE_URL = "https://api.openalex.org/works"

QUERIES = [
    "trajectory prediction autonomous driving",
    "motion forecasting autonomous driving",
    "interactive prediction autonomous driving",
    "vehicle trajectory prediction",
    "multimodal trajectory prediction",
    "behavior prediction autonomous driving",
    "trajectory forecasting autonomous driving",
    "interactive motion prediction autonomous driving",
    "future trajectory prediction vehicle",
    "vehicle behavior prediction",
    "intention prediction autonomous driving",
    "multi-agent prediction autonomous driving",
    "traffic participant prediction",
    "scene prediction autonomous driving",
    "motion prediction vehicle",
    "trajectory prediction vehicle",
    "interactive motion forecasting",
    "behavior prediction vehicle",
]

TARGET_COUNT = 2200  # 候选再拉大一点，给失败和去重留余量

def safe_filename(text: str, max_len: int = 180) -> str:
    text = re.sub(r'[\\/:*?"<>|]', "_", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len]

def normalize_title(title: str) -> str:
    title = title.lower().strip()
    title = re.sub(r"\s+", " ", title)
    return title

def get_with_retry(url, params=None, timeout=60, max_retry=5, stream=False):
    last_err = None
    for attempt in range(max_retry):
        try:
            resp = requests.get(
                url,
                params=params,
                timeout=timeout,
                stream=stream,
                allow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_err = e
            wait_time = min(2 ** attempt, 8)
            print(f"[WARN] request failed (attempt {attempt + 1}/{max_retry}): {e}")
            time.sleep(wait_time)
    raise last_err

def extract_pdf_candidates(item):
    urls = []

    best_oa = item.get("best_oa_location") or {}
    best_pdf = best_oa.get("pdf_url")
    if best_pdf:
        urls.append(best_pdf)

    for loc in item.get("locations") or []:
        pdf_url = (loc or {}).get("pdf_url")
        if pdf_url and pdf_url not in urls:
            urls.append(pdf_url)

    return urls

def collect_metadata(target_count=2200):
    all_records = []
    seen_titles = set()

    for query in QUERIES:
        cursor = "*"

        while True:
            params = {
                "search": query,
                "filter": "from_publication_date:2018-01-01,is_oa:true",
                "per-page": 200,
                "cursor": cursor,
                "mailto": "example@example.com"
            }

            resp = get_with_retry(
                BASE_URL,
                params=params,
                timeout=60,
                max_retry=5,
                stream=False,
            )
            data = resp.json()

            results = data.get("results", [])
            if not results:
                print(f"[INFO] query empty or finished: {query}")
                break

            for item in results:
                title = (item.get("title") or "").strip()
                if not title:
                    continue

                title_key = normalize_title(title)
                if title_key in seen_titles:
                    continue

                pdf_candidates = extract_pdf_candidates(item)
                if not pdf_candidates:
                    continue

                authorships = item.get("authorships") or []
                authors = []
                for a in authorships:
                    author_obj = a.get("author") or {}
                    name = author_obj.get("display_name")
                    if name:
                        authors.append(name)

                primary_location = item.get("primary_location") or {}
                source_obj = primary_location.get("source") or {}
                venue_name = source_obj.get("display_name")

                seen_titles.add(title_key)
                all_records.append({
                    "id": item.get("id"),
                    "title": title,
                    "year": item.get("publication_year"),
                    "venue": venue_name,
                    "cited_by_count": item.get("cited_by_count"),
                    "pdf_candidates": " ||| ".join(pdf_candidates),
                    "authors": "; ".join(authors),
                    "query_source": query,
                })

                if len(all_records) >= target_count:
                    print(f"[INFO] reached target_count={target_count}")
                    return all_records

            print(f"[INFO] current query: {query}")
            print(f"[INFO] collected so far: {len(all_records)}")

            meta = data.get("meta") or {}
            next_cursor = meta.get("next_cursor")
            if not next_cursor:
                break

            cursor = next_cursor
            time.sleep(0.5)

    return all_records

def try_download_one(pdf_urls, filepath):
    last_error = None

    for pdf_url in pdf_urls:
        try:
            resp = get_with_retry(
                pdf_url,
                timeout=30,
                max_retry=1,
                stream=True,
            )

            final_url = resp.url.lower()
            content_type = resp.headers.get("Content-Type", "").lower()

            if "pdf" not in content_type and not final_url.endswith(".pdf"):
                raise ValueError(f"Not a direct PDF: content_type={content_type}, final_url={resp.url}")

            with open(filepath, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            if os.path.getsize(filepath) < 10 * 1024:
                raise ValueError("Downloaded file too small, may be invalid PDF")

            return pdf_url, None

        except Exception as e:
            last_error = e
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception:
                    pass

    return None, last_error

def download_pdfs(records):
    success_rows = []
    failed_rows = []

    for rec in tqdm(records, desc="Downloading OpenAlex PDFs"):
        title = rec["title"]
        filename = safe_filename(title) + ".pdf"
        filepath = os.path.join(SAVE_DIR, filename)

        if os.path.exists(filepath):
            rec["local_path"] = filepath
            success_rows.append(rec)
            continue

        pdf_urls = [u.strip() for u in str(rec["pdf_candidates"]).split("|||") if u.strip()]

        used_url, err = try_download_one(pdf_urls, filepath)

        if used_url:
            rec["local_path"] = filepath
            rec["used_pdf_url"] = used_url
            success_rows.append(rec)
        else:
            failed_rows.append({
                "title": title,
                "pdf_candidates": " ||| ".join(pdf_urls),
                "error": str(err),
            })

        time.sleep(0.2)

    return success_rows, failed_rows

if __name__ == "__main__":
    print("[INFO] collecting metadata from OpenAlex...")
    records = collect_metadata(target_count=TARGET_COUNT)
    print(f"[INFO] collected candidate records: {len(records)}")

    print("[INFO] start downloading PDFs...")
    success_rows, failed_rows = download_pdfs(records)

    pd.DataFrame(success_rows).to_csv(META_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(failed_rows).to_csv(FAILED_PATH, index=False, encoding="utf-8-sig")

    print(f"[OK] downloaded: {len(success_rows)}")
    print(f"[OK] failed: {len(failed_rows)}")
    print(f"[OK] saved metadata to: {META_PATH}")
    print(f"[OK] saved failed logs to: {FAILED_PATH}")
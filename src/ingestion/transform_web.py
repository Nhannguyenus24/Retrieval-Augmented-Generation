#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
import re
import io
import os
import hashlib
from datetime import datetime
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup, Tag
from readability import Document
from pdfminer.high_level import extract_text as pdf_extract_text

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "MainTextScraper/1.3 (+contact: you@example.com)"
})
TIMEOUT = 30
SEP = "=" * 80

# ---------------- Utils ----------------

def is_pdf_url(url: str) -> bool:
    return url.lower().endswith(".pdf")

def fetch(url: str):
    resp = SESSION.get(url, timeout=TIMEOUT, allow_redirects=True)
    resp.raise_for_status()
    ctype = resp.headers.get("Content-Type", "")
    return resp.content, ctype

def content_type_is_pdf(ctype: str) -> bool:
    return "application/pdf" in (ctype or "").lower()

def clean_whitespace(text: str) -> str:
    # Chuẩn hoá xuống dòng và gom dòng trống dài
    text = re.sub(r'\r\n?', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Gọn khoảng trắng trong từng dòng
    lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in text.split('\n')]
    text = '\n'.join(lines).strip()
    # Loại các dòng chú thích rời rạc "Archived from..." / "Retrieved ..."
    text = '\n'.join(
        l for l in text.split('\n')
        if not re.match(r'^(Archived from the original|Retrieved)\b', l)
    )
    return text

def fix_linebreaks(text: str) -> str:
    """
    - Ghép các newline đơn thành khoảng trắng (giữ \n\n là ngắt đoạn thật).
    - Sửa từ bị gạch nối ở cuối dòng: "divi-\nsion" -> "division".
    - Dọn khoảng trắng trước dấu câu.
    """
    # Sửa gạch nối cuối dòng (thực hiện trước khi ghép newline)
    text = re.sub(r'(\w)-\n(?=\w)', r'\1', text)

    # Ghép newline đơn thành space, giữ nguyên \n\n
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # Gom nhiều space
    text = re.sub(r'[ \t]{2,}', ' ', text)

    # Xoá space trước dấu câu .,;:!?)
    text = re.sub(r'\s+([,.;:!?)])', r'\1', text)
    # Xoá space sau mở ngoặc
    text = re.sub(r'([\[(])\s+', r'\1', text)

    return text.strip()

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def safe_folder_from_url(url: str) -> str:
    """
    Tạo tên folder ổn định từ URL: host + path (thay non-alnum bằng '_', cắt bớt chuỗi dài)
    Ví dụ: https://www.python.org/about/quotes/ ->
        www_python_org_about_quotes
    """
    p = urlparse(url)
    raw = (p.netloc + p.path).strip("/")
    if not raw:
        raw = p.netloc or "site"
    safe = re.sub(r'[^A-Za-z0-9]+', '_', raw)
    safe = re.sub(r'_+', '_', safe).strip('_')
    return safe[:150] if len(safe) > 150 else safe

def infer_ext_from_ctype(ctype: str) -> str:
    if not ctype:
        return ""
    ctype = ctype.lower().split(";")[0].strip()
    mapping = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
        "image/svg+xml": ".svg",
        "image/bmp": ".bmp",
        "image/tiff": ".tiff",
        "image/x-icon": ".ico",
        "image/vnd.microsoft.icon": ".ico",
    }
    return mapping.get(ctype, "")

def unique_filename(dirpath: str, base: str, ext: str) -> str:
    """
    Tránh trùng tên: nếu đã tồn tại, thêm số đếm; nếu base quá chung chung, kèm hash ngắn.
    """
    base = base or "image"
    base = re.sub(r'[^A-Za-z0-9_-]+', '_', base).strip('_')
    if not base:
        base = "image"
    ext = ext or ""
    cand = os.path.join(dirpath, base + ext)
    if not os.path.exists(cand):
        return cand
    # thêm số đếm
    i = 2
    while True:
        cand = os.path.join(dirpath, f"{base}_{i}{ext}")
        if not os.path.exists(cand):
            return cand
        i += 1

def pick_src_from_img(img: Tag, base_url: str) -> str | None:
    """
    Lấy URL ảnh tốt nhất: ưu tiên srcset (w lớn nhất), sau đó data-src, cuối cùng src.
    """
    # srcset dạng: "url1 320w, url2 640w, url3 1200w"
    srcset = img.get("srcset") or img.get("data-srcset")
    if srcset:
        candidates = []
        for part in srcset.split(","):
            part = part.strip()
            if not part:
                continue
            toks = part.split()
            url = toks[0]
            w = 0
            if len(toks) > 1 and toks[1].endswith("w"):
                try:
                    w = int(toks[1][:-1])
                except Exception:
                    w = 0
            candidates.append((w, urljoin(base_url, url)))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[-1][1]  # lớn nhất
    # data-src
    data_src = img.get("data-src") or img.get("data-original") or img.get("data-lazy-src")
    if data_src:
        return urljoin(base_url, data_src)
    # src
    src = img.get("src")
    if src:
        return urljoin(base_url, src)
    return None

def download_image(img_url: str, save_dir: str) -> str | None:
    try:
        r = SESSION.get(img_url, timeout=TIMEOUT, stream=True)
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
        # Lấy tên file từ URL
        path = urlparse(img_url).path
        base = os.path.basename(path).split("?")[0].split("#")[0]
        base_noext, _, ext_from_url = base.partition(".")
        ext_from_url = "." + ext_from_url if ext_from_url else ""

        # Nếu URL không có đuôi, suy đoán từ Content-Type
        ext = ext_from_url.lower() or infer_ext_from_ctype(ctype) or ".bin"

        # Nếu tên gốc rỗng, đặt base theo hash của URL
        if not base_noext:
            base_noext = hashlib.sha1(img_url.encode("utf-8")).hexdigest()[:12]

        # Tạo tên file duy nhất
        save_path = unique_filename(save_dir, base_noext, ext)

        # Giới hạn kích thước tải (ví dụ 25MB) để tránh file quá lớn
        max_bytes = 25 * 1024 * 1024
        written = 0
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    written += len(chunk)
                    if written > max_bytes:
                        # Xoá file dở nếu quá lớn
                        f.close()
                        try:
                            os.remove(save_path)
                        except Exception:
                            pass
                        return None
                    f.write(chunk)
        return save_path
    except Exception:
        return None

# ---------------- Heuristics to DROP noise ----------------
DROP_SELECTORS = [
    "header", "footer", "nav", "aside", "form", "iframe", "noscript",
    "script", "style", "svg", "canvas", "button", "input",
    "[role='navigation']", "[aria-role='navigation']",
    ".advertisement", ".ads", ".ad", ".promo", ".breadcrumb", ".breadcrumbs",
    ".cookie", ".newsletter", ".toc", "#toc", ".mw-jump-link", ".mw-editsection",
]

REF_SECTION_TITLES = re.compile(
    r'^(references?|reference|sources?|notes?|bibliography|external links?|footnotes?)$',
    re.IGNORECASE
)

CITATION_SELECTORS = [
    "sup.reference", "a[role='doc-noteref']", "a[rel='footnote']",
    ".reference", ".footnote", ".cite_ref", ".citation"
]

BREADCRUMB_SELECTORS = [".breadcrumb", ".breadcrumbs", "nav.breadcrumbs", "nav.breadcrumb"]

def strip_inline_citations(text: str) -> str:
    # Xoá [1], [12], dấu ^, "(link)" lẻ
    text = re.sub(r'\s*\[\d+\]', '', text)
    text = re.sub(r'\s*\^\s*', ' ', text)
    text = re.sub(r'\s*\(link\)\s*', ' ', text, flags=re.IGNORECASE)
    return re.sub(r'\s{2,}', ' ', text)

# ---------------- HTML extraction ----------------
def readability_main_text(html: str) -> str:
    doc = Document(html)
    summary_html = doc.summary(html_partial=True)
    soup = BeautifulSoup(summary_html, "lxml")

    # Loại bỏ noise
    for sel in DROP_SELECTORS + CITATION_SELECTORS + BREADCRUMB_SELECTORS:
        for node in soup.select(sel):
            node.decompose()

    _drop_reference_sections_inplace(soup)

    text = soup.get_text("\n", strip=True)
    text = clean_whitespace(text)
    text = strip_inline_citations(text)
    return text

def fallback_main_text(html: str, base_url: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    for sel in DROP_SELECTORS + BREADCRUMB_SELECTORS:
        for node in soup.select(sel):
            node.decompose()

    candidates = [
        soup.find(id="mw-content-text"),
        soup.find("main"),
        soup.find("article"),
        soup.find("div", {"class": re.compile(r"(post-content|article-body|content|entry-content|main-content)")}),
        soup.body
    ]
    container = next((c for c in candidates if c), soup)

    for sel in CITATION_SELECTORS:
        for node in container.select(sel):
            node.decompose()

    _drop_reference_sections_inplace(container)

    # Chỉ lấy text (bỏ code/table/media nếu không cần)
    for tagname in ("table", "figure", "figcaption", "code", "pre", "video", "audio"):
        for t in container.find_all(tagname):
            t.decompose()

    text = container.get_text("\n", strip=True)
    text = clean_whitespace(text)
    text = strip_inline_citations(text)
    return text

def _drop_reference_sections_inplace(root: Tag):
    for h in root.find_all(re.compile(r'^h[1-6]$')):
        title = h.get_text(" ", strip=True)
        if REF_SECTION_TITLES.match(title or ""):
            nxt = h
            while nxt:
                rm = nxt
                nxt = nxt.find_next_sibling()
                rm.decompose()

def extract_main_text_from_html(html_bytes: bytes, base_url: str) -> str:
    html = html_bytes.decode("utf-8", errors="ignore")
    try:
        txt = readability_main_text(html)
        if len(txt) >= 200:
            return txt
    except Exception:
        pass
    return fallback_main_text(html, base_url)

# --------- HTML: download images from main content ---------
def get_main_container_for_images(html_bytes: bytes, base_url: str) -> Tag | None:
    """
    Tìm container nội dung chính nhưng KHÔNG xoá <img>, để còn link ảnh để tải.
    Vẫn bỏ header/footer/nav/ads để không dính ảnh logo/quảng cáo.
    """
    html = html_bytes.decode("utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")

    for sel in DROP_SELECTORS + BREADCRUMB_SELECTORS:
        for node in soup.select(sel):
            node.decompose()

    candidates = [
        soup.find(id="mw-content-text"),
        soup.find("main"),
        soup.find("article"),
        soup.find("div", {"class": re.compile(r"(post-content|article-body|content|entry-content|main-content)")}),
        soup.body
    ]
    return next((c for c in candidates if c), soup)

def download_images_from_html(html_bytes: bytes, base_url: str, out_base_dir: str = "../../images") -> list[str]:
    """
    Tải toàn bộ ảnh trong nội dung chính vào ../../images/<safe-folder>/
    Trả về list đường dẫn file đã lưu.
    """
    container = get_main_container_for_images(html_bytes, base_url)
    if container is None:
        return []

    # Tạo thư mục đích
    folder_name = safe_folder_from_url(base_url)
    save_dir = os.path.join(out_base_dir, folder_name)
    ensure_dir(save_dir)

    saved = []
    imgs = container.find_all("img")
    seen = set()
    for img in imgs:
        url = pick_src_from_img(img, base_url)
        if not url:
            continue
        if url in seen:
            continue
        seen.add(url)
        saved_path = download_image(url, save_dir)
        if saved_path:
            saved.append(saved_path)
    return saved

# ---------------- PDF extraction ----------------
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        buf = io.BytesIO(pdf_bytes)
        txt = pdf_extract_text(buf) or ""
        txt = clean_whitespace(txt)
        # Cắt phần References trong PDF (nếu có)
        parts = re.split(r'\n(?:References|Bibliography|Notes|Footnotes)\n',
                         txt, maxsplit=1, flags=re.IGNORECASE)
        return parts[0].strip()
    except Exception as e:
        return f"(PDF extract error: {e})"

# ---------------- Output ----------------
def guess_title_from_html(html_bytes: bytes) -> str:
    try:
        soup = BeautifulSoup(html_bytes, "lxml")
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)
    except Exception:
        pass
    return "(untitled)"

def print_block(title: str, source: str, text: str, images_info: str | None = None):
    print(SEP)
    print(f"TITLE: {title}")
    print(f"SOURCE: {source}")
    print(f"EXTRACTED_AT: {datetime.utcnow().isoformat()}Z")
    if images_info:
        print(f"IMAGES: {images_info}")
    print(SEP)
    print(text if text.strip() else "(no main text found)")
    print("\n")

# ---------------- Main flow ----------------
def process_url(url: str):
    u = url.strip()
    try:
        content, ctype = fetch(u)
    except Exception as e:
        print(SEP, file=sys.stderr)
        print(f"[ERROR] Fetch failed: {u} -> {e}", file=sys.stderr)
        return

    # PDF: hiện chưa trích ảnh trong PDF (cần lib khác), chỉ tải text
    if content_type_is_pdf(ctype) or is_pdf_url(u):
        text = extract_text_from_pdf(content)
        text = fix_linebreaks(text)
        title = u.rsplit("/", 1)[-1] or "PDF"
        print_block(title, u, text, images_info="(images not extracted from PDF)")
        return

    # HTML: tải text + tải ảnh
    title = guess_title_from_html(content)
    text = extract_main_text_from_html(content, u)
    text = fix_linebreaks(text)

    # download images from main content
    saved_paths = download_images_from_html(content, u, out_base_dir="../../images")
    images_info = None
    if saved_paths:
        folder_name = safe_folder_from_url(u)
        images_dir = os.path.join("../../images", folder_name)
        images_info = f"{len(saved_paths)} saved to {images_dir}"
    else:
        images_info = "0 images"

    host = urlparse(u).netloc.lower()
    if ("jstor.org" in host) and (len(text) < 400):
        text += "\n\n(NOTE: JSTOR có thể yêu cầu đăng nhập/cookie; nội dung chính có thể bị ẩn.)"

    print_block(title, u, text, images_info=images_info)

def main():
    ap = argparse.ArgumentParser(description="Scrape ONLY main text from given URLs; download main images to ../../images/<url-folder>.")
    ap.add_argument("urls", nargs="+", help="One or more URLs to scrape")
    args = ap.parse_args()
    for url in args.urls:
        process_url(url)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, re
from urllib.parse import urlparse, urljoin
import requests
from bs4 import BeautifulSoup, Tag, NavigableString

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "TechDocsStaticScraper/1.1"})
TIMEOUT = 25

STATIC_PREFIXES = [
    "/api/techdocs/static/docs/{suffix}index.html",
    "/techdocs/static/docs/{suffix}index.html",
    "/static/docs/{suffix}index.html",
    "/docs/{suffix}index.html",
]

def normalize_ws(s: str) -> str:
    s = re.sub(r"\r\n?", "\n", s)
    s = re.sub(r"(\w)-\n(?=\w)", r"\1", s)
    s = re.sub(r"(?<!\n)\n(?!\n)", " ", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"([\[(])\s+", r"\1", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def build_suffix(docs_url: str) -> tuple[str, str]:
    p = urlparse(docs_url)
    parts = [x for x in p.path.split("/") if x]
    if "docs" not in parts:
        raise SystemExit("URL không chứa segment 'docs': " + docs_url)
    i = parts.index("docs")
    suffix = "/".join(parts[i+1:])
    if not suffix.endswith("/"):
        suffix += "/"
    base = f"{p.scheme}://{p.netloc}"
    return base, suffix

def looks_like_html(text: str) -> bool:
    t = text.lstrip().lower()
    # chấp nhận khi có các dấu hiệu HTML rõ rệt
    return ("<html" in t) or ("<article" in t) or ("<div" in t and "<h1" in t)

def fetch_html_lenient(url: str) -> str | None:
    r = SESSION.get(url, timeout=TIMEOUT, allow_redirects=True)
    if not r.ok:
        return None
    ct = r.headers.get("Content-Type","").lower()
    # chấp nhận text/html *hoặc* text/plain miễn trông giống HTML
    if "text/html" in ct:
        return r.text if looks_like_html(r.text) else r.text  # cứ trả, TechDocs đôi khi tối giản
    if "text/plain" in ct:
        return r.text if looks_like_html(r.text) else None
    # fallback: nếu server set ct lạ nhưng nội dung là HTML
    return r.text if looks_like_html(r.text) else None

def pick_md_article(soup: BeautifulSoup) -> Tag:
    md = soup.select_one('div.md-content[data-md-component="content"]') \
         or soup.select_one("div.md-content") \
         or soup
    return md.select_one("article.md-content__inner.md-typeset") or md

def drop_noise(root: Tag):
    for sel in [
        "header","footer","nav",".md-header",".md-footer",".md-sidebar",".md-nav",
        ".toc","#toc",".breadcrumb",".breadcrumbs",
        "#toggle-sidebar",".md-content__button",".edit-this-page",
        "script","style","noscript","form","iframe"
    ]:
        for n in root.select(sel):
            n.decompose()

def node_to_text(n: Tag) -> str:
    if isinstance(n, NavigableString): return str(n).strip()
    if not isinstance(n, Tag): return ""
    nm = n.name.lower()
    if nm == "p": return n.get_text(" ", strip=True)
    if nm in ("ul","ol"):
        return "\n".join("- " + li.get_text(" ", strip=True) for li in n.find_all("li", recursive=False))
    if nm in ("h1","h2","h3","h4"): return "\n" + n.get_text(" ", strip=True) + "\n"
    if nm in ("pre","code"): return "\n" + n.get_text("\n", strip=True) + "\n"
    return n.get_text(" ", strip=True)

def extract_text_and_images(html: str, base_url: str) -> tuple[str, list[str]]:
    soup = BeautifulSoup(html, "lxml")
    base_tag = soup.find("base", href=True)
    base = base_tag["href"] if base_tag else base_url

    article = pick_md_article(soup)
    drop_noise(article)

    parts = []
    for blk in article.find_all(["h1","h2","h3","p","ul","ol","pre","code"], recursive=True):
        t = node_to_text(blk)
        if t: parts.append(t)
    text = normalize_ws("\n".join(parts))

    seen, imgs = set(), []
    for img in article.find_all("img"):
        chosen = None
        srcset = img.get("srcset")
        if srcset:
            best = None
            for part in srcset.split(","):
                part = part.strip()
                if not part: continue
                toks = part.split()
                u = toks[0]
                w = int(toks[1][:-1]) if len(toks)>1 and toks[1].endswith("w") and toks[1][:-1].isdigit() else 0
                if not best or w > best[0]:
                    best = (w, urljoin(base, u))
            if best: chosen = best[1]
        if not chosen:
            src = img.get("src") or img.get("data-src")
            if src: chosen = urljoin(base, src)
        if chosen and chosen not in seen:
            seen.add(chosen); imgs.append(chosen)

    return text, imgs

def main():
    ap = argparse.ArgumentParser(description="Scrape Backstage TechDocs static HTML (accept text/plain).")
    ap.add_argument("url", help="Backstage /docs/... URL")
    args = ap.parse_args()

    base, suffix = build_suffix(args.url)

    html = None
    source = None
    for pat in STATIC_PREFIXES:
        cand = urljoin(base, pat.format(suffix=suffix))
        html = fetch_html_lenient(cand)
        print(f"TRY: {cand} -> {'OK' if html else 'NO'}")
        if html:
            source = cand
            break

    if not html:
        print("===== TEXT ====="); print("(no text found)")
        print("\n===== IMAGE LINKS ====="); print("(none)")
        return

    text, imgs = extract_text_and_images(html, source)
    print("===== TEXT =====")
    print(text if text else "(no text found)")
    print("\n===== IMAGE LINKS =====")
    if imgs:
        for i,u in enumerate(imgs,1): print(f"{i}. {u}")
    else:
        print("(none)")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import sys
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup

TIMEOUT = 25

def fetch(url, headers=None, verify=True):
    sess = requests.Session()
    sess.headers.update({"User-Agent": "ScrapeDiag/1.0 (+diagnostic)"})
    if headers:
        sess.headers.update(headers)
    try:
        resp = sess.get(url, timeout=TIMEOUT, allow_redirects=True, verify=verify)
        return resp, None
    except Exception as e:
        return None, e

def summarize_redirects(resp):
    hops = []
    for r in resp.history:
        hops.append((r.status_code, r.url))
    hops.append((resp.status_code, resp.url))
    return hops

def is_html(resp):
    ctype = resp.headers.get("Content-Type", "")
    return "text/html" in ctype.lower() or resp.text.strip().startswith("<")

def looks_like_login(html_text, url):
    txt = html_text.lower()
    if any(x in txt for x in ["signin", "log in", "login", "đăng nhập", "authenticate", "sso"]):
        return True
    # Backstage auth page often has /auth or provider buttons
    if any(x in txt for x in ["/auth/", "backstage", "google", "github", "microsoft"]) and "login" in txt:
        return True
    # redirect to auth domain
    parsed = urlparse(url)
    if "auth" in parsed.path.lower():
        return True
    return False

def detect_spa(html_text):
    # Heuristics: very small body, lots of scripts, root container divs, noscript warning
    soup = BeautifulSoup(html_text, "html.parser")
    scripts = soup.find_all("script")
    body_text_len = len(soup.get_text(" ", strip=True))
    has_root = soup.find(id=re.compile(r"^root$|^app$|^__next$|^docs-root$")) or soup.find("main") and body_text_len < 400
    noscript = soup.find("noscript")
    many_scripts = len(scripts) >= 5
    tiny_html = body_text_len < 300
    return (has_root or noscript) and (many_scripts or tiny_html)

def detect_backstage(html_text, resp):
    txt = html_text.lower()
    h = resp.headers
    clues = []
    if "backstage" in txt:
        clues.append("HTML mentions 'Backstage'")
    # asset names
    if any(k for k in h.keys() if k.lower().startswith("x-backstage-")):
        clues.append("Response has X-Backstage-* headers")
    # look for TechDocs marker
    if "techdocs" in txt or "/api/techdocs/" in txt:
        clues.append("Mentions TechDocs")
    return (len(clues) > 0), clues

def guess_techdocs_url(url):
    """
    Nếu URL là dạng /docs/<namespace>/<kind>/<name>/..., đoán đường dẫn static:
    /api/techdocs/static/docs/<namespace>/<kind>/<name>/index.html
    """
    p = urlparse(url)
    parts = [x for x in p.path.split("/") if x]
    try:
        i = parts.index("docs")
        segs = parts[i+1:i+4]  # namespace, kind, name
        if len(segs) == 3:
            return urljoin(f"{p.scheme}://{p.netloc}", f"/api/techdocs/static/docs/{segs[0]}/{segs[1]}/{segs[2]}/index.html")
    except ValueError:
        pass
    return None

def short_body_preview(text):
    s = re.sub(r"\s+", " ", text.strip())
    return s[:300] + ("..." if len(s) > 300 else "")

def main():
    ap = argparse.ArgumentParser(description="Diagnose what is needed to scrape a page (auth, SPA, TechDocs, etc.)")
    ap.add_argument("url", help="Target URL")
    ap.add_argument("--cookie", help="Cookie header to try (e.g., 'backstage-session=...; other=...')", default=None)
    ap.add_argument("--auth", help="Authorization header to try (e.g., 'Bearer TOKEN')", default=None)
    ap.add_argument("--insecure", action="store_true", help="Disable TLS verify")
    args = ap.parse_args()

    headers = {}
    if args.cookie:
        headers["Cookie"] = args.cookie
    if args.auth:
        headers["Authorization"] = args.auth

    resp, err = fetch(args.url, headers=headers, verify=not args.insecure)
    print("="*80)
    print("URL:", args.url)
    if err:
        print("ERROR:", err)
        print("=> Có thể cần VPN, tin cậy CA nội bộ (--insecure), hoặc cung cấp Cookie/Bearer token.")
        sys.exit(2)

    # Redirects
    hops = summarize_redirects(resp)
    print("REDIRECT CHAIN:")
    for code, u in hops:
        print(f"  {code} -> {u}")

    # Status & headers
    print("STATUS:", resp.status_code)
    ctype = resp.headers.get("Content-Type", "")
    clen = resp.headers.get("Content-Length", "")
    set_cookies = resp.headers.get("Set-Cookie", "")
    print("CONTENT-TYPE:", ctype or "(unknown)")
    print("CONTENT-LENGTH:", clen or f"(len(body)={len(resp.content)})")

    # Basic inferences
    html_flag = is_html(resp)
    print("IS_HTML:", html_flag)
    print("IS_PDF:", ("application/pdf" in ctype.lower()) if ctype else False)

    # Need auth?
    need_auth = False
    note_auth = []
    if resp.status_code in (401, 403):
        need_auth = True
        note_auth.append(f"HTTP {resp.status_code} suggests auth needed")
    if resp.history and any(r.status_code in (301,302,303,307,308) and "auth" in r.headers.get("Location","").lower() for r in resp.history if r.headers.get("Location")):
        need_auth = True
        note_auth.append("Redirected to an auth endpoint")
    if html_flag and looks_like_login(resp.text, resp.url):
        need_auth = True
        note_auth.append("Login keywords found in HTML")

    # SPA?
    spa = False
    if html_flag:
        spa = detect_spa(resp.text)

    # Backstage / TechDocs?
    backstage, backstage_clues = (False, [])
    techdocs_guess = None
    techdocs_ok = None
    techdocs_status = None
    techdocs_url = None
    if html_flag:
        backstage, backstage_clues = detect_backstage(resp.text, resp)
        techdocs_guess = guess_techdocs_url(resp.url)
        techdocs_url = techdocs_guess
        if techdocs_guess:
            test_resp, terr = fetch(techdocs_guess, headers=headers, verify=not args.insecure)
            if terr:
                techdocs_ok = False
                techdocs_status = f"ERROR: {terr}"
            else:
                techdocs_ok = (test_resp.status_code == 200 and "text/html" in test_resp.headers.get("Content-Type","").lower())
                techdocs_status = f"{test_resp.status_code} {test_resp.headers.get('Content-Type','')}"
    # Cookie hints
    cookie_names = []
    if set_cookies:
        # try to extract cookie names from Set-Cookie header
        for part in set_cookies.split(","):
            m = re.match(r"\s*([^=;\s]+)=", part.strip())
            if m:
                cookie_names.append(m.group(1))
    cookie_names = list(dict.fromkeys(cookie_names))  # unique order

    # Print preview/body head
    preview = ""
    if html_flag:
        preview = short_body_preview(resp.text)

    print("-"*80)
    print("NEED_AUTH:", need_auth)
    if note_auth:
        for n in note_auth:
            print("  •", n)
    print("DETECTED_SPA:", spa)
    print("DETECTED_BACKSTAGE:", backstage)
    if backstage and backstage_clues:
        for c in backstage_clues:
            print("  •", c)

    if cookie_names:
        print("SET-COOKIE NAMES (server):", ", ".join(cookie_names))
    else:
        print("SET-COOKIE NAMES (server): (none seen)")

    if techdocs_guess:
        print("TECHDOCS_GUESS:", techdocs_guess)
        print("TECHDOCS_CHECK:", techdocs_status, "=>", "OK" if techdocs_ok else "NOT OK")
    else:
        print("TECHDOCS_GUESS: (n/a)")

    if html_flag:
        print("-"*80)
        print("HTML_PREVIEW:", preview)

    print("="*80)
    print("WHAT TO PROVIDE (if scraping fails):")
    if need_auth or resp.status_code in (401,403):
        print("  - Cookie header (ví dụ): Cookie: backstage-session=<JWT>; other=... ")
        print("  - HOẶC Authorization: Bearer <TOKEN>")
    print("  - Xác nhận có cần VPN / IP whitelist không")
    if spa:
        print("  - Nếu là SPA/JS-rendered, cân nhắc dùng Playwright. Có thể cần file cookie xuất từ trình duyệt.")
    if techdocs_guess and (techdocs_ok is False):
        print("  - Nếu Backstage dùng TechDocs, kiểm tra quyền truy cập vào static path trên.")
    if not need_auth and not spa and not (techdocs_guess and techdocs_ok):
        print("  - Có thể chỉ cần requests thường (không auth).")
    if cookie_names:
        print("  - Server gợi ý cookie tên:", ", ".join(cookie_names))
    print("="*80)

if __name__ == "__main__":
    main()

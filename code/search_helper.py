#!/usr/bin/env python3
"""Manual literature search via arXiv API and Semantic Scholar API."""
import sys, time, json, urllib.parse
import httpx
import xml.etree.ElementTree as ET

def arxiv_search(query, max_results=8):
    base = "https://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = base + "?" + urllib.parse.urlencode(params)
    r = httpx.get(url, timeout=30, follow_redirects=True)
    r.raise_for_status()
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(r.text)
    out = []
    for e in root.findall("a:entry", ns):
        aid = e.find("a:id", ns).text.strip()
        title = " ".join(e.find("a:title", ns).text.split())
        summ = " ".join(e.find("a:summary", ns).text.split())
        authors = [a.find("a:name", ns).text for a in e.findall("a:author", ns)]
        pub = e.find("a:published", ns).text[:10]
        out.append({"id": aid, "title": title, "authors": authors[:6],
                    "published": pub, "abstract": summ})
    return out

def ss_search(query, limit=8):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": query, "limit": limit,
              "fields": "title,abstract,year,authors,citationCount,externalIds,openAccessPdf"}
    for attempt in range(4):
        try:
            r = httpx.get(url, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1)); continue
            r.raise_for_status()
            return r.json().get("data", [])
        except Exception as ex:
            time.sleep(3 * (attempt + 1))
    return []

if __name__ == "__main__":
    source = sys.argv[1]  # arxiv or ss
    query = sys.argv[2]
    if source == "arxiv":
        res = arxiv_search(query, int(sys.argv[3]) if len(sys.argv) > 3 else 8)
    else:
        res = ss_search(query, int(sys.argv[3]) if len(sys.argv) > 3 else 8)
    print(json.dumps(res, indent=2))

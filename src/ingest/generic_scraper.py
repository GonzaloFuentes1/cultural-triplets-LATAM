import requests
from bs4 import BeautifulSoup

def scrape_url(url: str) -> str:
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script","style","sup","table","figure","header","footer"]):
        tag.decompose()
    return "\n".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))

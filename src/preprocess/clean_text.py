from bs4 import BeautifulSoup
import re

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script","style","sup","table","figure","header","footer"]):
        tag.decompose()
    text = "\n".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
    text = re.sub(r"\s+", " ", text).strip()
    return text

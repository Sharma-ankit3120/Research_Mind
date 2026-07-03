from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os 
from dotenv import load_dotenv
from rich import print
load_dotenv()



# Create a Tavily Tool

taviy = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def web_search(query:str)->str:
    "Search across the web and return accuracte reliable information acc to query, title ,therir urls and snippets"

    results = taviy.search(query=query,max_results=3)

    out = []

    for r in  results['results']:
        out.append(
        f"\n ✔️ Title: {r['title']}\n 🔗 URL: {r['url']} \n  📄 Snippet: {r['content'][:300]}\n"
        )

    return "\n----\n".join(out)


# print(web_search.invoke("Latest news on war"))



# web scapping tool

@tool
def scrape_url(url:str)->str:
    """Scrape and return clean text content from a given url for deeper understanding"""

    try:
        resp = requests.get(url,timeout=8,headers={"User-agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text,"html.parser")
        for tag in soup(["script","style","nav","footer"]):
            tag.decompose()
        return soup.get_text(separator=" ",strip=True)[:3000]
    except Exception as e:
        return f"Could not scrape text from URL : {str(e)}"
    


# print(scrape_url.invoke("https://docs.langchain.com/langsmith/prompt-template-format#prompt-template-format-guide"))


      
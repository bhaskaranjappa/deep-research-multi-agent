import urllib.request
import json
import re

def native_web_search(query: str) -> str:
    """
    Executes a zero-cost public web lookup via HTML/Text parsing or open API mock.
    Returns synthesized text chunks from the top search vectors.
    """
    try:
        # Clean query string for URL parameters
        safe_query = urllib.parse.quote_plus(query)
        
        # Using DuckDuckGo's open HTML/lite endpoint or an unauthenticated API hook
        url = f"https://html.duckduckgo.com/html/?q={safe_query}"
        
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        
        with urllib.request.urlopen(req, timeout=5) as response:
            html = response.read().decode('utf-8')
            
        # Clean out HTML tags to extract raw text snippets for context assembly
        snippets = re.findall(r'<a class="result__snippet".*?>(.*?)</a>', html, re.DOTALL)
        clean_text = "\n".join([re.sub(r'<[^>]+>', '', s).strip() for s in snippets[:3]])
        
        return clean_text if clean_text else f"No live search results found for: {query}"
    except Exception as e:
        return f"Search execution failed for query '{query}': {str(e)}"
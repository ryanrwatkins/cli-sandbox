import requests
from bs4 import BeautifulSoup
import time

def get_recent_articles(search_query="cs.AI"):
    """
    Scrapes arXiv for articles from the last 24 hours.
    """
    base_url = "https://arxiv.org"
    list_url = f"{base_url}/list/{search_query}/pastweek?skip=0&show=1000"
    
    response = requests.get(list_url)
    soup = BeautifulSoup(response.content, 'lxml')
    
    articles = []
    
    dl_element = soup.find('dl')
    if not dl_element:
        return articles

    dt_elements = dl_element.find_all('dt')
    dd_elements = dl_element.find_all('dd')

    for i in range(len(dt_elements)):
        dt = dt_elements[i]
        dd = dd_elements[i]

        # Get link to abstract page
        abs_link_tag = dt.find('a', title='Abstract')
        if not abs_link_tag:
            continue
            
        abs_page_url = base_url + abs_link_tag['href']
        
        # Be polite and sleep for a second
        time.sleep(1)
        
        # Get content from abstract page
        try:
            abs_response = requests.get(abs_page_url)
            abs_soup = BeautifulSoup(abs_response.content, 'lxml')
            
            title_h1 = abs_soup.find('h1', class_='title')
            title = title_h1.text.replace('Title:', '').strip() if title_h1 else "No Title Found"
            
            abstract_bq = abs_soup.find('blockquote', class_='abstract')
            abstract = abstract_bq.text.replace('Abstract:', '').strip() if abstract_bq else ""
            
            pdf_link_tag = abs_soup.find('a', class_='download-pdf')
            pdf_link = base_url + pdf_link_tag['href'] if pdf_link_tag else ""
            
            articles.append({
                'title': title,
                'abstract': abstract,
                'link': pdf_link
            })
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {abs_page_url}: {e}")

    return articles
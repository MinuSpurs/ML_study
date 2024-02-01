import requests
from bs4 import BeautifulSoup

all_jobs = []


skills = [
    "python",
    "typescript",
    "javascript",
    "rust"
]

def scrape_page(url):
    print(f'Scrapping {url}...')
    response = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
    )

    soup = BeautifulSoup(response.content, 'html.parser',)
    
    jobs = soup.find('ul', class_='jobs-list-items').find_all('li')
    
    for job in jobs:
        position = job.find('div', class_='bjs-jlid__meta').find('a', class_='bjs-jlid__b').text
        title = job.find('div', class_='bjs-jlid__meta').find('h4', class_='bjs-jlid__h').text
        link = job.find('div', class_='bjs-jlid__meta').find('a')
        if link:
            link = link['href']        
        job_data = {
            'title': title,
            'position': position,
            'link': link
        }
        all_jobs.append(job_data)

    


def get_pages(url):
    response = requests.get(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })

    soup = BeautifulSoup(response.content, 'html.parser',)

    return len((soup.find('ul', class_='bsj-nav').text)[1:3])

url = "https://berlinstartupjobs.com/engineering/"

total_page = get_pages(url)

for x in range(total_page):
    url = f'https://berlinstartupjobs.com/engineering/page/{x+1}/'
    print(url)
    
url = ''
    
for skill in skills:
    url = f'https://berlinstartupjobs.com/skill-areas/{skill}/'
    scrape_page(url)

print(len(all_jobs))
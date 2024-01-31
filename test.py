import requests
from bs4 import BeautifulSoup

all_jobs = []

response = requests.get(
    "https://berlinstartupjobs.com/engineering/",
    headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
)

skills = [
    "python",
    "typescript",
    "javascript",
    "rust"
]

def scrape_page(url):
    print(f'Scrapping {url}...')
    response = requests.get(url)

    soup = BeautifulSoup(response.content, 'html.parser',)

    jobs = soup.find('ul', class_='jobs-list-items').find_all('li')

    for job in jobs:
        title = job.find('span', class_='title').text
        company, position = job.find_all('span', class_='company')
        url = job.find('div', class_='tooltip').next_sibling
        url = job.find('div', class_='a')
        if url:
            url = url['href']
        job_data = {
            'title': title,
            'company': company.text,
            'position': position.text,
            'url': f'{url}'
        }
        all_jobs.append(job_data)
'''
def get_pages(url):
    response = requests.get(url)

    soup = BeautifulSoup(response.content, 'html.parser',)

    return len(soup.find_all('div', class_='widget popular_skills').find_all('li', class_='link'))

url= 'https://berlinstartupjobs.com/engineering/'
total_page = get_pages(url)'''

for skill in skills:
    url = f'https://berlinstartupjobs.com/skill-areas/python/{skill}'
    scrape_page(url)

print(len(all_jobs))

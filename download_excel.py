import os
import pandas as pd
import requests
from bs4 import BeautifulSoup as bs 

def get_pdfs(_url):
    r = requests.get(_url)
    soup = bs(r.content,'html5lib')
    links = soup.findAll('a')
    pdf_links = ["https://link.springer.com" + link['href'] for link in links if link['href'].endswith('pdf')]

    #print(pdf_link)
 
    return pdf_links

def download_pdf(_url, title, cat):
    file_name = title
    print("Downloading book: {}".format(file_name))
    r = requests.get(_url[0], stream = True)
    dirr = cat + "/" + title + ".pdf"
    with open(dirr, "wb") as pdf:
        for chunk in r.iter_content(chunk_size=10000000):
            if chunk:
                pdf.write(chunk)

if __name__ == '__main__':
    path_ = os.getcwd() #mozna dac dir gdzie chcesz to zapisac, domyslnie tworzy foldery w katalogu ze skryptem
    xlsx_path = os.path.join(path_, 'm.xlsx')
    df = pd.read_excel(xlsx_path) ## tu dajesz dir do xlsx'a
    n = df.shape[0]


    names = pd.unique(df['English Package Name'])
    for name in names:
        name_ = os.path.join(path_, name)
        os.mkdir(name_)

    for i, _url in enumerate(df['OpenURL']):
        title = df['Book Title'][i]
        category = df['English Package Name'][i]
        download_pdf(get_pdfs(_url),title, category)
        print("{}/{}".format(i+1,n))




###############################################
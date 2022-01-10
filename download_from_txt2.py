# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 03:15:31 2021

@author: Forty
"""

import sys
import os
import requests
import pandas as pd


if __name__ == "__main__":
    if len(sys.argv)!=5:
        sys.exit("Wrong number of arguments given. Correct syntax: download_from_txt.py file_with_links no_of_column_with_links no_of_column_with_names no_of_header_lines")
    else:
        links_list = sys.argv[1]
        links = int(sys.argv[2])
        names = int(sys.argv[3])
        head = int(sys.argv[4])
	
table = pd.read_csv(links_list, skiprows = head, header=None)
os.mkdir("downloaded_files")
#main_link = "http://iimcb.genesilico.pl/NPDock/media/jobs//5d00de28-09df-419f-8438-16b3a92c9bfa/prot-na_15/"

for i in table[0]:
    i=i.split()
    link=i[links].rstrip('>').lstrip('<')
    outfile = requests.get(link)
    open("downloaded_files/"+i[names], 'wb').write(outfile.content)
    print(f"Downloading {i[names]} file.")



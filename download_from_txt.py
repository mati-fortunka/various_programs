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
        sys.exit("Wrong number of arguments given. Correct syntax: download_from_txt.py file_with_links no_of_column_with_link_endings no_of_header_lines main_part_of_links")
    else:
        links_list = sys.argv[1]
        column = int(sys.argv[2])
        head = int(sys.argv[3])
        main_link = sys.argv[4]

#main_link = "http://iimcb.genesilico.pl/NPDock/media/jobs//5d00de28-09df-419f-8438-16b3a92c9bfa/prot-na_15/"
links = pd.read_csv(links_list, skiprows = head, header=None)
os.mkdir("downloaded_files")

for i in links[0]:
    i=i.split()
    outfile = requests.get(main_link+i[column])
    open("downloaded_files/"+i[column], 'wb').write(outfile.content)
    print(f"Downloading {i[column]} file.")



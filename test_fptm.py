import os
import sys
import requests
import glob
from bs4 import BeautifulSoup
import pandas as pd

# fName = 'P51688'
fName = sys.argv[1]
proxies = {'https':'proxy.ibab.ac.in:3128', 'http':'proxy.ibab.ac.in:3128'}
url = 'https://legacy.uniprot.org/uniprot/'+fName+'#ptm_processing'
r = requests.get(url, proxies=proxies)
soup = BeautifulSoup(r.text, 'html.parser')
table = soup.find(id = "aaMod_section")
df = pd.read_html(str(table))[0]['Position(s)']
aaM = []
for sRes in df:
    if sRes.find(' ↔ ') != -1:
        res = sRes.split(' ↔ ')
        aaM.extend(res) 
    else:
        aaM.append(sRes)
aaMstr = ','.join(aaM)
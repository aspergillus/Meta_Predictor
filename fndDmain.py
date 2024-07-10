import os
import sys
import requests
import glob
from bs4 import BeautifulSoup
import pandas as pd

rndNum = sys.argv[1]
fName = sys.argv[2]
prntDir = os.getcwd()
# fName = glob.glob(f"{prntDir}/fileUpload/{rndNum}/"'*.fasta')[0].rsplit('/',1)[1].split('.')[0]
# rndNum = '395'
# fName = 'P34059'

# Post translation modification
proxies = {'https':'http://proxy.ibab.ac.in:3128', 'http':'http://proxy.ibab.ac.in:3128'}
url = f"https://legacy.uniprot.org/uniprot/{fName}#ptm_processing"
r = requests.get(url, proxies=proxies)
# r = requests.get(url)
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
# aaMstr = '79,204,308,419,423,489,518,501,507'

# Make domain file with which domain and post-translation modification
os.chdir(f"{prntDir}/featureModel")
os.system('bash 003-scan ../fileUpload/'+rndNum+'/'+fName+'.fasta')
os.system('bash 004-extract-coords ../fileUpload/'+rndNum+'/'+fName+'.fasta-found-domains.tab')
os.system('time ./005-extract-seq '+fName+'.fasta '+rndNum+' '+aaMstr+'')
os.system('cp hydrophobicity.csv polarity.csv residual_volume.csv ../fileUpload/'+rndNum+'/')
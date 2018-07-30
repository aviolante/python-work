# Train Data
# Total: 415775
# Label:
#     - Other:       23,299
#     - Clear:       139,865
#     - Precipitate: 199,567
#     - Crystals:    53,044
# Source:
#     - C3:          13,111
#     - GSK:         70,248
#     - HWI:         67,225
#     - Merck:       257,889
#     - BMS:         7,302
# Format:
#     - PNG:         67,225
#     - JPEG:        348,550

# Last Page for Range:
clear_end = 7886
precipitate_end = 11256
other_end = 1323
crystals_end = 3007

from bs4 import BeautifulSoup as soup
import requests
import contextlib
import re
import os

@contextlib.contextmanager
def get_images(url: str):
   d = soup(requests.get(url).text, 'html.parser')
   yield [[i.find('img')['src'], re.findall('(?<=\.)\w+$', i.find('img')['alt'])[0]] for i in
          d.find_all('a') if re.findall('/image/\d+', i['href'])]

# Edit directory for class Label
os.chdir('/Users/anviol/Desktop/MARCO Project/train/Clear')

# Edit get_images(URL) with new class
n = clear_end  # end value
for i in range(300,n):
   with get_images(f'https://marco.ccr.buffalo.edu/images?page={i}&score=Clear') as links:
       print(links)
       for c, [link, ext] in enumerate(links, 1):
           with open(f'MARCO_img_{i}{c}.{ext}', 'wb') as f:
               f.write(requests.get(f'https://marco.ccr.buffalo.edu{link}').content)

from urllib.request import urlopen
 
from bs4 import BeautifulSoup
import pandas as pd
import re

def get_product_id(cell_value):
    s = str(cell_value)
    return s[62:69]	

def get_all_products(html_soup):
    products = []
    all_rows_in_html_page = html_soup.findAll("article")
    for table_row in all_rows_in_html_page:
        row_cells=table_row.findAll("span")
        row_cells2=table_row.findAll("a")
        product_entry = {
                "id": get_product_id(row_cells2[0]),
                "name": row_cells[0].text,
                "price": row_cells[1].text,
                "quantity": row_cells[2].text,
        }
        products.append(product_entry)
    return products
	

html = urlopen("https://shop.maxi.rs/maxi-online.html?search=1&mark=1")
html_soup = BeautifulSoup(html, 'html.parser')
product_list = get_all_products(html_soup)

df = pd.DataFrame(product_list)
df.head(5)


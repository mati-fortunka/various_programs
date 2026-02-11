from os import wait

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

chrome_options = Options()
chrome_options.binary_location = "/usr/bin/google-chrome"
# chrome_options.add_argument("--headless=new")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

service = Service("/home/matifortunka/Programs/chromedriver-linux64/chromedriver")

driver = webdriver.Chrome(service=service, options=chrome_options)

driver.get("https://www.google.com")
print("Page title:", driver.title)
# wait()
driver.quit()

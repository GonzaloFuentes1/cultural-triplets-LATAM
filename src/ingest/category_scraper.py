from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import urllib.parse


def scrape_categories_by_country(country):
    categories = []
    driver = None
    try:
        driver = webdriver.Chrome()
        
        base_url = "https://es.wikipedia.org/w/index.php?title=Especial:Categor%C3%ADas&from="
        start_url = base_url + urllib.parse.quote(country)
        
        driver.get(start_url)
        time.sleep(2)

        try:
            five_hundred_link = driver.find_element(By.LINK_TEXT, "500")
            url_500 = five_hundred_link.get_attribute('href')
            driver.get(url_500)
            time.sleep(5)
        except Exception as e:
            print(f"Could not find or click the '500' link for {country}: {e}")

        page_num = 1
        while True:
            page_categories = []
            stop_scraping = False
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.mw-pager-navigation-bar + ul")))
                
                category_list = driver.find_element(By.CSS_SELECTOR, "div.mw-pager-navigation-bar + ul")
                for link in category_list.find_elements(By.TAG_NAME, "a"):
                    title = link.get_attribute("title")
                    if country in title:
                        page_categories.append(title)
                    else:
                        stop_scraping = True
                        break
            except Exception as e:
                print(f"Error waiting for page {page_num} to load for {country}: {e}")
                break

            if page_categories:
                categories.extend(page_categories)
            
            if stop_scraping:
                print(f"Found a category that does not contain '{country}'. Stopping.")
                break

            try:
                next_button = driver.find_element(By.PARTIAL_LINK_TEXT, "siguientes")
                next_page_url = next_button.get_attribute('href')
                driver.get(next_page_url)
                time.sleep(5)
                page_num += 1
            except Exception:
                break

    finally:
        if driver:
            driver.quit()
            
    return categories

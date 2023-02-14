def crawling3(area):
    import selenium
    from selenium import webdriver
    from selenium.webdriver import ActionChains

    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.common.by import By

    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import Select
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.common.action_chains import ActionChains
    from bs4 import BeautifulSoup
    from tqdm import tqdm

    import time
    import pandas as pd

    from selenium.webdriver.chrome.options import Options

           
    
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('window-size= 1920,1080')
    chrome_options.add_argument('--kiosk')

    driver = webdriver.Chrome(executable_path='/opt/ml/input/project/airflow/dags/chromedriver', chrome_options=chrome_options)
    
    data = pd.read_csv(f'/opt/ml/input/project/airflow/dags/area_csv/{area}/rest_concat.csv')
    path = f'/opt/ml/input/project/airflow/dags/user_csv/{area}/'
    current_status = 0
    start = 0
    end = len(data)
    
    url_list = list(data['url'].values)
    userlink = pd.DataFrame()
    
    for _url in tqdm(url_list[start:end]):
        try:
            driver = webdriver.Chrome(executable_path='/opt/ml/input/project/airflow/dags/chromedriver', chrome_options=chrome_options)
            action = ActionChains(driver)
            print(_url)
            URL = f"https://m.place.naver.com/restaurant/{_url}/review/visitor"
            driver.get(URL)
            time.sleep(2.5)
            count = 0
            flag = False
            while True:
                try: action.move_to_element(driver.find_element(By.CLASS_NAME, "lfH3O")).click().perform()
                except: break
                print("\r",count, end="")
                count+= 1
                if count >= 60: flag = True; break
            print("click 1/2 complete")
            time.sleep(2.5)
            try:
                #action.move_to_element(driver.find_elements(By.CLASS_NAME, "YeINN")[-1]).perform()  #선택 리뷰 창 + 버튼 누르기
                driver.find_element(By.CLASS_NAME, 'I8cuq').click()
            except: print("NO 선택리뷰창")
            time.sleep(2.5)
            action = ActionChains(driver)
            while True:
                try:
                    action.move_to_element(driver.find_element(By.CLASS_NAME, "lfH3O")).click().perform()
                except:
                    break
                print("\r",count, end="")
                count+= 1
                if count >= 60: flag = True; break
            print("click 2/2 complete")
            if flag:
                with open(path + f"notsaved_{start}.txt", "a") as file:
                    file.write(f"{str(current_status)}\n")
                    file.close()
                current_status += 1
                continue
            html = driver.page_source
            soup = BeautifulSoup(html,'html.parser')
            user = soup.find_all(class_='YeINN')
            link_list = [i.a['href'] for i in user]
            user_list = [i.text for i in user]
            #time.sleep(5)
            print(_url, len(link_list), len(user_list))
            userlink2 = pd.DataFrame({'link' : link_list, 'user' : user_list}, dtype = str)
            userlink2['rest'] = _url
            userlink = pd.concat([userlink, userlink2], axis = 0, sort=False)
            userlink.to_csv(path + f'user_{start}.csv', index=False)#river_behind500
            with open(path + f"log.txt", "w") as file:
                file.writelines(str(current_status))
            current_status += 1
        except:
            with open(path + f"notsaved_{start}.txt", "a") as file:
                file.write(f"{str(current_status)}\n")
                file.close()
            current_status += 1
            continue
    print("End~")
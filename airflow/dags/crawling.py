def crawling(area, URLS):
    import selenium
    from selenium import webdriver
    from selenium.webdriver import ActionChains

    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.common.by import By

    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import Select
    from selenium.webdriver.support.ui import WebDriverWait

    from selenium.webdriver.chrome.options import Options

    import time
    import pandas as pd

    from tqdm import tqdm
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')

    # URLS : 주소 입력하기.
    # 각자 구의 행정동에 해당하는 링크들 하나씩 넣으면 됨  ex) 현재는 종로구 링크

    # 최근 네이버측의 주소록 변경 이슈 해결
    for i in range(len(URLS)):
        newURLs = URLS[i].split('?c=')
        newURL = newURLs[0] + '?c=14146181.1186290,4514995.3198595,14,0,0,0,dh'
        URLS[i] = newURL

    # 실행 전에 ./area_csv/ 폴더 안에 해당 구 폴더 생성하세요.
    # ex) ./area_csv/Jongno/

    driver = webdriver.Chrome(executable_path='/opt/ml/input/project/airflow/dags/chromedriver', options=chrome_options)
    path = f'/opt/ml/input/project/airflow/dags/area_csv/{area}/'
    # num : 1부터 시작해서 하나씩 알아서 늘어남
    for num, URL in enumerate(URLS, start=1):
        print(num)
        is_repeat = True
        while is_repeat:
            try:
                driver.quit()
                restaurants_list = pd.DataFrame()
                
                driver = webdriver.Chrome(executable_path='/opt/ml/input/project/airflow/dags/chromedriver', options=chrome_options)
                driver.get(URL)

                driver.switch_to.frame('searchIframe') #iframe 으로 이동하기 위한 코드

                epoch = 1 # 6이 최대인듯
                for _ in tqdm(range(epoch)):
                    click_list = driver.find_elements(By.CLASS_NAME, "tzwk0")
                    click_list[0].click() #아무거나 클릭을 해야 end키가 먹혀서 임의로 아무 클래스 클릭한거임
                    for _ in range(10):
                        driver.find_element(By.CSS_SELECTOR, "body").send_keys(Keys.END) # end키를 눌러서 아래까지 로딩
                    time.sleep(2)
                    click_list = driver.find_elements(By.CLASS_NAME, "tzwk0") # 식당 리스트
                    name_list = driver.find_elements(By.CLASS_NAME, "TYaxT") # 식당 이름 리스트
                    name_list = [user.text for user in name_list]
                    tag_list = driver.find_elements(By.CLASS_NAME, "KCMnt") # 식당 종류 리스트
                    tag_list = [user.text for user in tag_list]
                    review_list = driver.find_elements(By.CLASS_NAME, "MVx6e") # 리뷰 개수 리스트
                    review_list = [user.text.split("\n")[-1] for user in review_list]
                    url_list = []
                    for restaurant_link in click_list:
                        restaurant_link.click()
                        url_list.append(driver.current_url)
                        time.sleep(0.5)
                    restaurants_list2 = pd.DataFrame({'restaurant' : name_list, 'tag' : tag_list, 'url' : url_list, 'review' : review_list}, dtype = str)
                    restaurants_list = pd.concat([restaurants_list, restaurants_list2], axis = 0, sort=False)
                    driver.find_elements(By.CLASS_NAME, "yUtES")[1].click() #다음페이지 클릭
                    time.sleep(2)
                
                is_repeat = False

            except IndexError:
                print("IndexError!! Retry")
                ...

        restaurants_list.to_csv(f'{path}/rest_{num}.csv', index=False)
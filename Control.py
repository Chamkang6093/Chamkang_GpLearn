import os
import time
import itertools
import pandas as pd
from threading import Thread

from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions as EC



def login_worldquant_brain(account_num = 0, 
                           account_thread = 1, 
                           headless = False, 
                           log_info = False, 
                           parallel_mode = False, 
                           parallel_index = None, 
                           parallel_dict = None):

    account_dict = {
        0 : {"email": "xxxxxx", "pwd": "xxxxxx"},
        1 : {"email": "xxxxxx", "pwd": "xxxxxx"},
        2 : {"email": "xxxxxx", "pwd": "xxxxxx"},
        3 : {"email": "xxxxxx", "pwd": "xxxxxx"},
        4 : {"email": "xxxxxx", "pwd": "xxxxxx"},
        5 : {"email": "xxxxxx", "pwd": "xxxxxx"},
        6 : {"email": "xxxxxx", "pwd": "xxxxxx"}
    }

    if account_num not in account_dict.keys():
        raise AttributeError("Please reset account (engine) number !")

    selenium_path ='C:/Program Files/Mozilla Firefox/geckodriver.exe'

    service = Service(selenium_path)
    firefoxOptions = webdriver.FirefoxOptions()
    firefoxOptions.add_argument('--disable-infobars') 
    firefoxOptions.add_argument('--user-agent=""')
    firefoxOptions.add_argument('--blink-settings=imagesEnabled=false')
    firefoxOptions.add_argument('--incognito')
    firefoxOptions.add_argument('--hide-scrollbars')
    firefoxOptions.add_argument('--disable-javascript')
    firefoxOptions.add_argument('--disable-extensions')
    
    # (Might cause error after setting)
    # *Disable GPU 
    # firefoxOptions.add_argument('--disable-gpu')
    # *Maximize Browser
    # firefoxOptions.add_argument('--start-maximized')
    # *Set Browser Resolution (window size)
    # firefoxOptions.add_argument('--window-size=1280x1024')
    # firefoxOptions.add_argument('--ignore-certificate-errors') 
    # firefoxOptions.add_argument('log-level=3')
    # firefoxOptions.add_argument('–disable-software-rasterizer')
    
    if headless:
        firefoxOptions.add_argument('--headless')
        
    driver = webdriver.Firefox(service = service, options=firefoxOptions) 
    
    login_error_count = 0
    while True:
        try:
            driver.get('https://platform.worldquantbrain.com/sign-in')
            if log_info:
                print("---------------------------------------------------")
                print("Try to Access Login Website.")
                
            email_input = WebDriverWait(driver, 30, 0.5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="email"]')))
            email_input.send_keys(account_dict[account_num]["email"])
            pwd_input = driver.find_element(by=By.XPATH, value='//*[@id="password"]')
            pwd_input.send_keys(account_dict[account_num]["pwd"])
            if log_info:
                print("Input Username and Password.")
                print("Try to Login on %s (Thread %s)" % (account_dict[account_num]["email"], account_thread))
                
            login_btn = driver.find_element(by=By.XPATH, value='/html/body/div/div/section/div/article/div/div/form/div[3]/button')
            try:
                login_btn.click()
                if log_info:
                    print("Click Login Button.")
            except:
                if log_info:
                    print("Try to click Login Button. But there are Cookie Buttons.")
                cookie_btn = driver.find_element(by=By.XPATH, value='/html/body/div/div[2]/div/div/div/div[2]/button[2]')
                cookie_btn.click()
                if log_info:
                    print("Click Cookie Accept Button.")
                login_btn.click()
                if log_info:
                    print("Click Login Button.")
            if log_info:
                print("Successfully Enter Simulate Website.")
                print("---------------------------------------------------")
                
            code_input = WebDriverWait(driver, 60, 0.5).until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div.view-line:nth-child(1)')))
            try:
                code_input.click()
                if log_info:
                    print("Click Coding Block.")
            except:
                if log_info:
                    print("Try to Click Coding Block. But there are Skip Buttons.")
                time.sleep(9)
                skip_btn = WebDriverWait(driver, 30, 0.5).until(EC.presence_of_element_located((By.XPATH, '/html/body/div[6]/div/div[5]/a[1]')))
                skip_btn.click()
                if log_info:
                    print("Click Skip Button.")
                time.sleep(1)
                code_input = WebDriverWait(driver, 30, 0.5).until(EC.presence_of_element_located((By.XPATH, '/html/body/div[1]/div/div[2]/div[3]/div[2]/div/div[1]/div/div[2]/div/div[1]/div/div[1]/div[2]')))
                code_input.click()
                if log_info:
                    print("Click Coding Block.")
                    
            try:
                driver.find_element(by = By.XPATH, value = "/html/body/div[1]/div/div[2]/div[3]/div[2]/div/div[2]/div/div/div[1]/div[1]/div/span")
                if log_info:
                    print("We have Already Opened RESULTS Panel.")
            except:
                results_btn = driver.find_element(by = By.XPATH, value = "/html/body/div[1]/div/div[2]/div[2]/div[4]/div/div[2]/div[1]")
                results_btn.click()
                if log_info:
                    print("We do not Open RESULTS Panel, so Click RESULTS Button to Open it.")
                driver.find_element(by = By.XPATH, value = "/html/body/div[1]/div/div[2]/div[3]/div[2]/div/div[2]/div/div/div[1]/div[1]/div/span")
                    
            if log_info:
                print("And We Can Start to Create Alphas!")
                print("---------------------------------------------------")
        except:
            login_error_count += 1
            if login_error_count == 3:
                print("Login on %s (Thread %s) Failed! Login Error for 3 times."% (account_dict[account_num]["email"], account_thread))
                if parallel_mode:
                    parallel_dict[parallel_index] = False
                    return
                else:
                    return False
            else:
                driver.quit()
                if log_info:
                    print("=========================")
                    print("Login Error for %s time(s). " % (login_error_count), end="")
                    print("Try to login again.")
                    print("=========================")
        else:
            print("Login on %s (Thread %s) Succeeded!" % (account_dict[account_num]["email"], account_thread))
            if parallel_mode:
                parallel_dict[parallel_index] = driver
                return
            else:
                return driver



def multi_worldquant_brain(thread_num, headless = False, log_info = True):

    threads_dict = {
                        1: {"account" : 1, "thread" : 1},
                        2: {"account" : 1, "thread" : 2},
                        3: {"account" : 2, "thread" : 1},
                        4: {"account" : 2, "thread" : 2},
                        5: {"account" : 3, "thread" : 1},
                        6: {"account" : 3, "thread" : 2},
                        7: {"account" : 4, "thread" : 1},
                        8: {"account" : 4, "thread" : 2},
                        9: {"account" : 5, "thread" : 1},
                        10: {"account" : 5, "thread" : 2},
                        11: {"account" : 6, "thread" : 1},
                        12: {"account" : 6, "thread" : 2}
    }

    if thread_num > 12 or thread_num <= 0:
        raise AttributeError("'thread_num' must be 'int' type and between 0 and 12 !")

    process_pool = []
    engines = {}
    for i in range(1, thread_num + 1):
        engines[i] = None
    for thread_i in range(1, thread_num + 1):
        process_pool.append(Thread(target=login_worldquant_brain, args=(threads_dict[thread_i]["account"], 
                                                                        threads_dict[thread_i]["thread"], 
                                                                        headless, 
                                                                        log_info,
                                                                        True,
                                                                        thread_i,
                                                                        engines)))
    for p in process_pool:
        p.start()
    for p in process_pool:
        p.join()

    while True:
        # Check engines
        rework_engine_nums = []
        for thread_i, engine in engines.items():
            if not engine:
                rework_engine_nums.append(thread_i)

        # Rework engines with exceptions
        if not rework_engine_nums:
            print("Successfully create %s engines ! Enjoy !" % thread_num)
            return engines
        else:
            process_pool = []
            for thread_i in rework_engine_nums:
                print("Try to restart engine - %s ." % thread_i)
                process_pool.append(Thread(target=login_worldquant_brain, args=(threads_dict[thread_i]["account"], 
                                                                                 threads_dict[thread_i]["thread"], 
                                                                                 headless, 
                                                                                 log_info,
                                                                                 True,
                                                                                 thread_i,
                                                                                 engines)))
            for p in process_pool:
                p.start()
            for p in process_pool:
                p.join()



def apply_setting(driver, region, universe, delay, neutralization, decay, truncation, setting_remain = False):
    if setting_remain:
        return
    
    # Define CSS Selector dictionaries.
    region_dict = {"USA": 'li.select-portal__item:nth-child(1)',
                   "CHN": 'li.select-portal__item:nth-child(2)'}
    
    if region == "CHN":
        universe_dict = {3000: 'li.select-portal__item:nth-child(1)',
                         2000: 'li.select-portal__item:nth-child(2)'}
        
        delay_dict = {0: 'li.select-portal__item:nth-child(1)', 
                      1: 'li.select-portal__item:nth-child(2)'}
    else:
        universe_dict = {3000: 'li.select-portal__item:nth-child(1)',
                         1000: 'li.select-portal__item:nth-child(2)',
                         500: 'li.select-portal__item:nth-child(3)',
                         200: 'li.select-portal__item:nth-child(4)'}
        
        delay_dict = {1: 'li.select-portal__item:nth-child(1)', 
                      0: 'li.select-portal__item:nth-child(2)'}
    
    neutralization_dict = {"None": 'li.select-portal__item:nth-child(1)',
                          "Market": 'li.select-portal__item:nth-child(2)',
                          "Sector": 'li.select-portal__item:nth-child(3)',
                          "Industry": 'li.select-portal__item:nth-child(4)',
                          "Subindustry": 'li.select-portal__item:nth-child(5)'}
    
    # Get the locations of the options
    try:
        universe_btn = driver.find_element(by = By.XPATH, value = '//*[@id="universe"]')
    except:
        setting_btn = driver.find_element(by = By.XPATH, value = '/html/body/div[1]/div/div[2]/div[3]/div[2]/div/div[1]/div/div[1]/div[2]/div[1]/button')
        setting_btn.click()
        universe_btn = driver.find_element(by = By.XPATH, value = '//*[@id="universe"]')
    region_btn = driver.find_element(by = By.XPATH, value = '//*[@id="region"]')
    delay_btn = driver.find_element(by = By.XPATH, value = '//*[@id="delay"]')
    neutralization_btn = driver.find_element(by = By.XPATH, value = '//*[@id="neutralization"]')
    decay_input = driver.find_element(by = By.CSS_SELECTOR, value = 'div.four:nth-child(7) > div:nth-child(1) > div:nth-child(2) > input:nth-child(1)')
    truncation_input = driver.find_element(by = By.CSS_SELECTOR, value = 'div.four:nth-child(8) > div:nth-child(1) > div:nth-child(2) > input:nth-child(1)')

    # Adjust settings
    region_btn.click()
    driver.find_element(by = By.CSS_SELECTOR, value = region_dict[region]).click()
    universe_btn.click()
    driver.find_element(by = By.CSS_SELECTOR, value = universe_dict[universe]).click()
    delay_btn.click()
    driver.find_element(by = By.CSS_SELECTOR, value = delay_dict[delay]).click()
    neutralization_btn.click()
    driver.find_element(by = By.CSS_SELECTOR, value = neutralization_dict[neutralization]).click()
    decay_input.clear()
    decay_input.send_keys(decay)
    truncation_input.clear()
    truncation_input.send_keys(truncation)
    
    # Apply
    apply_btn = driver.find_element(by = By.CSS_SELECTOR, value = '.button--lg')
    apply_btn.click()



def execute_unit(driver, 
                 i, 
                 settings, 
                 alpha_code, 
                 results_df, 
                 log_address, 
                 output_suf,
                 gen,
                 negative_adj=True,
                 write_in_attr="Score",
                 score_func=lambda f,s:0.45*f+0.55*s,
                 each_sleep=0, 
                 get_corr=False, 
                 setting_remain=False, 
                 alpha_remain=False,
                 log_hide_setting=False,
                 name_dict={".csv": "raw_fitness", ".txt": "log_info"}
                 ):

    def exception_exit(handle_exception=True,
                       index=i,
                       log_info="", 
                       write_in_value=None, 
                       write_in_attr=write_in_attr,
                       results_df=results_df, 
                       log_address=log_address, 
                       output_suf=output_suf
                       ):
        if handle_exception:
            if log_info == "" or type(write_in_value) == type(None):
                raise AttributeError("Please reset 'log_info' or 'write_in_value' in exception_exit() function !")
            with open(log_address + name_dict[".txt"] + "_" + output_suf + ".txt", "a") as f:
                f.write(log_info + "\n")
            results_df.loc[index, write_in_attr] = str(write_in_value)
        results_df.to_csv(log_address + name_dict[".csv"] + "_" + output_suf + ".csv")
        return float(results_df.loc[index, write_in_attr]) 

    # Exception check-list:
    """
    4.03 : Warning persists (30 times). Exit.
    4.04 : Vital Warning Exit.
    4.05 : Execution Time Out. Exit.
    4.06 : Input Error 2 times. Exit.
    4.07 : Setting Input Error 5 times. Exit.            (Rare)
    4.08 : Read Backtest Infomation Error 3 times. Exit. (Rare)
    4.09 : Read Failure Infomation Error 3 times. Exit.  (Rare)
    """

    # Check settings
    if len(settings) != 6:
        raise AttributeError("Please check the length of 'settings' and reset it !")
    region, universe, delay, neutralization, decay, truncation = settings
    if region not in ["USA", "CHN"]:
        raise AttributeError("Please check 'region' in 'settings' and reset it !")
    if region == "CHN":
        if universe not in [2000, 3000]:
            raise AttributeError("Please check 'universe' in 'settings' and reset it !")
    else:
        if universe not in [200, 500, 1000, 3000]:
            raise AttributeError("Please check 'universe' in 'settings' and reset it !")
    if delay not in [0, 1]:
        raise AttributeError("Please check 'delay' in 'settings' and reset it !")
    if neutralization not in ["None", "Market", "Sector", "Industry", "Subindustry"]:
        raise AttributeError("Please check 'neutralization' in 'settings' and reset it !")
    if type(decay) != int or decay < 0 or decay >= 50:
        raise AttributeError("Please check 'decay' in 'settings' and reset it !")
    if type(truncation) != float or truncation < 0 or truncation > 0.1:
        raise AttributeError("Please check 'truncation' in 'settings' and reset it !")

    results_df.loc[i, "Generation"] = gen

    # Apply settings block
    setting_error_count = 0
    while True:
        try:
            apply_setting(driver, region, universe, delay, neutralization, decay, truncation, setting_remain)
            break
        except:
            # It is almost impossible to get here, only to be safe 
            setting_error_count += 1
            if setting_error_count == 5:
                return exception_exit(write_in_value = -4.07, log_info = "Setting Input Error 5 times. Exit.")

    # Input Alpha
    if not log_hide_setting:
        with open(log_address + name_dict[".txt"] + "_" + output_suf + ".txt", "a") as f:
            f.write("Setting: Region: " + region + "; Delay: " + str(delay) + "; Universe: " + str(universe) + "; Neutralization: " 
                    + neutralization + "; Decay: " + str(decay) + "; Truncation: " + str(truncation) + "\n")

    with open(log_address + name_dict[".txt"] + "_" + output_suf + ".txt", "a") as f:
        f.write("No.%d\n" % (i))
        f.write("*Alpha*:\n" + alpha_code + "\n")

    if (i == 1) or (not alpha_remain and (i != 1)):

        input_error_count = 0
        while True:
            try:
                code_input = WebDriverWait(driver, 2, 0.5).until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div.view-line:nth-child(1)')))
                code_input.click()
                ActionChains(driver).key_down(Keys.CONTROL).send_keys('a').key_up(Keys.CONTROL).perform()
                ActionChains(driver).send_keys(alpha_code).perform()
                break
                """[Old version (is not effective)]
                ActionChains(driver).key_down(Keys.CONTROL).send_keys('a').key_up(Keys.CONTROL).perform()
                ActionChains(driver).key_down(Keys.CONTROL).send_keys('c').key_up(Keys.CONTROL).perform()
                input_code = pd.read_clipboard()
                suffix = "".join(input_code.fillna("_").values.flatten())
                input_code = "".join(pd.read_clipboard().columns) + suffix
                for j in range(input_code.count(")") - input_code.count("(")):
                    ActionChains(driver).send_keys(Keys.END).send_keys(Keys.BACKSPACE).perform()
                """
            except:
                input_error_count += 1
                if input_error_count == 2:
                    return exception_exit(write_in_value = -4.06, log_info = "Input Error 2 times. Exit.")

    # Click Simulate Button
    simulate_btn = driver.find_element(by = By.XPATH, value = '/html/body/div[1]/div/div[2]/div[3]/div[2]/div/div[1]/div/div[3]/div[2]/div/button/span')
    simulate_btn.click()
    
    # Waiting for computing
    computing = [True, 0]
    warning_count = 0
    start_time = time.time()
    while computing[0]:
        #try:
        #    warning1_text = driver.find_element(By.CSS_SELECTOR, value='.flash-messages__container').text
        #except:
        #    pass
        #else:
        #    print("catch abnormal message1")
        #    if 'Unexpected character' in warning1_text:
        #        ActionChains(driver).send_keys(Keys.END).send_keys(Keys.BACKSPACE).perform()
        #        simulate_btn.click()
        
        try:
            warning = driver.find_element(By.CSS_SELECTOR, 
                    value = '.editor-code__flash-message > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > ul:nth-child(1) > li:nth-child(1) > div:nth-child(1)')
        except:
            pass
        else:
            warning_set = (
                'There was an error while running the simulation. Please try again or contact BRAIN support if this problem persists.',
                'Your simulation probably took too much resource.',
                'WorldQuant BRAIN is experiencing some difficulties. Please contact support if this problem persists.',
                'There was an error while running the simulation. Please try again or contact BRAIN support if this problem persists.'
                           )

            warning_count += 1
            if warning_count == 1:
                with open(log_address + name_dict[".txt"] + "_" + output_suf + ".txt", "a") as f:
                    f.write("*Warning*:\n" + warning.text + "\n") 

            if warning.text in warning_set or 'Unexpected character' in warning.text:
                return exception_exit(write_in_value = -4.04, log_info = "Vital Warning Exit.")

            if warning_count == 30:
                return exception_exit(write_in_value = -4.03, log_info = "Warning persists (30 times). Exit.")
            
            """[Old version (is not effective)]
            if 'Unexpected character' in warning.text:
                ActionChains(driver).send_keys(Keys.END).send_keys(Keys.BACKSPACE).perform()
                simulate_btn.click()
            """
        try:     
            temp_Returns = driver.find_element(by = By.CSS_SELECTOR, 
                    value = 'div.summary-metrics-info:nth-child(5) > div:nth-child(2)').text
            results_df.loc[i, "Returns"] = temp_Returns
            temp_Turnover = driver.find_element(by = By.CSS_SELECTOR, 
                    value = 'div.summary-metrics-info:nth-child(3) > div:nth-child(2)').text
            results_df.loc[i, "Turnover"] = temp_Turnover
            temp_Drawdown = driver.find_element(by = By.CSS_SELECTOR, 
                    value = 'div.summary-metrics-info:nth-child(6) > div:nth-child(2)').text
            results_df.loc[i, "Drawdown"] = temp_Drawdown

            end_time = time.time()
            if temp_Returns == '0.00%' and temp_Drawdown == '0.00%' and temp_Turnover == '0.00%':
                read_result_error_count = 0
                while True:
                    try:
                        results_df.loc[i, "Sharpe"] = driver.find_element(by = By.CSS_SELECTOR, 
                                value = '#alphas-summary > div:nth-child(1) > div:nth-child(2) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(2) > div:nth-child(2)').text
                        results_df.loc[i, "Exec_time"] = "%.2fs" % (end_time - start_time)
                        return exception_exit(write_in_value = 0, log_info = "Did not take any action.")
                    except:
                        read_result_error_count += 1
                        if read_result_error_count == 3:
                            return exception_exit(write_in_value = -4.08, log_info = "Read Backtest Infomation Error 3 times. Exit.")
        except:
            computing[1] += 1
            time.sleep(2)
        else:
            if (end_time - start_time) < 2:
                driver.find_element(By.CSS_SELECTOR, 'div.view-line:nth-child(1)').click()
                ActionChains(driver).key_down(Keys.CONTROL).send_keys('a').key_up(Keys.CONTROL).perform()
                ActionChains(driver).send_keys(Keys.END).send_keys(' * 1').perform()
                simulate_btn.click()
            else:
                computing[0] = False
        
        if computing [1] == 30:
            with open(log_address + name_dict[".txt"] + "_" + output_suf + ".txt", "a") as f:
                f.write("Trying to Fix Stuck Program" + "\n")
            driver.find_element(By.CSS_SELECTOR, 'div.view-line:nth-child(1)').click()
            ActionChains(driver).key_down(Keys.CONTROL).send_keys('a').key_up(Keys.CONTROL).perform()
            ActionChains(driver).send_keys(Keys.END).send_keys(' * 1').perform()
            simulate_btn.click()

        """[Old version (is not effective)]
            try:
                level = driver.find_element(by = By.XPATH, 
                    value = '/html/body/div[1]/div/div[2]/div[3]/div[2]/div/div[2]/div/div/div[1]/div[1]/div/div[2]/div/div[2]/div/div[1]/div[2]/div').text
            except:
                pass
            else:
                driver.find_element(By.CSS_SELECTOR, 'div.view-line:nth-child(1)').click()
                ActionChains(driver).key_down(Keys.CONTROL).send_keys('a').key_up(Keys.CONTROL).perform()
                ActionChains(driver).send_keys(Keys.END).send_keys(' * 1').perform()
                simulate_btn.click()
        """
        if computing[1] == 55:
            return exception_exit(write_in_value = -4.05, log_info = "Execution Time Out. Exit.")

        """[Old version (is not effective)]
        if computing[1] >= 50:

            try:
                interupt = driver.find_element(by = By.CSS_SELECTOR, 
                    value = '.editor-code__flash-message > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > ul:nth-child(1) > li:nth-child(1) > div:nth-child(1)')
            except:
                pass
            else:
                simulate_btn.click()
        """

        if computing[1] == 60:
            raise Exception("Execution Time Out!")

    # read backtest results block
    read_result_error_count = 0
    while True:
        try:
            results_df.loc[i, "Level"] = driver.find_element(by = By.XPATH, 
                value = '/html/body/div[1]/div/div[2]/div[3]/div[2]/div/div[2]/div/div/div[1]/div[1]/div/div[2]/div/div[2]/div/div[1]/div[2]/div').text
            temp_Fitness = driver.find_element(by = By.CSS_SELECTOR, 
                value = 'div.summary-metrics-info:nth-child(4) > div:nth-child(2)').text
            temp_Sharpe = driver.find_element(by = By.CSS_SELECTOR, 
                    value = '#alphas-summary > div:nth-child(1) > div:nth-child(2) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(2) > div:nth-child(2)').text
            if temp_Fitness[0] == '-' and temp_Sharpe[0] == '-' and negative_adj == True:
                results_df.loc[i, "Fitness"] = temp_Fitness[1:]
                results_df.loc[i, "Sharpe"] = temp_Sharpe[1:]
                results_df.loc[i, "Score"] = "%.4f" % score_func(float(temp_Fitness[1:]), float(temp_Sharpe[1:]))
            else:
                results_df.loc[i, "Fitness"] = temp_Fitness
                results_df.loc[i, "Sharpe"] = temp_Sharpe
                results_df.loc[i, "Score"] = "%.4f" % score_func(float(temp_Fitness), float(temp_Sharpe))
            results_df.loc[i, "Exec_time"] = "%.2fs" % (end_time - start_time)
            break
        except:
            read_result_error_count += 1
            if read_result_error_count == 3:
                return exception_exit(write_in_value = -4.08, log_info = "Read Backtest Infomation Error 3 times. Exit.")

    # read failure infomation block
    read_fail_error_count = 0
    while True:
        try:
            failnum_btn = driver.find_element(by = By.CSS_SELECTOR, value = '.sumary__testing-checks-FAIL-title')
        except:
            results_df.loc[i, "Fail_num"] = 0
            with open(log_address + name_dict[".txt"] + "_" + output_suf + ".txt", "a") as f:
                f.write("No Error Info."+ "\n")
            break    
        else:
            try:
                results_df.loc[i, "Fail_num"] = int(failnum_btn.text[:-5])
                failnum_btn.click()
                for k in range(int(results_df.loc[i, "Fail_num"])):
                    error_info = driver.find_element(by = By.CSS_SELECTOR, 
                            value = '.sumary__testing-checks-FAIL-list > li:nth-child(%d) > div:nth-child(1) > div:nth-child(1)' % (k + 1)).text
                    if k == 0:
                        with open(log_address + name_dict[".txt"] + "_" + output_suf + ".txt", "a") as f:
                            f.write("*Failure Infomation*:" + "\n")
                    with open(log_address + name_dict[".txt"] + "_" + output_suf + ".txt", "a") as f:
                        f.write(error_info + "\n")
                break
            except:
                read_fail_error_count += 1
                if read_fail_error_count == 3:
                     return exception_exit(write_in_value = -4.09, log_info = "Read Failure Infomation Error 3 times. Exit.")
    
    # get correlation block
    if get_corr:
        corr_btn = driver.find_element(By.XPATH, 
               value='/html/body/div[1]/div/div[2]/div[3]/div[2]/div/div[2]/div/div/div[1]/div[1]/div/div[2]/div/div[3]/div[2]/div/div[1]/div[3]/div[3]')
        corr_btn.click()
        # Waiting for correlation coefficient computing
        corr_computing = [True, 0]
        while corr_computing[0]:
            #try:
            #    warning1_text = driver.find_element(By.CSS_SELECTOR, value='').text
            #except:
            #    pass
            #else:
            #    if 'Unexpected character' in warning1_text:
            #        ActionChains(driver).send_keys(Keys.END).send_keys(Keys.BACKSPACE).perform()
            #        simulate_btn.click()

            try:
                corr_score = float(driver.find_element(By.CSS_SELECTOR, value = '.correlation__content-status-higher-value').text)
            except:
                corr_computing[1] += 1
                time.sleep(2)
            else:
                corr_computing[0] = False
                results_df.loc[i, "Corr"] = corr_score

            if corr_computing[1] == 90:
                raise Exception("Find Corr Time Out!")
    
    # sleep block
    if each_sleep > 0:
        time.sleep(each_sleep)
    
    return exception_exit(handle_exception=False)



def aggregate_threads(thread_num, log_address, if_sort=True, name_dict={".csv": "raw_fitness", ".txt": "log_info"}):

    def gen_file_name(thread_num, file_type, name_dict = name_dict):
        if file_type not in [".csv", ".txt"]:
            raise AttributeError("Please reset 'file_type' as '.csv' or '.txt'. ")
        return name_dict[file_type] + "_" + str(thread_num) + file_type

    results_df = pd.DataFrame([])
     # .csv part
    for i in range(1, thread_num + 1):       
        temp = pd.read_csv(log_address + gen_file_name(i, ".csv"), index_col="Unnamed: 0")
        temp["Thread_num"] = i
        temp["Task_num"] = temp.index
        results_df = pd.concat([results_df, temp], axis = 0)
        results_df = results_df[(results_df.isna().sum(axis=1) < results_df.shape[1] - 3)]
    results_df = results_df.reset_index(drop=True)
    results_df["Alpha"] = None

    # .txt part
    for i in range(1, thread_num + 1):
        with open(log_address + name_dict[".txt"] + ".txt", "a") as f:
            f.write("*********************" + "\n") 
            f.write("Thread Number - " + str(i) + "\n")
            f.write("*********************" + "\n")
            alpha_signal = [False, -1]
            for line in open(log_address + gen_file_name(i, ".txt"), encoding="utf-8"):
                if line[:3] == "No.":
                    alpha_signal[1] = int(line.replace("\n", "")[3:])
                if alpha_signal[0]:
                    alpha = line.replace("\n", "")
                    results_df.loc[results_df[(results_df["Thread_num"] == i) & (results_df["Task_num"] == alpha_signal[1])].index[0], "Alpha"] = alpha
                    alpha_signal[0] = False
                f.writelines(line)
                if line == "*Alpha*:\n":
                    alpha_signal[0] = True
            f.write("\n")

    if if_sort:
        results_df = results_df.sort_values(by=['Score', 'Fail_num', 'Thread_num'], ascending=[False, True, True])
    results_df.to_csv(log_address + name_dict[".csv"] + ".csv")



def aggregate_process(log_address, if_sort=True, name_dict={".csv": "raw_fitness", ".txt": "log_info"}):
    results_df = pd.DataFrame([])
    for s in os.listdir(log_address):
        if '.npy' not in s and '.txt' not in s and name_dict[".csv"] not in s:
            temp = pd.read_csv(log_address + s + "/" + name_dict[".csv"] + ".csv", index_col="Unnamed: 0")
            temp["From_process"] = s
            results_df = pd.concat([results_df, temp], axis = 0)

    if if_sort:
        results_df = results_df.sort_values(by=['Score', 'Fail_num', 'From_process'], ascending=[False, True, True])
    results_df = results_df.reset_index(drop=True)
    results_df.to_csv(log_address + name_dict[".csv"] + ".csv")
    
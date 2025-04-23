from selenium import webdriver
import time
import pyautogui
import pandas as pd
import csv
import os

from PIL import Image
mainstring=["Marek’s Disease","Egg peritonitis and salpingitis","Neoplasm","Colisepticcemia","Coccidiosis "]
snipped=1

def writeincsv(string1,string2,string3,string4):
    with open("DataSet.csv", "a", newline="") as csvfile:
    
        writer = csv.writer(csvfile)
        writer.writerow([string1,string2,string3,string4])  # Each word becomes one column
    
     
def function000(maincategoryindex,mainstringx):
    writeincsv(mainstringx,"all","all","all")
    snip()

def function001(maincategoryindex,mainstringx):
     age(maincategoryindex)
     writeincsv(mainstringx,"adult","all","all")
     writeincsv(mainstringx,"other","all","all")
    
     

def function010(maincategoryindex,mainstringx):
     Year(maincategoryindex)
     for i in range (5):
         writeincsv(mainstringx,all,str(2020+i),all)


def function100(maincategoryindex,mainstringx):
    Region(maincategoryindex)
    regionstring1=["London","Scotland","Wales"]
    regionstring2=["Scotland","Wales"]
    if maincategoryindex !=3:
        regionstring=regionstring1
    else:
        regionstring=regionstring2
    for i in range (len(regionstring)):
         
         writeincsv(mainstringx,"all","all",regionstring[i])
         
     
def snip():
    global snipped
    x1=810
    y1=210
    x2=1293
    y2=428
    width = x2 - x1
    height = y2 - y1
    output_file=str(snipped)+".png"
    output_file=os.path.join("SnipFolder", output_file)
    time.sleep(2)
    screenshot = pyautogui.screenshot(region=(x1, y1, width, height))
    screenshot.save(output_file)
    print(f"✅ Snipped region saved to {output_file}")
    snipped=snipped+1



def ClickCordinate(df):
    driver.execute_script("window.scrollTo(0,400);")#same position here 
    time.sleep(3)
    row = df
    print("row:")
    print(row)
    category= str(row.iloc[0])     # First column (by position)
    x = int(row.iloc[1])     # Second column (by position)
    y = int(row.iloc[2])
    pyautogui.moveTo(x, y)
    time.sleep(1.5)
    pyautogui.click()
    time.sleep(1)
    print(f"cliked on corrdinate ({x},{y})")

    



def age(maincategoryindex):
   
    if maincategoryindex >2:
        df = pd.read_csv('age3.csv')
    elif maincategoryindex==1:
        df=pd.read_csv('age2.csv')

    else:
        df=pd.read_csv('age1.csv')
    num = len(df)
    #loop for clicking adult
    for i in range(num):
        row=df.iloc[i]
        ClickCordinate(row)
    snip()
    
    for i in range(1,num):
        row=df.iloc[i]
        ClickCordinate(row)
    snip()
    ClickCordinate(df.iloc[1])
    ClickCordinate(df.iloc[num-1])
    ClickCordinate(df.iloc[0])
    


def Year(maincategoryindex):
    if maincategoryindex==0 or maincategoryindex==2:
        df=pd.read_csv('Years1.csv')
    else:
        df=pd.read_csv('Years2.csv')
    num= len(df)
    row=df.iloc[0]
    ClickCordinate(row)
    for i in range (2,num-1) :
        
        ClickCordinate(df.iloc[1])
        ClickCordinate(df.iloc[i])
        ClickCordinate(df.iloc[num-1])
        snip()
        ClickCordinate(df.iloc[1])
    
    ClickCordinate(df.iloc[num-1])
    ClickCordinate(row)
    
         
         
def Region(maincategoryindex):
    print(f"Maincateryindex:{maincategoryindex}")
    if maincategoryindex==3:
        df = pd.read_csv('Region2.csv')
    else:
        df=pd.read_csv('Region1.csv')
    num = len(df)
    print("dataframe:")
    print(df)
    row=df.iloc[0]
    ClickCordinate(row)
    for i in range (2,num-1) :
        
        ClickCordinate(df.iloc[1])
        ClickCordinate(df.iloc[i])
        ClickCordinate(df.iloc[num-1])
        print(f"This is region of {df.iloc[i].iloc[0]}")
        snip()
        print(f"{df.iloc[i].iloc[0]} snipped")
        ClickCordinate(df.iloc[1])

    ClickCordinate(df.iloc[num-1])
    ClickCordinate(row)



# Launch browser
driver = webdriver.Chrome()

# OPTIONAL: Position browser to known location (e.g., top-left)
driver.set_window_position(0, 0)
driver.maximize_window()
 # Adjust to match your screen resolution

# Load the page
driver.get("https://public.tableau.com/app/profile/siu.apha/viz/AvianDashboard/AvianDashboard")
time.sleep(7)  # Wait for the page to fully load

# Scroll the page if needed
driver.execute_script("window.scrollBy(0,400);")
time.sleep(3)
print("Scrolled")

# ====== Screen-based Click Coordinates ======
# These coordinates are relative to your screen
# Use Cmd+Shift+4 to measure
x_screen = 870
y_screen = 672

# Move and click on screen
for maincategoryindex in range (5):
    print(f"Clicking on screen at ({x_screen}, {y_screen})...")
    pyautogui.moveTo(x_screen, y_screen)
    pyautogui.click()
    print("Clicked!")
    driver.execute_script("window.scrollTo(0,400);")#same position here
    time.sleep(2)
    function000(maincategoryindex,mainstring[maincategoryindex]) 
    print("running function000")
    function001(maincategoryindex,mainstring[maincategoryindex])
    time.sleep(2)
    print("running function001")
    function010(maincategoryindex,mainstring[maincategoryindex])
    time.sleep(2)
    print("running function010")
    function100(maincategoryindex,mainstring[maincategoryindex])
    time.sleep(2)
    print("running function100")
    
    y_screen=y_screen+21

# Close browser
driver.quit()



from asyncio.windows_events import NULL
from ftplib import FTP
from ftplib import error_perm
import requests
import json
import time
from datetime import datetime
counter=0
"""This part is for upload to NAS """
class upload():
    def __init__(self):
        self.uploadtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        self.y = time.strftime('%Y', time.localtime(time.time()))
        self.m = time.strftime('%m', time.localtime(time.time()))
        self.d = time.strftime('%d', time.localtime(time.time()))
        self.HM = time.strftime('%H%M', time.localtime(time.time()))



"""This part is for upload to IBP"""
#上傳數字要包含上傳資料資訊以及網址，其他可以空白，若上傳圖片就要網址以及檔案，其他可以空白
#IBP upload information setting:
#IBP necessary information is upload type / time /upload number or picture
#1.Please input the type that you want to upload to IBP ,there is 2 options: "number" / "picture"
#2.Input the time information ,you can use time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())) to input , use upload class
#   Which is :
#       a.  OOO = upload() //setting class
#       b.  IBPuploader(datatype,  OOO.uploadtime  ,url,info)
#       or just leave it blank ,this program will auto input time 
#3.url is very import information to input, you can check url information in IBP 感測模組->("顯示上傳資訊"). Make sure address is correct.
#4.The info is the information that you want to upload to IBP,If you want to upload picture,you can leave it blank.
#   otherwise you need following the step below(上傳數字):
"""
{
  "資料代碼1": (你要上傳的數字)
  "資料代碼2": (你要上傳的數字)
  "資料代碼3": (你要上傳的數字)
  "dataTime": OOO.uploadtime ,
  "_overwrite_": true,
  "_token_": "XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXX"
}
"""
#5.In file input section,you can just input open("檔案名稱","rb") or input {'file': open("檔案名稱","rb")} which is also O.K.
#6.counter: this is the repeat counter ,if upload is fallure. 
def IBPuploader(datatype=NULL,timepoint=NULL,url=NULL,info=NULL,file=NULL,counter=5):
    """
    datatype: number , picture
    url : upload url target 
    """
    if(datatype==NULL):
        print("upload type is invalid, now into number upload mode.")
        return IBPuploader("number",timepoint=timepoint,url=url,info=info,file=file,counter=counter)
    if(timepoint==NULL):
        timenow = upload()
        print("upload time is invalid, now set upload time to :"+timenow.uploadtime)
        return IBPuploader(datatype=datatype,timepoint=timenow.uploadtime,url=url,info=info,file=file,counter=counter)

    else:
        headers = {"Content-Type":"application/json"}
        query_j = json.dumps(info)
        
        #Number upload area
        if datatype=="number" :
            if(info==NULL):
                print("please import IBP web address")
            if(info!=NULL):
                try:
                    res = requests.post(url,data=query_j,headers=headers,timeout=20)
                    print("successful upload number data to IBP")
                except:  
                    print("Failed to upload number to IBP.")
                    time.sleep(5)
                    if(counter>0):
                        counter-=1
                        return IBPuploader(datatype,timepoint=timepoint,url=url,info=info,file=file,counter=counter)
                    else:
                        print("Failed to upload number to IBP.")
            else:
                print("IBP upload url is invalid.")
        #Picture upload area
        if datatype=="picture" :
            if(url==NULL):
                return IBPuploader(datatype=datatype,timepoint=timepoint,url=url,info={'dataTime' : timepoint},file=file,counter=counter)
            if(url!=NULL):
                if isinstance(file,dict) is False:
                    return IBPuploader(datatype=datatype,timepoint=timepoint,url=url,info=info,file={'file': file},counter=counter)
                try:
                    res = requests.post(url,data=info,files=file)     
                    print("successful upload to IBP")
                    
                except:
                    
                    print("Failed to upload to IBP")
                    time.sleep(5)
                    if(counter>0):
                        counter-=1
                        return IBPuploader(datatype=datatype,timepoint=timepoint,url=url,info=info,file=file,counter=counter)
                    else:
                        print("Number data Upload to IBP failed.")
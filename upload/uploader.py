from asyncio.windows_events import NULL
from ftplib import FTP
from ftplib import error_perm
import requests
import json
import time
import os
#範例:上傳NAS
#uploader.NASuploader(address="01_Personal Folders/B61_朱濬謙",file="icecream.jpg")
#此處須注意address的"/"是向右的。
"""This part is for upload to NAS """
class upload():
    def __init__(self):
        self.uploadtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        self.y = time.strftime('%Y', time.localtime(time.time()))
        self.m = time.strftime('%m', time.localtime(time.time()))
        self.d = time.strftime('%d', time.localtime(time.time()))
        self.HM = time.strftime('%H%M', time.localtime(time.time()))


def NASuploader(address,file,counter=0):
    try:
        ftp = FTP("140.120.101.117")
        ftp.login('PMML', 'enjoyresearch')
        #ftp.retrlines('LIST')
        ftp.cwd(address)
        ftp.retrlines('LIST')
        ftp.storbinary('STOR %s' % os.path.basename(file),open(file, 'rb'))
    except:
        if(counter<5):  
            counter+=1  
            time.sleep(5)
            return NASuploader(address=address,file=file,counter=counter)
        else:
            print("upload to NAS failed.")



{
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

"""範例:上傳IBP 數字"""
#自訂時間
# cat = uploader.upload()
# infor = {"Activity": 4,"dataTime": cat.uploadtime,"_overwrite_": True, "_token_": "340f541d-009f-4d8c-be2c-0b0e38ee7842"}
# files = open("icecream.jpg", 'rb')
# uploader.IBPuploader(timepoint=cat.uploadtime,url="http://ibp.bime.ntu.edu.tw/rest/sensorDataLogs/NCHUBIME/PMML/05",info=infor)

"""範例:上傳IBP 圖片"""
#自訂時間、自己開啟
#cat = uploader.upload()
#files = open("icecream.jpg", 'rb')
#uploader.IBPuploader(datatype="picture",timepoint=cat.uploadtime,url="http://ibp.bime.ntu.edu.tw/rest/sensorDataLogs/NCHUBIME/PMML/05/pic/file",file=files)
#___________________________________________________________________________________________________________________________________________________________
#系統自己設定時間、系統開啟
#cat = uploader.upload()
#uploader.IBPuploader(datatype="picture",url="http://ibp.bime.ntu.edu.tw/rest/sensorDataLogs/NCHUBIME/PMML/05/pic/file",file="icecream.jpg")
}
def IBPuploader(datatype=NULL,timepoint=NULL,url=NULL,info=NULL,file=NULL,counter=5):
    """
    datatype: number , picture
    url : upload url target 
    """
    if(file==NULL):
        print("upload file is NULL, now into number upload mode.")
        return IBPuploader("number",timepoint=timepoint,url=url,info=info,file=file,counter=counter)
    if(file!=NULL):
        print("upload file , now into picture upload mode.")
        return IBPuploader("picture",timepoint=timepoint,url=url,info=info,file=file,counter=counter)
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
                    requests.post(url,data=query_j,headers=headers,timeout=20)
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
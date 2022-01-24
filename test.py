from nestedsegment import nested
from upload import uploader
cat = uploader.upload()
files = open("icecream.jpg", 'rb')

uploader.IBPuploader(datatype="picture",url="http://ibp.bime.ntu.edu.tw/rest/sensorDataLogs/NCHUBIME/PMML/05/pic/file",file=files)

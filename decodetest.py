# -*- coding: utf-8 -*-  
import numpy as np  
import urllib.request
import cv2  
import matplotlib.pyplot as plt
from io import BytesIO
url = 'http://www.pyimagesearch.com/wp-content/uploads/2015/01/google_logo.png'  
resp = urllib.request.urlopen(url)  
image = np.asarray(bytearray(resp.read()), dtype="uint8")  
image = cv2.imdecode(image, cv2.IMREAD_COLOR)  
flag, encodedImage = cv2.imencode(".png", image)
binimg = np.array(encodedImage).tobytes()


# cv2.imshow('URL2Image',image)  
# cv2.waitKey()





fig = plt.figure()
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
my_stringIObytes = BytesIO()

# save fig without white margin
plt.axis('off')
# 去除图像周围的白边
height, width, channels = image.shape
fig.set_size_inches(width / fig.dpi, height / fig.dpi)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
plt.margins(0, 0)
plt.savefig(my_stringIObytes, dpi=fig.dpi)

my_stringIObytes.seek(0)
plt.close(fig)
outputFrame = my_stringIObytes.read()
print('fin')
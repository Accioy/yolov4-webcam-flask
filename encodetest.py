import numpy as np  
import cv2  
import matplotlib.pyplot as plt
from io import BytesIO

# img = np.random.sample([100,150,3])
image = cv2.imread('demo.png')
 
flag, encodedImage = cv2.imencode(".jpg", image)
# binimg = np.array(encodedImage).tobytes()
io_buf = BytesIO(encodedImage)
io_buf.seek(0)
binimg = io_buf.read()

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
binimg1 = my_stringIObytes.read()


print('fin')


image = np.asarray(bytearray(binimg), dtype="uint8") 
image = cv2.imdecode(image, cv2.IMREAD_COLOR)  
cv2.imwrite('cv.png',image)
image = np.asarray(bytearray(binimg1), dtype="uint8") 
image = cv2.imdecode(image, cv2.IMREAD_COLOR)  
cv2.imwrite('cv1.png',image)
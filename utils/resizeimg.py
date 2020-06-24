import PIL.Image as Image
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import os

# org_img_path = r'C:\Users\cswlu\Desktop\Red_Bellied_Woodpecker_0068_180949.jpg'
imgname = "false-plane.jpg"
org_img_path = os.path.join("/home/luowei/Downloads", imgname)
img = Image.open(org_img_path)

fig, ax = plt.subplots()
imgg = img.resize((448,448))
height, width = imgg.size
ax.imshow(imgg)
ax.set_axis_off()
fig.set_size_inches((width/300, height/300))
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
plt.margins(0, 0)

# plt.subplots_adjust(left=0, bottom=0, right=0.1,
#                     top=0.1, wspace=0, hspace=0)

# rsz_img_path = r'C:\Users\cswlu\Desktop\rsz.jpg'
imgname_rz = os.path.splitext(imgname)[0] + ".eps"
rsz_img_path = os.path.join("/home/luowei/Downloads", imgname_rz)
plt.savefig(rsz_img_path)

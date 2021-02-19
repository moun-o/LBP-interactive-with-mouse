import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import os
##events
def mouse_event(event,x,y,flags,param):
	#cv2.imshow('LBP', LBPimg)
	if event==cv2.EVENT_LBUTTONDOWN:
		#plt.clf()
		new_frame = np.copy(LBPimg)
		new_frame90 = np.copy(LBPimg90)
		new_frame45 = np.copy(LBPimg45)

		xb=int(x/16)
		yb=int(y/16)
		debx=xb * 16    + 1
		deby=yb * 16    + 1
		#print(xb,' ',yb)
		cv2.namedWindow('LBP')
		cv2.rectangle(new_frame, (debx, deby), (debx+14, deby+14),(255, 0, 0),2)
		cv2.imshow('LBP', new_frame)
		cv2.namedWindow('LBP90')
		cv2.rectangle(new_frame90, (debx, deby), (debx + 14, deby + 14), (0, 255, 0),2)
		cv2.imshow('LBP90', new_frame90)
		cv2.namedWindow('LBP45')
		cv2.rectangle(new_frame45, (debx, deby), (debx + 14, deby + 14), (0, 255, 255), 2)
		cv2.imshow('LBP45', new_frame45)
		i=0
		for ii in range(deby,deby+15):
			j=0
			for jj in range(debx,debx+15):
				hist_bloc[i][j]=LBPimg[ii][jj]
				hist_bloc90[i][j]=LBPimg90[ii][jj]
				hist_bloc45[i][j]=LBPimg45[ii][jj]

				j=j+1

			i=i+1

		#print(hist_bloc)
		hist_to_draw=frequence(hist_bloc)
		hist_to_draw90=frequence(hist_bloc90)
		hist_to_draw45=frequence(hist_bloc45)

		plt.figure('LBP 0°',figsize=(3,3))
		plt.get_current_fig_manager().window.setGeometry(0, 400, 300, 300)
		plt.clf()
		plt.title('LBP 0°')
		plt.bar(range(256),hist_to_draw,color='blue',width=1) # inform matplotlib of the new data
		plt.draw()


		plt.figure('LBP 90°',figsize=(3,3))
		plt.get_current_fig_manager().window.setGeometry(400, 400, 300, 300)
		plt.clf()
		plt.title('LBP 90°')
		plt.bar(range(256),hist_to_draw90,color='blue',width=1) # inform matplotlib of the new data
		plt.draw()

		plt.figure('LBP 45°',figsize=(3,3))
		plt.get_current_fig_manager().window.setGeometry(800, 400, 300, 300)
		plt.clf()
		plt.title('LBP 45°')
		plt.bar(range(256), hist_to_draw45, color='red', width=1)  # inform matplotlib of the new data
		plt.draw()


		plt.show()
	elif event == cv2.EVENT_RBUTTONDOWN:
		exit(-1)
#LBPimg_edited[yb*16:(yb*16)+16,xb*16:(xb*16)+16]=255
		#del LBPimg_edited


##functions

def pixel(bloc,pix):
	for i in range(0,bloc.size):
		bloc[i] = int(bloc[i]) - pix
		if bloc[i]>=0:
			bloc[i]=0
		else:
			bloc[i]=1

	s=''
	for bit in bloc:
		s=s+str(bit)
	return int(s, 2)

#To compute the histogram
def frequence(img):
	freq=np.zeros(256)
	for line in img:
		for px in line:
			freq[px] += 1
	return freq


# main program

bloc_size=16
# charger l'image
img = cv2.imread("pic.png", 0)
w=img.shape[0]
h=img.shape[1]
#cv2.imshow('mon image', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
LBPimg=np.zeros((h,w), dtype=np.uint8)
LBPimg90=np.zeros((h,w), dtype=np.uint8)
LBPimg45=np.zeros((h,w), dtype=np.uint8)
hist_bloc=np.zeros((bloc_size,bloc_size), dtype=np.uint8)
hist_bloc90=np.zeros((bloc_size,bloc_size), dtype=np.uint8)
hist_bloc45=np.zeros((bloc_size,bloc_size), dtype=np.uint8)
for i in range(1,w-1):
	for j in range(1,h-1):
		#in each LBP angle, the 1st neighbour to consider is the changing one
		voisins =   np.array([img[i - 1][j - 1], img[i - 1][j], img[i - 1][j + 1], img[i + 1, j], img[i + 1][j + 1], img[i + 1][j],img[i + 1][j - 1], img[i, j - 1]], dtype=np.int)
		voisins90 = np.array([img[i - 1][j + 1], img[i][j+1], img[i + 1][j + 1], img[i + 1, j], img[i + 1][j - 1], img[i][j -1],img[i - 1][j - 1], img[i-1, j]], dtype=np.int)
		voisins45 = np.array([img[i - 1][j],img[i - 1][j + 1], img[i][j+1], img[i + 1][j + 1], img[i + 1, j], img[i + 1][j - 1], img[i][j -1],img[i - 1][j - 1]], dtype=np.int)

		LBPimg[i][j]    =   pixel(voisins, img[i][j])
		LBPimg90[i][j]  =   pixel(voisins90, img[i][j])
		LBPimg45[i][j]  =   pixel(voisins45, img[i][j])


#cv2.setMouseCallback('image',mouse_event)
#cv2.imshow('image',img)

cv2.namedWindow('LBP')
cv2.moveWindow("LBP", 0, 10)
cv2.imshow('LBP',LBPimg)

cv2.namedWindow('LBP90')
cv2.moveWindow("LBP90", 400, 10)
cv2.imshow('LBP90',LBPimg90)

cv2.namedWindow('LBP45')
cv2.moveWindow("LBP45", 800, 10)
cv2.imshow('LBP45',LBPimg45)

cv2.setMouseCallback('LBP',mouse_event)
#plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.close()
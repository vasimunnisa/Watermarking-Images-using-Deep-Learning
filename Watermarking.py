from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
import cv2
from math import log10, sqrt
import os
from skimage.metrics import structural_similarity as ssim
import pywt

main = tkinter.Tk()
main.title("On the implementation of a secured watermarking mechanism based on cryptography and bit pairs matching")
main.geometry("1200x1200")

global host, watermark

def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 100 - (20 * log10(max_pixel / sqrt(mse))) 
    return psnr,mse

def imageSSIM(normal, embed):
    #original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    #super_image = cv2.cvtColor(super_image, cv2.COLOR_BGR2GRAY)
    ssim_value = ssim(normal, embed, data_range = embed.max() - embed.min())
    return ssim_value


def uploadHost():
    global host
    text.delete('1.0', END)
    host = filedialog.askopenfilename(initialdir = "hostImages")
    pathlabel.config(text=host+" host image loaded")
    
def uploadWatermark():
    global watermark
    watermark = filedialog.askopenfilename(initialdir = "watermarkImages")
    pathlabel.config(text=watermark+" watermark image loaded")
    
 
def runDWT():
    text.delete('1.0', END)
    global host, watermark
    coverImage = cv2.imread(host,0)
    watermarkImage = cv2.imread(watermark,0)

    coverImage = cv2.resize(coverImage,(300,300))
    cv2.imshow('Cover Image',cv2.resize(coverImage,(400,400)))
    watermarkImage = cv2.resize(watermarkImage,(150,150))
    cv2.imshow('Watermark Image',cv2.resize(watermarkImage,(400,400)))

    coverImage =  np.float32(coverImage)   
    coverImage /= 255;
    coeffC = pywt.dwt2(coverImage, 'haar')
    cA, (cH, cV, cD) = coeffC
    watermarkImage = np.float32(watermarkImage)
    watermarkImage /= 255;

    #Embedding
    coeffW = (0.4*cA + 0.1*watermarkImage, (cH, cV, cD))
    watermarkedImage = pywt.idwt2(coeffW, 'haar')
    psnr,mse = PSNR(coverImage,watermarkedImage)
    ssim = imageSSIM(coverImage,watermarkedImage)
    text.insert(END,"DWT PSNR : "+str(psnr)+"\n")
    text.insert(END,"DWT MSE  : "+str(mse)+"\n")
    text.insert(END,"DWT SSIM : "+str(ssim)+"\n\n")
    text.update_idletasks()
    name = os.path.basename(host)
    cv2.imwrite("OutputImages/"+name,watermarkedImage*255)
    np.save("model/"+name,watermarkedImage)
    np.save("model/CA_"+name,cA)
    cv2.imshow('Watermarked Image', cv2.resize(watermarkedImage,(400,400)))
    cv2.waitKey(0)

def runExtraction():
    wm = filedialog.askopenfilename(initialdir = "OutputImages")
    wm = os.path.basename(wm)
    img = np.load("model/"+wm+".npy")
    cA = np.load("model/CA_"+wm+".npy")
    coeffWM = pywt.dwt2(img, 'haar')
    hA, (hH, hV, hD) = coeffWM

    extracted = (hA-0.4*cA)/0.1
    extracted *= 255
    extracted = np.uint8(extracted)
    extracted = cv2.resize(extracted,(400,400))
    cv2.imshow('Extracted Image', extracted)
    cv2.waitKey(0)

def runSVD():
    coverImage = cv2.imread(host,0)
    watermarkImage = cv2.imread(watermark,0)
    cv2.imshow('Cover Image',cv2.resize(coverImage,(400,400)))
    [m,n]=np.shape(coverImage)
    coverImage=np.double(coverImage)
    cv2.imshow('Watermark Image',cv2.resize(watermarkImage,(400,400)))
    watermarkImage = np.double(watermarkImage)

    #SVD of cover image
    ucvr,wcvr,vtcvr=np.linalg.svd(coverImage,full_matrices=1,compute_uv=1)
    Wcvr=np.zeros((m,n),np.uint8)
    Wcvr[:m,:n]=np.diag(wcvr)
    Wcvr=np.double(Wcvr)
    [x,y] = np.shape(watermarkImage)
    
    #modifying diagonal component
    for i in range(0,x):
      for j in range(0,y):
          Wcvr[i,j]=(Wcvr[i,j]+0.01*watermarkImage[i,j])/255
          
    #SVD of wcvr
    u,w,v=np.linalg.svd(Wcvr,full_matrices=1,compute_uv=1)

    #Watermarked Image
    S=np.zeros((512,512),np.uint8)
    print(str(S.shape)+" "+str(m)+" "+str(n))
    S[:m,:n]=np.diag(w)
    S=np.double(S)
    wimg=np.matmul(ucvr,np.matmul(S,vtcvr))
    wimg=np.double(wimg)
    wimg*=255
    watermarkedImage = np.zeros(wimg.shape,np.double)
    normalized=cv2.normalize(wimg,watermarkedImage,1.0,0.0,cv2.NORM_MINMAX)
    psnr,mse = PSNR(coverImage,watermarkedImage)
    ssim = imageSSIM(coverImage,watermarkedImage)
    text.insert(END,"DWT PSNR : "+str(psnr)+"\n")
    text.insert(END,"DWT MSE  : "+str(mse)+"\n")
    text.insert(END,"DWT SSIM : "+str(ssim)+"\n\n")
    text.update_idletasks()
    watermarkedImage = cv2.resize(watermarkedImage,(400,400))
    cv2.imshow('Watermarked Image',watermarkedImage)
    cv2.waitKey(0)
    

font = ('times', 20, 'bold')
title = Label(main, text='On the implementation of a secured watermarking mechanism based on cryptography and bit pairs Matching')
title.config(bg='brown', fg='white')  
title.config(font=font)           
title.config(height=3, width=80)       
title.place(x=5,y=5)

font1 = ('times', 13, 'bold')

uploadHost = Button(main, text="Upload Host Image", command=uploadHost)
uploadHost.place(x=50,y=100)
uploadHost.config(font=font1)

loadwatermark = Button(main, text="Upload Watermark Image", command=uploadWatermark)
loadwatermark.place(x=50,y=150)
loadwatermark.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=380,y=100)

dwtButton = Button(main, text="Run DWT Watermarking", command=runDWT)
dwtButton.place(x=50,y=200)
dwtButton.config(font=font1)

svdButton = Button(main, text="Run SVD Watermarking", command=runSVD)
svdButton.place(x=50,y=250)
svdButton.config(font=font1)

extractionButton = Button(main, text="Extraction", command=runExtraction)
extractionButton.place(x=50,y=300)
extractionButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=15,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=350)
text.config(font=font1)


main.config(bg='brown')
main.mainloop()

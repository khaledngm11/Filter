from tkinter import *
import tkinter
from PIL import ImageTk,Image,ImageFilter
from tkinter import filedialog
import numpy as np
import cv2
import PIL.Image, PIL.ImageTk
import tkinter.font as font
root=Tk()
root.title("Image Processing")
root.geometry("790x680")
root['background']='#2C313C'
myFont = font.Font(size=13)
def choose():
    global canvas,file_path,image,image_tk
    canvas = Canvas(root, width= 390, height=370)
    canvas.place(x=200,y=20)
    file_path = filedialog.askopenfilename()
    image = Image.open(file_path)
    image_tk = ImageTk.PhotoImage(image)
    image=image.resize((400,380),Image.ANTIALIAS)
    image_tk = ImageTk.PhotoImage(image)
    canvas.create_image(0,0,anchor=NW, image=image_tk)


def negative():
    
    img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
    w,h,_=img.shape
    for r in range (w-1):
        for c in range (h-1):
            pixel=img[r][c]
            pixel=255-pixel
            img[r][c]=pixel
    img=cv2.resize(img,(390,370),interpolation = cv2.INTER_AREA)
    image_tk= PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img))
    canvas.create_image(0,0,anchor=NW, image=image_tk)
    root.mainloop()
def avg():
    img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
    kernel=np.array([[1/9,1/9,1/9],
                     [1/9,1/9,1/9],
                     [1/9,1/9,1/9]])
    img=cv2.filter2D(img,-1,kernel)
    img=cv2.resize(img,(390,370),interpolation = cv2.INTER_AREA)
    image_tk= PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img))
    canvas.create_image(0,0,anchor=NW, image=image_tk)
    root.mainloop()
def blur():
    img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
    img=cv2.blur(img,(7,7))
    img=cv2.resize(img,(390,370),interpolation = cv2.INTER_AREA)
    image_tk= PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img))
    canvas.create_image(0,0,anchor=NW, image=image_tk)
    root.mainloop()
def medianBlur():
    img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
    img=cv2.medianBlur(img,(11))
    img=cv2.resize(img,(390,370),interpolation = cv2.INTER_AREA)
    image_tk= PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img))
    canvas.create_image(0,0,anchor=NW, image=image_tk)
    root.mainloop()
def laplacian():
    img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
    img=cv2.Laplacian(img,cv2.CV_64F)
    img=cv2.resize(img,(390,370),interpolation = cv2.INTER_AREA)
    image_tk= PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img))
    canvas.create_image(0,0,anchor=NW, image=image_tk)
    root.mainloop()
def sobel():
    img =cv2.imread(file_path,1)
    img = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    img=cv2.resize(img,(390,370),interpolation = cv2.INTER_AREA)
    image_tk= PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(np.uint8(img)))
    canvas.create_image(0,0,anchor=NW, image=image_tk)
    root.mainloop()
def gaussian():
    img =cv2.imread(file_path,1)
    img = cv2.GaussianBlur(img,(5,5),0)
    img=cv2.resize(img,(390,370),interpolation = cv2.INTER_AREA)
    image_tk= PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(np.uint8(img)))
    canvas.create_image(0,0,anchor=NW, image=image_tk)
    root.mainloop()
def emboss():
    img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
    kernel=np.array([[-2,-1,0],
                     [-1,1,-1],
                     [0,1,2]])
    img=cv2.filter2D(img,-1,kernel)
    img=cv2.resize(img,(390,370),interpolation = cv2.INTER_AREA)
    image_tk= PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img))
    canvas.create_image(0,0,anchor=NW, image=image_tk)
    root.mainloop()
def BlackAndWhite():
    img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img,(390,370),interpolation = cv2.INTER_AREA)
    image_tk= PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img))
    canvas.create_image(0,0,anchor=NW, image=image_tk)
    root.mainloop()

def Sepia():
    img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
    kernel=np.array([[0.272, 0.534, 0.131],
    			    [0.349, 0.686, 0.168],
    			    [0.393, 0.769, 0.189]])
    img=cv2.filter2D(img,-1,kernel)
    img=cv2.resize(img,(390,370),interpolation = cv2.INTER_AREA)
    image_tk= PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img))
    canvas.create_image(0,0,anchor=NW, image=image_tk)
    root.mainloop()
def Prewitt():
    img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
    kernel = np.array([[1,1,1]
                       ,[0,0,0],
                       [-1,-1,-1]])
    img=cv2.filter2D(img,-1,kernel)
    img=cv2.resize(img,(390,370),interpolation = cv2.INTER_AREA)
    image_tk= PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img))
    canvas.create_image(0,0,anchor=NW, image=image_tk)
    root.mainloop()
def Histogram():
    img = cv2.imread(file_path,0)
    img=cv2.equalizeHist(img)
    img=cv2.resize(img,(390,370),interpolation = cv2.INTER_AREA)
    image_tk= PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img))
    canvas.create_image(0,0,anchor=NW, image=image_tk)
    root.mainloop()
def EdgeDetection():
    img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
    kernel=np.array([[-1, -1,-1],
    			    [-1, 8, -1],
    			    [-1, -1, -1]])
    img=cv2.filter2D(img,-1,kernel)
    img=cv2.resize(img,(390,370),interpolation = cv2.INTER_AREA)
    image_tk= PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img))
    canvas.create_image(0,0,anchor=NW, image=image_tk)
    root.mainloop()
def Sharpen():
    img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
    kernel=np.array([[0, -1,0],
    			    [-1, 5, -1],
    			    [0, -1, 0]])
    img=cv2.filter2D(img,-1,kernel)
    img=cv2.resize(img,(390,370),interpolation = cv2.INTER_AREA)
    image_tk= PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img))
    canvas.create_image(0,0,anchor=NW, image=image_tk)
    root.mainloop()

def contrast():
    img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
    w,h,m=img.shape
    for r in range (w-1):
        for c in range (h-1):
            for n in range(m-1):
                img[r,c,n]=np.clip(2*img[r,c,n]+0.2,0,255)
            
    img=cv2.resize(img,(390,370),interpolation = cv2.INTER_AREA)
    image_tk= PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img))
    canvas.create_image(0,0,anchor=NW, image=image_tk)
    root.mainloop()
def log():
    img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
    w,h,_=img.shape
    c=255/(np.log(1+np.max(img)))
    img=c*np.log(1+img)
    img=np.array(img,dtype=np.uint8)
    img=cv2.resize(img,(390,370),interpolation = cv2.INTER_AREA)
    image_tk= PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img))
    canvas.create_image(0,0,anchor=NW, image=image_tk)
    root.mainloop()
def Minimum():
    img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
    print(type(img))
    img=Image.fromarray(img)
    print(type(img))
    img=img.filter(ImageFilter.MinFilter())
    img=np.array(img,dtype=np.uint8)
    img=cv2.resize(img,(390,370),interpolation = cv2.INTER_AREA)
    image_tk= PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img))
    canvas.create_image(0,0,anchor=NW, image=image_tk)
    root.mainloop()
def Maximum():
    img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
    img=Image.fromarray(img)
    img=img.filter(ImageFilter.MaxFilter())
    img=np.array(img,dtype=np.uint8)
    img=cv2.resize(img,(390,370),interpolation = cv2.INTER_AREA)
    image_tk= PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img))
    canvas.create_image(0,0,anchor=NW, image=image_tk)
    root.mainloop()

label=Label(root,text='Image Processing',background="#1B1D23",fg='white',width='20',font=("Courier", 20)).place(x=230,y=500)
Button1=Button(root,text="Negative",padx=46,pady=20,fg='white',font=myFont,background='#333F50',command=negative)
Button1.place(x=15, y=20)
Button2=Button(root,text="Average",padx=48,pady=20,font=myFont,fg='white',bd=0.5,background='#333F50',command=avg)
Button2.place(x=15, y=100)
Button3=Button(root,text="Blur",padx=64,pady=20,font=myFont,fg='white',bd=0.5,background='#333F50',command=blur)
Button3.place(x=15, y=180)
Button4=Button(root,text="median",padx=52,pady=20,font=myFont,fg='white',bd=0.5,background='#1B1D23',command=medianBlur)
Button4.place(x=15, y=260)
Button5=Button(root,text="sobel",padx=60,pady=20,font=myFont,fg='white',bd=0.5,background='#1B1D23',command=sobel)
Button5.place(x=15, y=340)
Button6=Button(root,text="Laplacian",padx=44,pady=20,font=myFont,fg='white',bd=0.5,background='#1B1D23',command=laplacian)
Button6.place(x=15, y=420)
Button7=Button(root,text="Gaussian",padx=42,pady=20,font=myFont,fg='white',bd=0.5,background='#1B1D23',command=gaussian)
Button7.place(x=610, y=100)
Button8=Button(root,text="Emboss",padx=48,pady=20,font=myFont,fg='white',bd=0.5,background='#1B1D23',command=emboss)
Button8.place(x=610, y=20)
Button9=Button(root,text="Black_White",padx=30,pady=20,font=myFont,fg='white',bd=0.5,background='#1B1D23',command=BlackAndWhite)
Button9.place(x=610, y=180)
Button11=Button(root,text="Sepia",padx=54,pady=20,fg='white',font=myFont,background='#333F50',command=Sepia)
Button11.place(x=610, y=260)
Button12=Button(root,text="Prewitt",padx=52,pady=20,font=myFont,fg='white',bd=0.5,background='#333F50',command=Prewitt)
Button12.place(x=610, y=340)
Button13=Button(root,text="Histogram",padx=40,pady=20,font=myFont,fg='white',bd=0.5,background='#333F50',command=Histogram)
Button13.place(x=610, y=420)
Button14=Button(root,text="Edge Detection",padx=20,pady=20,font=myFont,fg='white',bd=0.5,background='#333F50',command=EdgeDetection)
Button14.place(x=610, y=500)
Button15=Button(root,text="Sharpen",padx=48,pady=20,font=myFont,fg='white',bd=0.5,background='#1B1D23',command=Sharpen)
Button15.place(x=15, y=500)
Button16=Button(root,text="High contrast",padx=30,pady=20,font=myFont,fg='white',bd=0.5,background='#1B1D23',command=contrast)
Button16.place(x=15, y=580)
Button17=Button(root,text="Log",padx=62,pady=20,font=myFont,fg='white',bd=0.5,background='#333F50',command=log)
Button17.place(x=610, y=580)
Button17=Button(root,text="Min",padx=62,pady=20,font=myFont,fg='white',bd=0.5,background='#1B1D23',command=Minimum)
Button17.place(x=400, y=420)
Button17=Button(root,text="Max",padx=62,pady=20,font=myFont,fg='white',bd=0.5,background='#1B1D23',command=Maximum)
Button17.place(x=230, y=420)
Button10=Button(root,text="Open Photo",padx=50,pady=20,font=myFont,fg='black',bd=0.5,background='#52A3B6',command=choose)
Button10.place(x=300, y=580)



root.mainloop()

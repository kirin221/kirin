import numpy as np
from pygame import mixer
import time
import cv2
from tkinter import *
import tkinter.messagebox

import os
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model

import serial

root=Tk()
root.geometry('500x450')
frame = Frame(root, relief=RIDGE, borderwidth=2)
frame.pack(fill=BOTH,expand=1)
root.title('Tugas Akhir')
frame.config(background='light blue')
label = Label(frame, text="Rejector Botol",bg='light blue',font=('Times 20 bold'))
labe2 = Label(frame, text="Tugas Akhir Teknik Otomasi",font=('Times 20 bold'))
label.pack(side=TOP)
labe2.pack(side=BOTTOM)
filename = PhotoImage(file="data.png")
background_label = Label(frame,image=filename)
background_label.pack(side=TOP)

def hel():
   help(cv2)

def Contri():
   tkinter.messagebox.showinfo("Pembuat","\n1.Irfano Arifin \n")


def anotherWin():
   tkinter.messagebox.showinfo("About",'Rejector Botol v1.0\n Made Using\OpenCV\tensorflow\n In Python 3')
                                    
   

menu = Menu(root)
root.config(menu=menu)

subm1 = Menu(menu)
menu.add_cascade(label="Tools",menu=subm1)
subm1.add_command(label="Open CV",command=hel)

subm2 = Menu(menu)
menu.add_cascade(label="About",menu=subm2)
subm2.add_command(label="Yapex",command=anotherWin)
subm2.add_command(label="Contributors",command=Contri)

start = time.time()

model_path = './LDAmodels/model.h5'
model_weights_path = './LDAmodels/weights.h5'

model_path2 = './LDAmodels2/model.h5'
model_weights_path2 = './LDAmodels2/weights.h5'

test_path = 'data/test_image'
test_path2 = 'data/test_image2'

model2 = load_model(model_path2)
model2.load_weights(model_weights_path2)

model = load_model(model_path)
model.load_weights(model_weights_path)
img_width, img_height = 150, 150

#ser = serial.Serial('COM30', 9600)

#print(ser.name)

def exitt():
   exit()

  
def web():
   ser = serial.Serial('COM7', 9600)
   print(ser.name)
   capture =cv2.VideoCapture(0)
   capture2 =cv2.VideoCapture(2)
   
   x, y, w, h = 210, 0, 221, 480 #drawing line untuk crop w,h = ukuran kotak x dan y = ukuran gambar
   x2, y2, w2, h2 = 210, 0, 221, 480 #drawing line untuk crop w,h = ukuran kotak x dan y = ukuran gambar
   
   while True:
      bytesToRead = ser.inWaiting()
      line = ser.read(bytesToRead).decode('UTF-8')
      line = str(line)
        
      #print(line)
      
      ret,frame=capture.read()
      ret2,frame2=capture2.read()
      
      frame = cv2.resize(frame, (640, 480))
      frame2 = cv2.resize(frame2, (640, 480))   
      
      frameCrop = frame[y:y+h, x:x+w]
      frameCrop2 = frame2[y:y+h, x:x+w]
      
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
      
      cv2.rectangle(frame, (x,y), (x+w, y+h), (255,10,10), 2)
      cv2.rectangle(frame2, (x2,y2), (x2+w2, y2+h2), (255,10,10), 2)
      
      cv2.imshow('frame kamera 1',frame)
      cv2.imshow('frame kamera 2',frame2)
      key = cv2.waitKey(1)
      
      if "0" in line:
         time.sleep(2) 
         cv2.imwrite(filename='data/test_image/saved.jpg', img=frameCrop) #save gambar yang ada
         cv2.imwrite(filename='data/test_image2/saved.jpg', img=frameCrop2) #save gambar yang ada
         
         img_ = cv2.imread('data/test_image/saved.jpg', cv2.IMREAD_ANYCOLOR) #membaca gambar
         img2_ = cv2.imread('data/test_image2/saved.jpg', cv2.IMREAD_ANYCOLOR) #membaca gambar
         
         #img_ = cv2.resize(img_,(150,150))
         #img2_ = cv2.resize(img2_,(150,150))
         
         cv2.imwrite(filename='data/test_image/saved.jpg', img=img_) #save gambar yang ada
         cv2.imwrite(filename='data/test_image2/saved.jpg', img=img2_) #save gambar yang ada
               
         cv2.destroyAllWindows()
         
         #Prediction Function
         def predict(file):
           x = load_img(file, target_size=(img_width,img_height))
           x = img_to_array(x)
           x = np.expand_dims(x, axis=0)
           array = model.predict(x)
           result = array[0]
           #print(result)
           answer = np.argmax(result)
           if answer == 0:
             print("Predicted: cacat")
           elif answer == 1:
             print("Predicted: cacatdikit")
           elif answer == 2:
             print("Predicted: normal")

           return answer
           
         def predict2(file):
           x = load_img(file, target_size=(img_width,img_height))
           x = img_to_array(x)
           x = np.expand_dims(x, axis=0)
           array = model2.predict(x)
           result = array[0]
           #print(result)
           answer = np.argmax(result)
           if answer == 0:
             print("Predicted: cacat")
           elif answer == 1:
             print("Predicted: cacatdikit")
           elif answer == 2:
             print("Predicted: normal")

           return answer
           
         #predict kamera 1
         for i, ret in enumerate(os.walk(test_path)):
           for i, filename in enumerate(ret[2]):
             if filename.startswith("."):
               continue
            
             print(ret[0] + '/' + filename)
             global result
             result = predict(ret[0] + '/' + filename)
             print(" ")
         
         
         #predict kamera 2
         for i, ret in enumerate(os.walk(test_path2)):
           for i, filename in enumerate(ret[2]):
             if filename.startswith("."):
               continue
            
             print(ret[0] + '/' + filename)
             global result2
             result2 = predict2(ret[0] + '/' + filename)
             print(" ")
         
         if result == 2 and result2 == 2:
             print("Botol normal")
             ser.write('2'.encode())
         else:
             print("Botol cacat")
             ser.write('1'.encode())
         
         
         #Calculate execution time
         end = time.time()
         dur = end-start

         if dur<60:
            print("Execution Time:",dur,"seconds")
         elif dur>60 and dur<3600:
            dur=dur/60
            print("Execution Time:",dur,"minutes")
         else:
            dur=dur/(60*60)
            print("Execution Time:",dur,"hours")
         
         continue

      elif  key == ord('a'): #nyala
         ser.write('x'.encode())
         print("444")
         continue
         
      elif  key == ord('s'): 
         ser.write('y'.encode())
         print("555")
         continue
         
         
      elif  key == ord('q'):
         capture.release()
         cv2.destroyAllWindows()
         ser.close()
         break
      

   
def alert():
   mixer.init()
   alert=mixer.Sound('beep-07.wav')
   alert.play()
   time.sleep(0.1)
   alert.play()   

   
but1=Button(frame,padx=5,pady=5,width=39,bg='white',fg='black',relief=GROOVE,command=web,text='Buka Kamera',font=('helvetica 15 bold'))
but1.place(x=5,y=150)

but5=Button(frame,padx=5,pady=5,width=5,bg='white',fg='black',relief=GROOVE,text='Exit',command=exitt,font=('helvetica 15 bold'))
but5.place(x=210,y=300)

root.mainloop()


import matplotlib.pyplot as plt
import matplotlib
import numpy as np

######   1. matplotlib.pyplot.plot
#Define a suitable style from the link:("https://matplotlib.org/3.5.0/gallery/style_sheets/style_sheets_reference.html")
plt.style.use('ggplot')
#Define X and y:
x = np.linspace(0,10)
y = np.sin(x)
y1 = np.cos(x)
z = 1-0.2*x
a = np.arange(0,10,0.05)
b = np.exp(-a/2)* np.sin(2*a*np.pi)
#plot X and Y where
# #(1)len(X) must == len(y)
#(2)sort(X) ascending or descending:
#parameters:
#1. marker شكل النقطة https://matplotlib.org/stable/api/markers_api.html
#2. markersize حجم النقطة
#3. color https://matplotlib.org/stable/tutorials/colors/colors.html
#4. ممكن ارسم اكثر من جراف
#5. linestyle https://matplotlib.org/2.0.2/api/lines_api.html#matplotlib.lines.Line2D.set_linestyle
## ممكن اكتب اللون و ال line style جنب بعض '--g'
#6 linewidth
#7 ممكن اعمل X,y, الشكل و X,y2 , hga;g 2 ,i;`h
plt.plot(a,b ,x,y,x, y1,marker = 'o', markersize = 1, color="r", linestyle = "dashed",linewidth= 7)
plt.plot(a,b ,'or',x,y,'-b',x, y1,'.c',y,y1,'--w',y,y/2,'black',x,((x/x)*-1),'black',x,z,'black')
plt.plot(y+1,y1,'black',x,((x/x)*-1),'black',x,z,'black')
plt.plot(x,y,'r.')
plt.plot(a,b,'1r')
#3 طرق لعمل ال X label و y label
# 1. name X and Y axis:
plt.xlabel('X')
plt.ylabel('y')

#2. create a variable contains the axes class
#كل دا يتعمل قبل ال plot
ax = plt.axes()
ax.set_xlabel('No')
ax.set_ylabel('yes')
plt.plot(x,y)

#3. ax.set(xlabel = "", ylabel ="")
ax.set_title("no world hello")
ax = plt.axes()
ax.set(xlabel = "BAZ" , ylabel = "Ahmed")
plt.plot(x,y)

#طريقتين لعمل ال Title
ax.set(title = "Hello world")
ax.set_title("no world hello")


######   1. matplotlib.pyplot.errorbar
x = np.linspace(0,10,10)
conf_interval = [-0.8,0.8]
#هنا بجمع طرفي ال CI علشان اديب المسافه بينهم
dy = np.sum(np.abs(conf_interval[0])+np.abs(conf_interval[1]))
# هنا بعمل ال y
y = np.sin(x) + dy*np.random.randn(50)
y = np.sin(x)
# دا البلوت العادي
plt.errorbar(x,y,yerr=dy, fmt='.b')
# دا بلوت يناسب عدد 10 X
plt.errorbar(x,y,yerr=[1,2,3,4,5,6,7,8,9,11], fmt='.b')
# طورت ال function علشان تاخد CIs بعدد ال نقط اللي في X و y
# بعد ما اكتشفت ان ال yerr ممكن تكون array
dy1=[]
CI =[[-0.1,0.21],[-0.5,0.1],[1.2,0.8],[0.3,0.6],[-0.1,0.1],[0.8,0.6],[0.4,0.3],[0.1,0.2],[0.1,0.2],[0.1,0.2]]
for i in range(0,len(x)):
    dy = np.sum(np.abs(CI[i][0])+np.abs(CI[i][1]))
    dy1.append(dy)
plt.errorbar(x,y,yerr=dy1,fmt='.b')


###### matplotlib.pyplot.scatter
#بتاخد اربع حاجات
#(mandatory) X and Y
#C: color https://matplotlib.org/stable/tutorials/colors/colors.html
#S: size (0.1,1,2,3,...)
#alpha: transparency [0:1]
X = np.random.normal(0,100,1000)
Y = np.random.normal(0,100,1000)
plt.scatter(X,Y, c='r',s=250,alpha=0.5,edgecolors='b')
plt.plot(np.sort(X),np.sort(Y),'r--',np.sort(Y))
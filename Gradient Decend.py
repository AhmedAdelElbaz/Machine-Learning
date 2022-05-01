from numpy import *
## Read the y x and m from a table and add ones to the first column in x matrix
y = array([[3,4,6]])
x = array([[1,2,4,2],[1,7,6,1],[1,8,5,7]])
m = y.shape[1]
alpha = float(input("what the alpha value ?"))
theta = array([[0,0,0,0]])
J = 1000
iters = int(input("How many steps do you need ? "))
optimumsteps = 0
for i in range(1,iters):
## for loap to form all hxmy
    hx = theta.dot(x.T)
    hxmy = hx - y
    Thxmy = sum(hxmy)
    hxmys = hxmy*hxmy
    Thxmys = sum(hxmys)
    hxmyx1 = hxmy*x[:,1]
    Thxmyx1 = sum(hxmyx1)
    hxmyx2 = hxmy*x[:,2]
    Thxmyx2 = sum(hxmyx2)
    hxmyx3 = hxmy*x[:,3]
    Thxmyx3 = sum(hxmyx3)
    newJ = (1/2*m)*Thxmys
    if newJ >= J:
        break
    else:
        J = newJ
    optimumsteps = i
## for loap to form theta array
    theta0 = theta[0][0]- alpha/m*Thxmy
    theta1 = theta[0][1]- alpha/m*Thxmyx1
    theta2 = theta[0][2]- alpha/m*Thxmyx2
    theta3 = theta[0][3]- alpha/m*Thxmyx3
    theta = array([[theta0,theta1,theta2,theta3]])

print('Theta = ' + str(theta))
print('J = '+ str(J))
print('expected values = ' + str(hx))
print('optimumsteps = ' + str(optimumsteps))





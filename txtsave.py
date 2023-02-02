
#log.txt 생성용

import datetime

name = 'ship'
x = 10
y = 20


f = open('log.txt','a')

msg = [str(datetime.datetime.now()),': [',str(name),'] is detected... Location : X = ',str(x),', Y = ',str(y),'\n']
msg = ''.join(msg)
f.write(msg)
f.close 


name2 = 'airplane'
x2 = 11
y2 = 22

f = open('log.txt','a')
msg = [str(datetime.datetime.now()),': [',str(name),'] is detected... Location : X = ',str(x),', Y = ',str(y),'\n']
msg = ''.join(msg)
f.write(msg)
f.close 

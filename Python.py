#variables

Name="John Doe"
Age=30
Height=5.9
Weight=160.5
IsEmployed=True

print(type(Name))      
print(type(Age))  
print(type(Height))      
print(type(Weight))
print(type(IsEmployed))

# assign vaules to variables
x="hero"
y="tvs"
z="honda"

print(x)
print(y) 
print(z)

x="John"
y="is good"
z="boy"
print(x,y,z)



# functions

def myfunction():
    print("In python" ,x,y,z)
myfunction()

# conditons

a= 10
b= 20
if a < b:
    print("a is less than b")
elif a > b: 
    print("a is greater than b")
else:
    print("a is equal to b")

x=10
if x>0:
    print("x is positive")
elif x<0:
    print("x is negative")


# loops
for i in range(5):
    print("Iteration:", i)  
for i in range(1, 6):
    print("Number:", i)

# while loop
count = 0
while count < 5:
    print("Count is:", count)
    count += 1
    
# operators
a=10
b=20

print(a + b)
print(a - b)
print(a * b)
print(a / b)
print(a % b)
print(a==b)
print(a!=b)

# data structures list, tuple, set, dictionary

    
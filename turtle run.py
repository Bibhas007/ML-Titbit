import turtle

# Create screen and turtle objects
screen = turtle.Screen()
t = turtle.Turtle()
t.speed(2) # Set drawing speed to fastest

# Define dimensions
rect_width = 150
rect_height = 80
tri_side = 80 # Side length of equilateral triangles

# Move to starting position for the rectangle
t.penup()
t.goto(-rect_width / 2, -rect_height / 2)
t.pendown()

# Draw the rectangle
for _ in range(2):
    t.forward(rect_width)
    t.left(90)
    t.forward(rect_height)
    t.left(90)

# left angle
t.left(144)
t.forward(70)
t.right(110)
t.forward(70)

# Right angle
t.penup()
t.goto(rect_width / 2, rect_height / 2)
t.pendown()
t.right(72)
t.forward(70)
t.left(250)
t.forward(67)

t.hideturtle()
turtle.done()

from tkinter import *
import tkinter.messagebox
import urllib.request
import turtle


def main():
    counts = analyzeFile(url.get())
    drawHistogram(counts)

def analyzeFile(url):
    try:
        infile = urllib.request.urlopen(url)
        s = str(infile.read().decode()) # Read the content as string from the URL

        counts = countLetters(s.lower())

        infile.close() # Close file
    except ValueError:
        tkinter.messagebox.showwarning("Analyze URL",
            "URL " + url + " does not exist")

    return counts

def countLetters(s):
    counts = 26 * [0] # Create and initialize counts
    for ch in s:
        if ch.isalpha():
            counts[ord(ch) - ord('a')] += 1
    return counts

def drawHistogram(list):

    WIDTH = 400
    HEIGHT = 300

    raw_turtle.penup()
    raw_turtle.goto(-WIDTH / 2, -HEIGHT / 2)
    raw_turtle.pendown()
    raw_turtle.forward(WIDTH)

    widthOfBar = WIDTH / len(list)

    for i in range(len(list)):
        height = list[i] * HEIGHT / max(list)
        drawABar(-WIDTH / 2 + i * widthOfBar,
            -HEIGHT / 2, widthOfBar, height, letter_number=i)

    raw_turtle.hideturtle()

def drawABar(i, j, widthOfBar, height, letter_number):
    alf='abcdefghijklmnopqrstuvwxyz'
    raw_turtle.penup()
    raw_turtle.goto(i+2, j-20)

    #sign letter on histogram
    raw_turtle.write(alf[letter_number])
    raw_turtle.goto(i, j)

    raw_turtle.setheading(90)
    raw_turtle.pendown()


    raw_turtle.forward(height)
    raw_turtle.right(90)
    raw_turtle.forward(widthOfBar)
    raw_turtle.right(90)
    raw_turtle.forward(height)

window = Tk()
window.title("Occurrence of Letters in a Histogram from URL")

frame1 = Frame(window)
frame1.pack()

scrollbar = Scrollbar(frame1)
scrollbar.pack(side = RIGHT, fill = Y)

canvas = tkinter.Canvas(frame1, width=450, height=450)
raw_turtle = turtle.RawTurtle(canvas)

scrollbar.config(command = canvas.yview)
canvas.config( yscrollcommand=scrollbar.set)
canvas.pack()

frame2 = Frame(window)
frame2.pack()

Label(frame2, text = "Enter a URL: ").pack(side = LEFT)
url = StringVar()
Entry(frame2, width = 50, textvariable = url).pack(side = LEFT)
Button(frame2, text = "Show Result", command = main).pack(side = LEFT)

window.mainloop()
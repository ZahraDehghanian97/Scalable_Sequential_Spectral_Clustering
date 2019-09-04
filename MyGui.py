from tkinter import *
from tkinter.filedialog import asksaveasfilename
from tkinter.filedialog import askopenfilename
from tkinter import ttk
import Kmeans
import SC
import SeqSC
def Save():
    asksaveasfilename(defaultextension = '.txt', \
                        filetypes = (('Text files', '*.txt'),
                                    ('Python files', '*.py *.pyw'),
                                    ('All files', '*.*')))


def setting():
    type = v.get()
    if type == 1 :
        SC.sc()
    if type == 2 :
        SeqSC.seqsc()
    if type == 3 :
        Kmeans.kmeans()
    if type == 4 :
        Kmeans.kmeansplusplus()


root = Tk()
root.title('Clustering')

label_frame = Frame(root)
label_frame.grid(sticky=W)
Label(root,text="""Choose type of clustering :""").grid(row=0, column=0)
Label(root,text=""" """).grid(row=1, column=0)
ratio_frame = Frame(root)

v = IntVar()
v.set(1)
Radiobutton(text = "Spectral Clustering",variable=v,value = 1).grid(row=2, column=0)
Radiobutton(text = "SeqSC",variable=v,value = 2).grid(row=2, column=1)
Radiobutton(text = "Kmeans",variable=v,value = 3).grid(row=3, column=0)
Radiobutton(text = "Kmeans++",variable=v,value = 4).grid(row=3, column=1)
Label(root,text=""" """).grid(row=4, column=0)
button_frame = Frame(root)
button_frame.grid(sticky=(E))
Button(button_frame, text='cluster',command = setting, height = 1, width = 10).grid(row=5, column=4)

root.mainloop()
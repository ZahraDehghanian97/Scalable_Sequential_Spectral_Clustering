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


def Open():
    askopenfilename(initialdir = 'Desktop')

def setting():
    type = v.get()
    if type == 1 :
        SC.guisc()
    if type == 2 :
        root2 = Tk()
        root2.title('Clustering')
        Label(root2, text="""please check desired value :""").grid(row=0, column=0,columnspan=3)
        Label(root2, text=""" """).grid(row=1, column=0)
        k = IntVar()
        k.set(10)
        n = IntVar()
        n.set(700)
        m = IntVar()
        m.set(70)
        filter0_255 = IntVar()
        filter0_255.set(1)
        k_entry = Entry(root2, width=7, textvariable=k)
        k_entry.insert(0,10)
        k_entry.grid(column=1, row=2, sticky=E,padx=5)
        Label(root2, text="k = ").grid(column=0, row=2, sticky=E)
        n_entry = Entry(root2, width=7, textvariable=n)
        n_entry.insert(0, 700)
        n_entry.grid(column=3, row=2, sticky=E,padx= 10)
        Label(root2, text="n = ").grid(column=2, row=2, sticky=E)
        m_entry = Entry(root2, width=7, textvariable=m)
        m_entry.insert(0, 70)
        m_entry.grid(column=5, row=2, sticky=E,padx=5)
        Label(root2, text="m = ").grid(column=4, row=2, sticky=E)
        Label(root2, text="Current Data Frame :").grid(column=0, row=3, sticky=E)
        Label(root2, text="Mnist    ").grid(column=1, row=3, sticky=E)
        Label(root2, text="New Data (optional):").grid(column=0, row=4, sticky=E)
        Button(root2, text='open', command=Open, height=1).grid(padx= 6,pady = 3,row=4, column=1,sticky = W)

        Checkbutton(root2, text='filter(0-1)', variable=filter0_255).grid(column=3, row=3,columnspan=2, sticky=E)
        Button(root2, text='cluster', command=SeqSC.guiseqsc(k.get(),n.get(),n.get(),filter0_255.get()),height=2,width= 15).grid(pady=3,padx=4, row=4,columnspan=3, column=3, sticky=W)

        root2.mainloop()
        # SeqSC.guiseqsc()
    if type == 3 :
        Kmeans.guikmeans()
    if type == 4 :
        Kmeans.guikmeansplusplus()


root = Tk()
root.title('Clustering')

label_frame = Frame(root)
label_frame.grid(sticky=W)
Label(root,text="""Choose type of clustering :""").grid(row=0, column=0)
Label(root,text=""" """).grid(row=1, column=0)
ratio_frame = Frame(root)

v = IntVar()
v.set(2)
Radiobutton(text = "Spectral Clustering",variable=v,value = 1).grid(row=2, column=0)
Radiobutton(text = "SeqSC",variable=v,value = 2).grid(row=2, column=1)
Radiobutton(text = "Kmeans",variable=v,value = 3).grid(row=3, column=0)
Radiobutton(text = "Kmeans++",variable=v,value = 4).grid(row=3, column=1)
Label(root,text=""" """).grid(row=4, column=0)
button_frame = Frame(root)
button_frame.grid(sticky=(E))
Button(button_frame, text='cluster',command = setting, height = 1, width = 10).grid(row=5, column=4)

root.mainloop()
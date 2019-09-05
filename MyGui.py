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


def ResultPage(x, y, labels):
    pass


def runseqsc():
    global k,n,m,f
    t_k = k.get()
    t_n = n.get()
    t_m = m.get()
    t_f = f.get()
    print("k,n,m,f")
    print(t_k,t_n,t_m,t_f)
    if t_n == 0 or t_m == 0 or t_k == 0 :
        print("please change 0 value")
        return 
    else :
        x, y, labels = SeqSC.guiseqsc(t_k,t_n,t_n,t_f)
        ResultPage(x,y,labels)
    return 


def setting():
    global v,k,m,n,f,asli
    type = v.get()
    if type == 1 :
        SC.guisc()
    if type == 2 :
        root2 = Toplevel(asli)
        root2.title('Clustering')
        Label(root2, text="""please check desired value :""").grid(row=0, column=0,columnspan=3)
        Label(root2, text=""" """).grid(row=1, column=0)
        k_entry = Entry(root2, width=7, textvariable=k)
        k_entry.insert(0,0)
        k_entry.grid(column=1, row=2, sticky=E,padx=5)
        Label(root2, text="k = ").grid(column=0, row=2, sticky=E)
        n_entry = Entry(root2, width=7, textvariable=n)
        n_entry.insert(0, 0)
        n_entry.grid(column=3, row=2, sticky=E,padx= 10)
        Label(root2, text="n = ").grid(column=2, row=2, sticky=E)
        m_entry = Entry(root2, width=7, textvariable=m)
        m_entry.insert(0, 0)
        m_entry.grid(column=5, row=2, sticky=E,padx=5)
        Label(root2, text="m = ").grid(column=4, row=2, sticky=E)
        Label(root2, text="Current Data Frame :").grid(column=0, row=3, sticky=E)
        Label(root2, text="Mnist    ").grid(column=1, row=3, sticky=E)
        Label(root2, text="New Data (optional):").grid(column=0, row=4, sticky=E)
        Button(root2, text='open', command=Open, height=1).grid(padx= 6,pady = 3,row=4, column=1,sticky = W)

        Checkbutton(root2, text='filter(0-1)', variable=f).grid(column=3, row=3,columnspan=2, sticky=E)
        Button(root2, text='cluster', command=runseqsc,height=2,width= 15).grid(pady=3,padx=4, row=4,columnspan=3, column=3, sticky=W)

        root2.mainloop()
    if type == 3 :
        Kmeans.guikmeans()
    if type == 4 :
        Kmeans.guikmeansplusplus()


asli = Tk()
root = Toplevel(asli)
root.title('Clustering')
asli.geometry("5x5")
Label(root,text="""Choose type of clustering :""").grid(row=0, column=0)
Label(root,text=""" """).grid(row=1, column=0)
k = IntVar()
k.set(10)
n = IntVar()
n.set(300)
m = IntVar()
m.set(30)
v = IntVar()
v.set(2)
f = IntVar()
f.set(1)
Radiobutton(root,text = "Spectral Clustering",variable=v,value = 1).grid(row=2, column=0)
Radiobutton(root,text = "SeqSC",variable=v,value = 2).grid(row=2, column=1)
Radiobutton(root,text = "Kmeans",variable=v,value = 3).grid(row=3, column=0)
Radiobutton(root,text = "Kmeans++",variable=v,value = 4).grid(row=3, column=1)
Label(root,text=""" """).grid(row=4, column=0)
Button(root, text='cluster',command = setting, height = 1, width = 10).grid(row=5,columnspan = 2,padx = 10, column=0,sticky = E)

root.mainloop()
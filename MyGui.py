from tkinter import *
from tkinter import ttk

def calculate(*args):
    try:
        value = float(feet.get())
        meters.set((0.3048 * value * 10000.0 + 0.5)/10000.0)
    except ValueError:
        pass

root = Tk()
root.title("clustering")

mainframe = ttk.Frame(root, padding="3 3 12 12")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

feet = StringVar()
meters = StringVar()

# feet_entry = ttk.Entry(mainframe, width=7, textvariable=feet)
# feet_entry.grid(column=2, row=1, sticky=(W, E))
ttk.Label(mainframe, text="choose clustering type first ...").grid(column=1, row=1, sticky=W)

ttk.Radiobutton(root, text="Kmeans", value=1).grid(column=1, row=2, sticky=(W))
ttk.Radiobutton(root, text="Spectral Clustering",value=2).grid(column=2, row=2, sticky=(W))
ttk.Radiobutton(root, text="SeqSC",  value=3).grid(column=3, row=2, sticky=(W))
# ttk.Label(mainframe, textvariable=meters).grid(column=2, row=2, sticky=(W, E))
ttk.Button(mainframe, text="Cluster", command=calculate).grid(column=3, row=3, sticky=W)

# ttk.Label(mainframe, text="feet").grid(column=3, row=1, sticky=W)
# ttk.Label(mainframe, text="is equivalent to").grid(column=1, row=2, sticky=E)

# for child in mainframe.winfo_children(): child.grid_configure(padx=5, pady=5)
#
# feet_entry.focus()
# root.bind('<Return>', calculate)

root.mainloop()
from ipywidgets import IntProgress, Layout, Text
from IPython.display import clear_output, display
import time
from itertools import groupby

def progress_bar(result, size):
    w = IntProgress()
    w.max = size
    w.add_class('mypb')
    display(w)

    w2 = Text(value='0%')
    display(w2)

    while not result.ready():
        w.value = w.max - result._number_left
        w2.value = "{:.1f}%".format(100.0 * w.value / w.max)
        time.sleep(.1)
        # print('.', end='')

    w.value = w.max
    w2.value = "Completed: {:.1f}%".format(100.0 * w.value / w.max)



def niners(num):
    flag = 0
    for k,g in groupby(str(num)):
        if k == '.':
            flag = 1
        elif flag == 1 and k == '9':
            return (k, len(list(g)))

    return ('9', 0)

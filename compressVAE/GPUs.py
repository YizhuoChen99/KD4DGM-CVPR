from time import sleep
from pynvml import *


def select(wait):
    try:
        nvmlInit()
    except Exception:
        return '0'
    else:
        pass

    deviceCount = nvmlDeviceGetCount()
    while True:

        if_find = 0
        index = 0
        min_occupy_ratio = 999
        name = ''
        for i in range(deviceCount):
            handle = nvmlDeviceGetHandleByIndex(i)
            meminfo = nvmlDeviceGetMemoryInfo(handle)
            occupy_ratio = meminfo.used / meminfo.total
            if occupy_ratio < min_occupy_ratio:
                if_find = 1
                index = i
                min_occupy_ratio = occupy_ratio
                name = nvmlDeviceGetName(handle)

        print(min_occupy_ratio, flush=True)
        if min_occupy_ratio < wait:
            break
        sleep(67)

    nvmlShutdown()
    if if_find:
        print("Free GPU find (index): {}"
              "its name: {}"
              "occupy_ratio: {}".format(index, name, min_occupy_ratio),
              flush=True)
        return str(index)
    else:
        print("No free GPU now, please wait!")
        raise KeyboardInterrupt

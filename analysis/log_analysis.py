from xml.dom import INDEX_SIZE_ERR
import pandas as pd
import numpy as np
import os

def log_analysis(log_file):
    with open(log_file,'r') as f:
        lines = f.readlines()
        whole_cur_acc = []
        whole_avg_acc = []
        whole_ratio = []  # noise ration in memory
        whole_forget = []
        for i in range(len(lines)):
            if "task--10" in lines[i]:
                cur_acc = eval(lines[i+3].strip("\n"))
                avg_acc = eval(lines[i+4].strip("\n"))
                whole_cur_acc.append(cur_acc)
                whole_avg_acc.append(avg_acc)
                ## calculate forget 
                whole_acc = eval(lines[i+5].strip("\n"))
                last_acc = np.array(whole_acc[-1])
                whole_acc = np.array(list(map(lambda l:l + [0]*(10 - len(l)), whole_acc)))
                max_acc = np.max(whole_acc, axis=0)
                forget_rate = max_acc - last_acc
                whole_forget.append(forget_rate.tolist())
                i = i+7
    whole_cur_acc = np.mean(np.array(whole_cur_acc), axis=0)
    whole_avg_acc = np.mean(np.array(whole_avg_acc), axis=0)
    whole_forget = np.mean(np.array(whole_forget), axis=0)
    print("log_file: ", log_file)
    print("whole_cur_acc: ", whole_cur_acc)
    print("whole_avg_acc:", whole_avg_acc)
    print("whole_forget: ", whole_forget)

def log_forget(log_file):
    with open(log_file,'r') as f:
        lines = f.readlines()
        whole_forget = []
        for i in range(len(lines)):
            if "task--10" in lines[i]:
                cur_forget = []
                whole_acc = eval(lines[i+5].strip("\n"))
                print("whole_acc: ", whole_acc)
                first_acc = whole_acc[0][0]
                for j in range(9):
                    print("cur_forget: ", first_acc - whole_acc[j+1][0])
                    cur_forget.append(first_acc - whole_acc[j+1][0])
                whole_forget.append(cur_forget)
                i = i+7
    whole_forget = np.mean(np.array(whole_forget), axis=0)
    print("forget_rate: ", whole_forget)


if __name__ == "__main__":
    file_name = "outlogs/fewrel_clean_forget_94.out"
    log_forget(file_name)



    
   
        
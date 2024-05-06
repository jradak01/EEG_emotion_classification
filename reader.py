from PyNeuro.PyNeuro import PyNeuro
from time import sleep
import csv

#create PyNeuro object - inicialize
pn = PyNeuro()

#stop connection with mindwave
def stop():
    pn.disconnect()
    pn.close()

#start connection to mindwave
def start():
    pn.connect()
    pn.start()
    
    #create new csv
    fieldnames = ["attention", "meditation", "delta", "theta", "lowalpha", "highalpha", "lowbeta", "highbeta", "lowgamma", "highgamma", "status" ]
    with open('data.csv', 'w', newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
    #update csv with new readings
    while True:
        with open('data.csv', 'a', newline='') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            info = {
                "attention": pn.attention,
                "meditation": pn.meditation,
                "delta": pn.delta, 
                "theta": pn.theta, 
                "lowalpha": pn.lowAlpha, 
                "highalpha": pn.highAlpha, 
                "lowbeta": pn.lowBeta, 
                "highbeta": pn.highBeta, 
                "lowgamma": pn.lowGamma, 
                "highgamma": pn.highGamma,
                "status":pn.status
            }
            csv_writer.writerow(info)
        sleep(1)

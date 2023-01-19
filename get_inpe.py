from datetime import datetime
from datetime import timedelta

import os
import time
import requests
import json

##### Constants
url = "https://queimadas.dgi.inpe.br/queimadas/exportacaobdq/exportar"
countries = "33" # Basil
continent = "8" # Soult America
satellites = "" # All
email = "tccinpeufrgs@gmail.com"
format = "csv"
gap_request = timedelta(days=30)
min_date = datetime(1998, 1, 1).date()
###############################################

data = {
    "dateTimeFrom": "",
    "dateTimeTo": "",
    "satellites": satellites,
    "biomes": "",
    "continent": continent,
    "countries": countries,
    "states": "",
    "cities": "",
    "specialRegions": "",
    "industrialFires": False,
    "bufferInternal": "false",
    "bufferFive": "false",
    "bufferTen": "false",
    "email": email,
    "format": format
}
directory = os.getcwd()
path = os.path.join(directory, 'request_list.csv')

count = 0
max_requests = 3000
file = open(path, "r")
lines = file.readlines()
file.close()

data_files = os.listdir(os.path.join(directory, 'data'))

for index, line in enumerate(lines):
    dateTimeFrom, dateTimeTo, state = line.strip().split(",")
    
    local_file = "Focos_" + dateTimeFrom.split(" ")[0] + "_" + dateTimeTo.split(" ")[0] + ".csv"
    if state == 'AWAIT':
        if local_file in data_files:
            print("Fold local file", local_file)
            lines[index] = dateTimeFrom + "," + dateTimeTo + ",OK\n"
            with open(path, 'w') as file:
                file.writelines(lines)
    
    if state == 'OK':
        if local_file not in data_files:
            print("Not fold local file", local_file, "tring again...")
            state = 'IDLE'
            lines[index] = dateTimeFrom + "," + dateTimeTo + ",IDLE\n"
            with open(path, 'w') as file:
                file.writelines(lines)

    if state == 'IDLE':
        data["dateTimeFrom"] = dateTimeFrom
        data["dateTimeTo"] = dateTimeTo

        print("Request from", dateTimeFrom, "to", dateTimeTo)
        response = requests.get(url, params={"data": json.dumps(data)})
        print("Requested", count)
        if (response.status_code == 200):
            lines[index] = dateTimeFrom + "," + dateTimeTo + ",AWAIT\n"
            with open(path, 'w') as file:
                file.writelines(lines)

            count += 1
            if count == max_requests or len(lines) == index - 1:
                break
            time.sleep(90)
        
    
        

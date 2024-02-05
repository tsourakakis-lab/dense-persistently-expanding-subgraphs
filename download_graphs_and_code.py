import urllib.request
import zipfile
import os

if __name__ == "__main__": 
        urls = ['https://nrvis.com/download/data/dynamic/aves-wildbird-network.zip',
                'https://nrvis.com/download/data/dynamic/ia-radoslaw-email.zip',
                'https://networks.skewed.de/net/us_air_traffic/files/us_air_traffic.csv.zip',
                'https://networks.skewed.de/net/sp_primary_school/files/sp_primary_school.csv.zip'
                ]
        folders = ['animal', 'email',  'airport', 'school']
        filename = ['aves-wildbird-network.zip', 'ia-radoslaw-email.zip', 'airport.csv.zip', 'school.csv.zip']

        if not os.path.exists('./Graphs'):
                os.makedirs('./Graphs')
        if not os.path.exists('./Results_'):
                os.makedirs('./Results_')
        if not os.path.exists('./Plots_'):
                os.makedirs('./Plots_')
        for name in folders:
                if not os.path.exists(f'./Graphs/{name}'):
                        os.makedirs(f'./Graphs/{name}')

        for i in range(len(urls)):
                print(urls[i])
                urllib.request.urlretrieve(urls[i], f'./Graphs/{folders[i]}/{filename[i]}')
                try:
                        with zipfile.ZipFile(f'./Graphs/{folders[i]}/{filename[i]}', 'r') as zip_ref:
                                zip_ref.extractall(f'./Graphs/{folders[i]}/')
                except:
                        print(f'Error while downloading {urls[i]}...')
        
        code_url = 'https://github.com/ksemer/BestFriendsForever-BFF-/archive/refs/heads/master.zip'
        urllib.request.urlretrieve(code_url, f'./BestFriendsForever-BFF.zip')
        try:
                with zipfile.ZipFile(f'./BestFriendsForever-BFF.zip', 'r') as zip_ref:
                        zip_ref.extractall(f'./')
        except:
                print(f'Error while downloading {code_url}...')

        #mkdir venv && cd venv && python3 -m venv env && cd .. && source ./venv/env/bin/activate && pip3 install -r requirements.txt
        #mkdir -p ./BestFriendsForever-BFF--master/java/config && mv ./settings.properties ./BestFriendsForever-BFF--master/java/config
        #cd ./BestFriendsForever-BFF--master/java && javac system/Main.java && java system/Main.java && cd .. && cd ..
        
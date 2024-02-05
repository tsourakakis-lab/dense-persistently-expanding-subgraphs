# Finding Dense and Persistently Expansive Subgraphs

Requirments: Python3, Java >=12, Mosek

## Install Packages
```zsh 
mkdir venv && cd venv && python3 -m venv env && cd .. && source ./venv/env/bin/activate && pip3 install -r requirements.txt
```

## Download Datasets and Code for Baselines
```zsh 
python3 download_graphs_and_code.py
```

### Generate Graphs
```zsh 
python3 make_graphs.py
```

### Run experiments for SDP and NDS
```zsh 
python3 experiments.py
```

### Run experiments for AM and MM baselines
```zsh 
mkdir -p ./BestFriendsForever-BFF--master/java/config && mv ./settings.properties ./BestFriendsForever-BFF--master/java/config

cd ./BestFriendsForever-BFF--master/java && javac system/Main.java && java system/Main.java && cd .. && cd ..

python3 AM_MM.py
```


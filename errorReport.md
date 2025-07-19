# Report of Errors

This document is solely to report all the mistakes/errors I made
during this project and how it made me improve.

## Errors during the fetching part

0. Missing Libraries (pandas, matplotlib) and not using the `venv` when executing python3.

1. Without the `df.reset_index` in **fetch.py**, everytime I fetched the 
.csv data it would not have a "Date" column, so **eda.py** would not work 
properly.

2. Everytime I'd fetch a .csv file I'd have to manually erase the second row of trash because if not, matplotlib would go crazy with the trends.

3. When it came to training the model, I chose LSTM in order to create better
quality predictions. I had to install `tensorflow` and I couldnt with my
normal `venv`, so I created an enviroment using anaconda.

4. Using recursive prediction accumulates mistakes over time. First model I
trained looked good in the beginning... not so good in the end. I overtrained the model and future predictions were no good. 

5. Switched to bidirectional LSTM to improve future predictions. I Have to add 
more steps
Hello, this is a crop recommendation system with completely functional UI and backend that exploits SHAP to make a smart crop recommendation system. 

Steps to run the project : 
1) Create a python3.10 virtual environment by the name of 'venv' such that the first parent directory of Data, api and frontend folder is exactly the same.
2) Activate the venv and upgrade your pip in the virtual environment.
3) With the virtual environment activated, hit 'pip install -r requirements.txt' to download the components of the virtual environment.
4) Go to api folder in terminal and hit 'npm i' to download the node modules. Make sure that you have node installed in your system.
5) Similarly, go to frontend folder and hit 'npm i' to download the node modules. Make sure that you have react installed in your system.
6) Run the brainIt2.py on the descripted data in the Data folder. This is essential to make the prediction model. Or you can simply go with my created model as well. You can use your own dataset for training the model or my dataset works just fine as well. 
7) Then, go to api folder via terminal and launch the npm backend using 'node index.js' to run the model in a temporary python shell and connect it as a backend server. 
8) Using the 'npm start' command in frontend folder, launch the UI on react local host. 

File description in the order of their creation : 

1) /Data/Crop_recommendation.csv : Kaggle fetched dataset that suggests crops out of 22 different crops using 7 parameters(N, P, K content, pH, humidity, rainfall, temperature)

2) brainIt1.py : This makes a model that uses all the above mentioned parameters to predict the results. Cause of rejection : Uses all the parameters. In a real world scenario, a farmer might not be aware of certain factors. Name is brainIt1.py since it is the first iteration of the main brain of the model. 

3) brainIt2.py : This was an improvement on the previous version of the model that helped to randomly remove a few labels and then train the model. 

4) whacker1.py : This was a test to load the model, input a few parameters and run the model. 

5) whacker2.py : This was a sensitivity analysis which selects 1000 random inputs(equally taken from each class) and removes 2 features randomly from each row to analyse which factor is most crucial for a good result. After about 15 tests, rainfall and humidity values are the most critical ones. 

6) shapExplainer.py : Testing for SHAP explainability so that I can verify the effectiveness of the model and the explainability part of it as well. 

7) predict_and_explainpy : The final model that uses the model brain to predict results, make waterfall graphs using SHAP and make a comprehensive report about the prediction. 

8) api/index.js : This file is the API call handler for the back end. It takes the user input and spawns a python thread to run the model, make and store explainability graphs in api/results folder and send the data back to front end. 

9) frontend/src/App.js : This file is the main frontend of the project that takes the data from the user, preprocesses it and sends to back end for getting results. In case the user doesn't enter an input, it is considered that the value for that is unknown and the result is made on that basis. 

Please refer to the testing videos for seeing the results and to get a guide on the use of the model in the guide directory on this repository. 

Scope of improvement : A weather API can be used to estimate the rainfall, humidity and temperature by asking the user's region instead of asking it from the user. 


This project is the sole property, creation and realisation of Vishesh Goyal, that is me. 
Any suggestions and changes are welcome at visheshvishu1@outlook.com or visheshvishu1@gmail.com. 

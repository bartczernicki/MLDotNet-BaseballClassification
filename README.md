# Baseball Predictions - Training Model Job
![Training Job](https://github.com/bartczernicki/MLDotNet-BaseballClassification/blob/master/images/BaseballPredictionsTrainingModelJob.gif)

A .Net Core model building job that builds several models using MLB Baseball data from 1876 - 2019.  

The outcome are two classification supervised learning predictions:
* On Hall Of Fame Ballot - whether a batter will be on the Hall of Fame Ballot, based on their career statistics
* Inducted To Hall Of Fame - whether a batter will be inducted to the Hall of Fame, based on their career statistics

The model building job includes the following features:
* Builds multiple ML.NET binary classification models in a single C# "script" (training job)
* Dynamic Feature Selection - Select features from a configuration array to adjust model input dynamically
* Dynamic Supervised Learning - Includes two label fields in a single data set, that can be switched dynamically
* Base data transformer pipeline that is re-used for all trained models as a base
* Reports various performance metrics using a pre-defined holdout set
* Persists the trained models in two different formats: native ML.NET and ONNX
* Loads the persisted models from storage and performs model explainability
* Applies simple perscriptive/rules engine to select the "best model"
* Selected "best model" is used for inference on new ficticious baseball player careers (to verify overall performance)

Requirements (what the solution has been developed, compiled with orginally through current):
* Visual Studio 2019 IDE (Community SKU or higher), .NET Core 3.x - .NET 5.x, ML.NET v1.1 - v1.5.5


# MLDotNet-BaseballClassification
A .Net Core model building job that builds several models using MLB Baseball data from 1876 - 2017.  

The outcome are two supervised learning predictions:
* On Hall Of Fame Ballot - whether a batter will be on the Hall of Fame Ballot, based on their career statistics
* Inducted To Hall Of Fame - whether a batter will be inducted to the Hall of Fame, based on their career statistics

The model building job includes the following features:
* Builds multiple binary classification models in a single C# script
* Dynamic Feature Selection - Select features from an array to adjust model input
* Dynamic Supervised Learning - Includes two label fields in a single data set, that can be switched dynamically
* Persists the trained models in two different formats: native ML.NET and ONNX
* Reports various performance metrics
* Applies simple perscriptive/rules engine to select the "best model"
* Selected model is used for inference on new ficticious baseball player careers

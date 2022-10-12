using System;
using System.Reflection;

using Microsoft.ML;


string _appPath = Environment.CurrentDirectory;
string _trainDataPath = Path.Combine(_appPath,"data","issues_train.tsv");
string _testDataPath = Path.Combine(_appPath,"data","issues_test.tsv");
string _modelPath = Path.Combine(_appPath,"models","model.zip");

MLContext _mlContext = new(seed: 0);
PredictionEngine<GitHubIssue, IssuePrediction> _predEngine ;
ITransformer _trainedModel;

//LOAD DATA
IDataView _trainingDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(_trainDataPath, hasHeader: true);


//TRANFORM DATA INPUT
var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Area", outputColumnName: "Label").Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Title", outputColumnName: "TitleFeaturized")).Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Description", outputColumnName: "DescriptionFeaturized")).Append(_mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized")).AppendCacheCheckpoint(_mlContext);

//BUILD AND TRAIN MODEL
var trainingPipeline = pipeline.Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
        .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

//TRAIN THE MODEL
_trainedModel = trainingPipeline.Fit(_trainingDataView);
_predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(_trainedModel);


//PREDICT WITH THE TRAINED MODEL
GitHubIssue issue = new()
{
    Title = "WebSockets communication is slow in my machine",
    Description = "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.."
};

var prediction = _predEngine.Predict(issue);

Console.WriteLine($"\n=============== Single Prediction just-trained-model - Result: {prediction.Area} ===============\n");

var testDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(_testDataPath, hasHeader: true);
var testMetrics = _mlContext.MulticlassClassification.Evaluate(_trainedModel.Transform(testDataView));
Console.WriteLine($"\n*************************************************************************************************************");
Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
Console.WriteLine($"*       MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
Console.WriteLine($"*       MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
Console.WriteLine($"*************************************************************************************************************\n");

if (!Directory.Exists(Path.GetDirectoryName(_modelPath)))
{
    Directory.CreateDirectory(Path.GetDirectoryName(_modelPath));
}

//SAVING MODEL
Console.Write($"Saving model on : {_modelPath}  .... ");
_mlContext.Model.Save(_trainedModel, _trainingDataView.Schema, _modelPath);
Console.WriteLine("OK\n");



//DEPLOY AND PREDICT WITH A MODEL
ITransformer loadedModel = _mlContext.Model.Load(_modelPath, out var modelInputSchema);
GitHubIssue singleIssue = new GitHubIssue() { Title = "Entity Framework crashes", Description = "When connecting to the database, EF is crashing" };
_predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(loadedModel);
prediction = _predEngine.Predict(singleIssue);
Console.WriteLine($"=============== Single Prediction - Result: {prediction.Area} ===============");

Console.ReadKey();


using Microsoft.ML;
using Microsoft.ML.Data;
using SentimentAnalysis;
using static Microsoft.ML.DataOperationsCatalog;

//dataset file path
string _dataPath = Path.Combine(Environment.CurrentDirectory, "data", "yelp_labelled.txt");

MLContext mlContext = new();
TrainTestData splitDataView = LoadData(mlContext);

ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);

Evaluate(mlContext, model, splitDataView.TestSet);

UseModelWithSingleItem(mlContext, model);

UseModelWithBatchItems(mlContext, model);

Console.ReadKey();




//The UseModelWithBatchItems() method executes the following tasks:
//Creates batch test data.
//    Predicts sentiment based on test data.
//    Combines test data and predictions for reporting.
//    Displays the predicted results.
void UseModelWithBatchItems(MLContext mlContext, ITransformer model)
{
    IEnumerable<SentimentData> sentiments = new[]
    {
        new SentimentData
        {
            SentimentText = "This was a horrible meal"
        },
        new SentimentData
        {
            SentimentText = "I love this spaghetti."
        }
    };
    IDataView batchComments = mlContext.Data.LoadFromEnumerable(sentiments);

    IDataView predictions = model.Transform(batchComments);

    // Use model to predict whether comment data is Positive (1) or Negative (0).
    IEnumerable<SentimentPrediction> predictedResults = mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);

    Console.WriteLine("\n=============== Prediction Test of loaded model with multiple samples ===============\n");
    foreach (SentimentPrediction prediction in predictedResults)
    {
        Console.WriteLine($"Sentiment: {prediction.SentimentText} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {prediction.Probability} ");
    }
    Console.WriteLine("\n=============== End of predictions ===============\n");
}


//The UseModelWithSingleItem() method executes the following tasks:
//Creates a single comment of test data.
//    Predicts sentiment based on test data.
//    Combines test data and predictions for reporting.
//    Displays the predicted results.
void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
{
    //The PredictionEngine is a convenience API, which allows you to perform a prediction on a single instance of data. PredictionEngine is not thread-safe. It's acceptable to use in single-threaded or prototype environments. For improved performance and thread safety in production environments, use the PredictionEnginePool service, which creates an ObjectPool of PredictionEngine objects for use throughout your application. 
    PredictionEngine<SentimentData, SentimentPrediction> predictionFunction = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
    
    SentimentData sampleStatement = new SentimentData
    {
        SentimentText = "This was a very bad steak"
    };

    //The Predict() function makes a prediction on a single row of data.
    var resultPrediction = predictionFunction.Predict(sampleStatement);
    Console.WriteLine("\n=============== Prediction Test of model with a single sample and test dataset ===============\n");

    Console.WriteLine();
    Console.WriteLine($"Sentiment: {resultPrediction.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probability} ");

    Console.WriteLine("\n=============== End of Predictions ===============\n");
    Console.WriteLine();
}



//The Evaluate() method executes the following tasks:

//    Loads the test dataset.
//    Creates the BinaryClassification evaluator.
//    Evaluates the model and creates metrics.
//    Displays the metrics.
void Evaluate(MLContext mlContext, ITransformer model, IDataView testSet)
{
    Console.WriteLine("\n=============== Evaluating Model accuracy with Test data===============\n");

    //uses the Transform() method to make predictions for multiple provided input rows of a test dataset.
    IDataView predictions = model.Transform(testSet);

    //Once you have the prediction set (predictions), the Evaluate() method assesses the model, which compares the predicted values with the actual Labels in the test dataset and returns a CalibratedBinaryClassificationMetrics object on how the model is performing.
    CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");


    Console.WriteLine();
    Console.WriteLine("Model quality metrics evaluation\n");
    Console.WriteLine("--------------------------------");
    
    //The Accuracy metric gets the accuracy of a model, which is the proportion of correct predictions in the test set.
    Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
    //The AreaUnderRocCurve metric indicates how confident the model is correctly classifying the positive and negative classes. You want the AreaUnderRocCurve to be as close to one as possible.
    Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
    //The F1Score metric gets the model's F1 score, which is a measure of balance between precision and recall. You want the F1Score to be as close to one as possible.
    Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
    Console.WriteLine("\n=============== End of model evaluation ===============\n");
}


//The BuildAndTrainModel() method executes the following tasks:

//    Extracts and transforms the data.
//    Trains the model.
//    Predicts sentiment based on test data.
//    Returns the model.
ITransformer BuildAndTrainModel(MLContext mlContext, IDataView trainSet)
{
    //converts the text column (SentimentText) into a numeric key type Features column used by the machine learning algorithm and adds it as a new dataset column
    //The SdcaLogisticRegressionBinaryTrainer is your classification training algorithm. This is appended to the estimator and accepts the featurized SentimentText (Features) and the Label input parameters to learn from the historic data.
    var estimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText)).Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

    Console.WriteLine("\n=============== Create and Train the Model ===============\n");

    //The Fit() method trains your model by transforming the dataset and applying the training.
    var model = estimator.Fit(trainSet);

    Console.WriteLine("\n=============== End of training ===============\n");
    Console.WriteLine();

    return model;
}



//The LoadData() method executes the following tasks:
//    Loads the data.
//    Splits the loaded dataset into train and test datasets.
//    Returns the split train and test datasets.
TrainTestData LoadData(MLContext mlContext)
{
    //The LoadFromTextFile() method defines the data schema and reads in the file. It takes in the data path variables and returns an IDataView.
    IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(_dataPath, hasHeader: false);

    //Split the loaded dataset into train and test datasets and return them in the DataOperationsCatalog.TrainTestData class. Specify the test set percentage of data with the testFractionparameter. The default is 10%, in this case you use 20% to evaluate more data.
    TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

    return splitDataView;
}
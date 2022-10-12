namespace SentimentAnalysis
{
    using System.ComponentModel;
    using Microsoft.ML.Data;

    using static System.Formats.Asn1.AsnWriter;
    using static System.Net.Mime.MediaTypeNames;


    /// <summary>
    /// The input dataset class, SentimentData, has a string for user comments (SentimentText) and a bool (Sentiment) value of either 1 (positive) or 0 (negative) for sentiment. Both fields have LoadColumn attributes attached to them, which describes the data file order of each field. In addition, the Sentiment property has a ColumnName attribute to designate it as the Label field. 
    /// </summary>
    public class SentimentData
    {
        [LoadColumn(0)]
        public string SentimentText;

        [LoadColumn(1), ColumnName("Label")]
        public bool Sentiment;
    }


    /// <summary>
    /// SentimentPrediction is the prediction class used after model training. It inherits from SentimentData so that the input SentimentText can be displayed along with the output prediction. The Prediction boolean is the value that the model predicts when supplied with new input SentimentText. The output class SentimentPrediction contains two other properties calculated by the model: Score - the raw score calculated by the model, and Probability - the score calibrated to the likelihood of the text having positive sentiment.
    /// </summary>
    public class SentimentPrediction : SentimentData
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }
        public float Probability { get; set; }

        public float Score { get; set; }
    }
}

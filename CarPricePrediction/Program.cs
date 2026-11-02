using CarPricePrediction.Models;
using Microsoft.ML;
using Microsoft.ML.Transforms.Text;

namespace CarPricePrediction 
{
    class Program 
    {
        static void Main(string[] args)
        {
            Console.WriteLine("START :: CarPricePrediction");

            // Data file's location.
            var dataFilePath = "Data/used_cars_1000.csv";
            Console.WriteLine(new string('-', 30));
            Console.WriteLine("- Data file to use: " + Path.GetFileName(dataFilePath));

            // Create the ML Context
            var mlContext = new MLContext();

            // Load Data
            Console.WriteLine(new string('-', 30));
            Console.WriteLine("- Loading dataset (LoadFromTextFile())");

            var dataview = mlContext.Data.LoadFromTextFile<CarData>(
                path: dataFilePath,
                hasHeader: true,
                separatorChar: ',');

            Console.WriteLine(new string('-', 30));
            Console.WriteLine("- Preview Data (5 observations)");

            // Preview Data
            var preview = dataview.Preview(maxRows: 5);

            foreach (var row in preview.RowView)
            {
                foreach (var column in row.Values)
                {
                    Console.WriteLine($"    - {column.Key}: {column.Value}");
                }
                Console.WriteLine("");
                Console.WriteLine("");
            }

            // Split the data
            var dataSplit = mlContext.Data.TrainTestSplit(dataview, testFraction: 0.2); // 80-20

            const string FeaturesColumn = "Features";
            const string FeaturizedMakeAndModel = "FeaturizedMakeAndModel";
            const string EncodedTransmission = $"{nameof(CarData.Transmission)}Encoded";
            const string EncodedFuelType = $"{nameof(CarData.FuelType)}Encoded";

            // Define the pipeline
            var pipeline =
               // Featurizing Text for both Make and Model together.
               mlContext.Transforms.Text.FeaturizeText(
                   outputColumnName: FeaturizedMakeAndModel,
                   options: new TextFeaturizingEstimator.Options
                   {
                       CaseMode = TextNormalizingEstimator.CaseMode.Lower
                   },
                   inputColumnNames: [nameof(CarData.Make), nameof(CarData.Model)])

               // Replacing missing Mileage values
               .Append(mlContext.Transforms.ReplaceMissingValues(outputColumnName: nameof(CarData.Mileage), inputColumnName: nameof(CarData.Mileage), replacementMode: Microsoft.ML.Transforms.MissingValueReplacingEstimator.ReplacementMode.Mean))

               // Replacing missing EngineSize values
              .Append(mlContext.Transforms.ReplaceMissingValues(outputColumnName: nameof(CarData.EngineSize), inputColumnName: nameof(CarData.EngineSize), replacementMode: Microsoft.ML.Transforms.MissingValueReplacingEstimator.ReplacementMode.Mean))

               // Encoding Transmission column values
               .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: EncodedTransmission, inputColumnName: nameof(CarData.Transmission)))
                             
               // Encoding FuelType column values
               .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: EncodedFuelType, inputColumnName: nameof(CarData.FuelType)))
                
               // Concatenate Features
               .Append(mlContext.Transforms.Concatenate(
                   FeaturesColumn, 
                   FeaturizedMakeAndModel,
                   nameof(CarData.Year), 
                   nameof(CarData.Mileage), 
                   nameof(CarData.EngineSize),
                   nameof(CarData.Horsepower),
                   EncodedTransmission, 
                   EncodedFuelType,
                   nameof(CarData.Doors)))

               // Normalize features
               .Append(mlContext.Transforms.NormalizeMinMax(FeaturesColumn))

               // Select Task and traineR
               .Append(mlContext.Regression.Trainers.LbfgsPoissonRegression(labelColumnName: nameof(CarData.Price), featureColumnName: FeaturesColumn))
               ;


            // Train the model
            var trainedModel = pipeline.Fit(dataSplit.TrainSet);

            // Evaluate the test set
            var predictions = trainedModel.Transform(dataSplit.TestSet);

            // Obtain metrics
            var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: nameof(CarData.Price));

            Console.WriteLine(new string('-', 30));
            Console.WriteLine($"- R²:  {metrics.RSquared:0.###}");
            Console.WriteLine($"- MAE: {metrics.MeanAbsoluteError:0.##}");
            Console.WriteLine($"- RMSE:{metrics.RootMeanSquaredError:0.##}");

            // Create predcition engine
            var engine = mlContext.Model.CreatePredictionEngine<CarData, CarPricePrediction.Models.CarPricePrediction>(trainedModel);

            Console.WriteLine(new string('-', 30));
            Console.WriteLine("- Test prediction:");

            // Sample prediction
            var sample = new CarData
            {
                Make = "Lexus",
                Model = "IS",
                Year = 2020,
                Mileage = 55000,
                EngineSize = 3.5f,
                Horsepower = 311,
                Transmission = "Automatic",
                FuelType = "Gasoline",
                Doors = 4
            };

            Console.WriteLine($"    Make: {sample.Make}");
            Console.WriteLine($"    Model: {sample.Model}");
            Console.WriteLine($"    Year: {sample.Year}");
            Console.WriteLine($"    Mileage: {sample.Mileage}");
            Console.WriteLine($"    EngineSize: {sample.EngineSize}");
            Console.WriteLine($"    Horsepower: {sample.Horsepower}");
            Console.WriteLine($"    Transmission: {sample.Transmission}");
            Console.WriteLine($"    FuelType: {sample.FuelType}");
            Console.WriteLine($"    Doors: {sample.Doors}");

            var prediction = engine.Predict(sample);

            Console.WriteLine(new string('-', 30));
            Console.WriteLine($"- Predicted price: ${prediction.Score:0}");
            Console.WriteLine(new string('-', 30));
            Console.WriteLine("END :: CarPricePrediction");
            Console.WriteLine("    Press enter to exit...");
            Console.ReadLine();
        }    
    }


}



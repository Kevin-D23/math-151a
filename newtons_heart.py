import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

# Set style for better-looking plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def loadAndExploreData(filepath):
    """Load heart disease dataset and display basic information."""
    heartData = pd.read_csv(filepath)

    print("=" * 60)
    print("HEART DISEASE PREDICTION - LOGISTIC REGRESSION")
    print("=" * 60)
    print(f"Dataset shape: {heartData.shape}")
    print(f"Features: {list(heartData.columns[:-1])}")
    print(f"Target distribution:")
    print(heartData["target"].value_counts().sort_index())
    print(f"  0: No heart disease")
    print(f"  1: Heart disease present")
    print()

    return heartData


def visualizeDataDistribution(heartData):
    """Create visualizations for data exploration."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "Heart Disease Dataset - Data Distribution Analysis",
        fontsize=16,
        fontweight="bold",
    )

    # Target distribution
    targetCounts = heartData["target"].value_counts()
    axes[0, 0].pie(
        targetCounts.values,
        labels=["No Disease", "Disease"],
        autopct="%1.1f%%",
        colors=["lightblue", "lightcoral"],
        startangle=90,
    )
    axes[0, 0].set_title("Target Distribution")

    # Age distribution by target
    axes[0, 1].hist(
        [
            heartData[heartData["target"] == 0]["age"],
            heartData[heartData["target"] == 1]["age"],
        ],
        bins=20,
        alpha=0.7,
        label=["No Disease", "Disease"],
        color=["lightblue", "lightcoral"],
    )
    axes[0, 1].set_xlabel("Age")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Age Distribution by Heart Disease Status")
    axes[0, 1].legend()

    # Chest pain type distribution
    cpCounts = heartData.groupby(["cp", "target"]).size().unstack(fill_value=0)
    cpCounts.plot(kind="bar", ax=axes[1, 0], color=["lightblue", "lightcoral"])
    axes[1, 0].set_xlabel("Chest Pain Type")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("Chest Pain Type vs Heart Disease")
    axes[1, 0].legend(["No Disease", "Disease"])
    axes[1, 0].tick_params(axis="x", rotation=0)

    # Max heart rate distribution by target
    axes[1, 1].hist(
        [
            heartData[heartData["target"] == 0]["thalach"],
            heartData[heartData["target"] == 1]["thalach"],
        ],
        bins=20,
        alpha=0.7,
        label=["No Disease", "Disease"],
        color=["lightblue", "lightcoral"],
    )
    axes[1, 1].set_xlabel("Maximum Heart Rate Achieved")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Max Heart Rate Distribution by Heart Disease Status")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()


def splitData(data, trainRatio=0.8, validationRatio=0.1):
    """Split data into training, validation, and test sets."""
    totalSamples = data.shape[0]
    validationSplitIndex = int(totalSamples * trainRatio)
    testSplitIndex = int(totalSamples * (trainRatio + validationRatio))

    trainingData = data[:validationSplitIndex]
    validationData = data[validationSplitIndex:testSplitIndex]
    testData = data[testSplitIndex:]

    print(f"Data splits:")
    print(f"  Training set: {trainingData.shape[0]} samples")
    print(f"  Validation set: {validationData.shape[0]} samples")
    print(f"  Test set: {testData.shape[0]} samples")
    print()

    return trainingData, validationData, testData


def prepareFeaturesAndTargets(
    trainingData, validationData, testData, targetColumn="target"
):
    """Separate features and targets for all datasets."""
    # Training set
    trainingTargets = trainingData[targetColumn]
    trainingFeatures = trainingData.drop(targetColumn, axis=1)

    # Validation set
    validationTargets = validationData[targetColumn]
    validationFeatures = validationData.drop(targetColumn, axis=1)

    # Test set
    testTargets = testData[targetColumn]
    testFeatures = testData.drop(targetColumn, axis=1)

    return (
        trainingTargets,
        trainingFeatures,
        validationTargets,
        validationFeatures,
        testTargets,
        testFeatures,
    )


def sigmoidActivation(linearCombination):
    """Apply sigmoid activation function."""
    return 1 / (
        1 + np.exp(-np.clip(linearCombination, -250, 250))
    )  # Clip to prevent overflow


def computeLogLikelihood(features, targets, coefficients):
    """Compute log-likelihood for logistic regression."""
    # Handle both DataFrame and numpy array inputs for coefficients
    if hasattr(coefficients, "values"):
        coeffArray = coefficients.values.flatten()
    else:
        coeffArray = coefficients.flatten()

    # Handle both DataFrame and numpy array inputs for targets
    if hasattr(targets, "values"):
        targetArray = targets.values.flatten()
    else:
        targetArray = targets.flatten() if hasattr(targets, "flatten") else targets

    linearOutput = features.dot(coeffArray)
    probabilities = sigmoidActivation(linearOutput)
    # Add small epsilon to prevent log(0)
    epsilon = 1e-15
    probabilities = np.clip(probabilities, epsilon, 1 - epsilon)

    logLikelihood = np.sum(
        targetArray * np.log(probabilities)
        + (1 - targetArray) * np.log(1 - probabilities)
    )
    return -logLikelihood  # Return negative log-likelihood (loss)


def computeNewtonStep(
    currentCoefficients, targetValues, featureMatrix, regularizationLambda=None
):
    """Compute one step of Newton's method for logistic regression."""
    # Calculate probabilities
    linearOutput = featureMatrix.dot(currentCoefficients[:, 0])
    predictedProbabilities = np.array(sigmoidActivation(linearOutput), ndmin=2).T

    # Calculate weight matrix (diagonal)
    probabilityWeights = predictedProbabilities * (1 - predictedProbabilities)
    weightMatrix = np.diag(probabilityWeights[:, 0])

    # Calculate Hessian and gradient
    hessianMatrix = featureMatrix.T.dot(weightMatrix).dot(featureMatrix)
    gradientVector = featureMatrix.T.dot(targetValues - predictedProbabilities)

    # Apply regularization if specified
    if regularizationLambda:
        regularizationTerm = regularizationLambda * np.eye(currentCoefficients.shape[0])
        newtonStep = np.dot(
            np.linalg.inv(hessianMatrix + regularizationTerm), gradientVector
        )
    else:
        newtonStep = np.dot(np.linalg.inv(hessianMatrix), gradientVector)

    updatedCoefficients = currentCoefficients + newtonStep
    return updatedCoefficients


def checkConvergenceCriteria(
    previousCoefficients,
    currentCoefficients,
    tolerance,
    currentIteration,
    maxIterations,
):
    """Check if the optimization has converged."""
    coefficientChanges = np.abs(previousCoefficients - currentCoefficients)
    hasConverged = not (
        np.any(coefficientChanges > tolerance) and currentIteration < maxIterations
    )
    return hasConverged


def calculateModelAccuracy(featureMatrix, trueTargets, modelCoefficients):
    """Calculate accuracy of the logistic regression model."""
    linearPredictions = featureMatrix.dot(modelCoefficients)
    predictedProbabilities = np.array(sigmoidActivation(linearPredictions))

    # Convert probabilities to binary predictions (threshold = 0.5)
    binaryPredictions = np.greater(
        predictedProbabilities, 0.5 * np.ones((predictedProbabilities.shape[1], 1))
    )
    correctPredictions = np.count_nonzero(np.equal(binaryPredictions, trueTargets))
    accuracyPercentage = (correctPredictions / predictedProbabilities.shape[0]) * 100

    return accuracyPercentage


def trainWithNewtonsMethod(
    trainingFeatures,
    trainingTargets,
    validationFeatures,
    validationTargets,
    maxIterations=20,
    convergenceTolerance=0.1,
    regularizationLambda=1,
):
    """Train logistic regression using Newton's method."""
    print("=" * 60)
    print("NEWTON'S METHOD OPTIMIZATION")
    print("=" * 60)
    print(f"Hyperparameters:")
    print(f"  Max iterations: {maxIterations}")
    print(f"  Convergence tolerance: {convergenceTolerance}")
    print(f"  Regularization term: {regularizationLambda}")
    print()

    # Initialize coefficients
    numFeatures = trainingFeatures.shape[1]
    previousCoefficients = np.ones((numFeatures, 1))
    currentCoefficients = np.zeros((numFeatures, 1))

    iterationCount = 0
    hasConverged = False

    # Track training history
    trainingHistory = {
        "iterations": [],
        "validation_accuracy": [],
        "training_loss": [],
        "validation_loss": [],
    }

    print("Training Progress:")
    print("-" * 40)

    startTime = time.time()

    while not hasConverged:
        validationAccuracy = calculateModelAccuracy(
            validationFeatures, validationTargets.to_frame(), previousCoefficients
        )
        trainingLoss = computeLogLikelihood(
            trainingFeatures, trainingTargets, previousCoefficients
        )
        validationLoss = computeLogLikelihood(
            validationFeatures, validationTargets, previousCoefficients
        )

        print(
            f"Iteration {iterationCount:2d}: Validation Accuracy = {validationAccuracy:6.2f}%, Loss = {validationLoss:.4f}"
        )

        # Store history
        trainingHistory["iterations"].append(iterationCount)
        trainingHistory["validation_accuracy"].append(validationAccuracy)
        trainingHistory["training_loss"].append(trainingLoss)
        trainingHistory["validation_loss"].append(validationLoss)

        previousCoefficients = currentCoefficients
        currentCoefficients = computeNewtonStep(
            currentCoefficients,
            trainingTargets.to_frame(),
            trainingFeatures,
            regularizationLambda,
        )
        iterationCount += 1
        hasConverged = checkConvergenceCriteria(
            previousCoefficients,
            currentCoefficients,
            convergenceTolerance,
            iterationCount,
            maxIterations,
        )

    trainingTime = time.time() - startTime

    print("-" * 40)
    print(f"Training completed in {trainingTime:.4f} seconds")

    return currentCoefficients, iterationCount, trainingHistory, trainingTime


def computeGradientDescentStep(
    currentCoefficients, targetValues, featureMatrix, learningRate=0.0001
):
    """Compute one step of gradient descent for logistic regression."""
    linearOutput = featureMatrix.dot(currentCoefficients)
    predictedProbabilities = np.array(sigmoidActivation(linearOutput))
    gradientStep = learningRate * (
        featureMatrix.T.dot(targetValues - predictedProbabilities)
    )
    updatedCoefficients = currentCoefficients + gradientStep

    return updatedCoefficients


def trainWithGradientDescent(
    trainingFeatures,
    trainingTargets,
    validationFeatures,
    validationTargets,
    batchSize=50,
    learningRate=0.0001,
    maxEpochs=100,
):
    """Train logistic regression using gradient descent."""
    print("=" * 60)
    print("GRADIENT DESCENT OPTIMIZATION (for comparison)")
    print("=" * 60)
    print(f"Hyperparameters:")
    print(f"  Batch size: {batchSize}")
    print(f"  Learning rate: {learningRate}")
    print(f"  Max epochs: {maxEpochs}")
    print()

    # Initialize coefficients
    numFeatures = trainingFeatures.shape[1]
    currentCoefficients = np.zeros((numFeatures, 1))

    epochCount = 0

    # Track training history
    trainingHistory = {
        "epochs": [],
        "validation_accuracy": [],
        "training_loss": [],
        "validation_loss": [],
    }

    print("Training Progress:")
    print("-" * 40)

    startTime = time.time()

    while epochCount < maxEpochs:
        if epochCount % 10 == 0:
            validationAccuracy = calculateModelAccuracy(
                validationFeatures, validationTargets.to_frame(), currentCoefficients
            )
            trainingLoss = computeLogLikelihood(
                trainingFeatures, trainingTargets, currentCoefficients
            )
            validationLoss = computeLogLikelihood(
                validationFeatures, validationTargets, currentCoefficients
            )

            print(
                f"Epoch {epochCount:2d}: Validation Accuracy = {validationAccuracy:6.2f}%, Loss = {validationLoss:.4f}"
            )

            # Store history
            trainingHistory["epochs"].append(epochCount)
            trainingHistory["validation_accuracy"].append(validationAccuracy)
            trainingHistory["training_loss"].append(trainingLoss)
            trainingHistory["validation_loss"].append(validationLoss)

        # Mini-batch gradient descent
        for batchStart in range(0, trainingFeatures.shape[0], batchSize):
            batchEnd = batchStart + batchSize
            batchTargets = trainingTargets[batchStart:batchEnd].to_frame()
            batchFeatures = trainingFeatures[batchStart:batchEnd]

            currentCoefficients = computeGradientDescentStep(
                currentCoefficients, batchTargets, batchFeatures, learningRate
            )
        epochCount += 1

    trainingTime = time.time() - startTime

    print("-" * 40)
    print(f"Training completed in {trainingTime:.4f} seconds")

    return currentCoefficients, epochCount, trainingHistory, trainingTime


def trainWithSklearn(
    trainingFeatures,
    trainingTargets,
    validationFeatures,
    validationTargets,
    maxIter=1000,
):
    """Train logistic regression using scikit-learn."""
    print("=" * 60)
    print("SCIKIT-LEARN LOGISTIC REGRESSION (for comparison)")
    print("=" * 60)
    print(f"Hyperparameters:")
    print(f"  Max iterations: {maxIter}")
    print(f"  Solver: lbfgs")
    print(f"  Regularization: L2 (C=1.0)")
    print()

    startTime = time.time()

    # Initialize and train sklearn model
    sklearnModel = LogisticRegression(
        max_iter=maxIter, solver="lbfgs", C=1.0, random_state=42
    )
    sklearnModel.fit(trainingFeatures, trainingTargets)

    trainingTime = time.time() - startTime

    # Calculate accuracies
    trainAccuracy = sklearnModel.score(trainingFeatures, trainingTargets) * 100
    valAccuracy = sklearnModel.score(validationFeatures, validationTargets) * 100

    print(f"Training Progress:")
    print("-" * 40)
    print(f"Training Accuracy: {trainAccuracy:.2f}%")
    print(f"Validation Accuracy: {valAccuracy:.2f}%")
    print(f"Converged in {sklearnModel.n_iter_[0]} iterations")
    print(f"Training completed in {trainingTime:.4f} seconds")
    print("-" * 40)

    return sklearnModel, sklearnModel.n_iter_[0], trainingTime


def plotErrorConvergence(newtonHistory, gdHistory):
    """Plot error convergence for both methods."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "Training Progress and Error Convergence Analysis",
        fontsize=16,
        fontweight="bold",
    )

    # Newton's method - Loss convergence
    axes[0, 0].plot(
        newtonHistory["iterations"],
        newtonHistory["training_loss"],
        "o-",
        linewidth=2,
        markersize=6,
        color="blue",
        label="Training Loss",
    )
    axes[0, 0].plot(
        newtonHistory["iterations"],
        newtonHistory["validation_loss"],
        "o-",
        linewidth=2,
        markersize=6,
        color="red",
        label="Validation Loss",
    )
    axes[0, 0].set_xlabel("Iterations")
    axes[0, 0].set_ylabel("Log Loss")
    axes[0, 0].set_title("Newton's Method - Loss Convergence")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].set_yscale("log")

    # Newton's method - Accuracy convergence
    axes[0, 1].plot(
        newtonHistory["iterations"],
        newtonHistory["validation_accuracy"],
        "o-",
        linewidth=2,
        markersize=6,
        color="green",
        label="Validation Accuracy",
    )
    axes[0, 1].set_xlabel("Iterations")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].set_title("Newton's Method - Accuracy Convergence")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Gradient descent - Loss convergence
    axes[1, 0].plot(
        gdHistory["epochs"],
        gdHistory["training_loss"],
        "o-",
        linewidth=2,
        markersize=4,
        color="blue",
        label="Training Loss",
    )
    axes[1, 0].plot(
        gdHistory["epochs"],
        gdHistory["validation_loss"],
        "o-",
        linewidth=2,
        markersize=4,
        color="red",
        label="Validation Loss",
    )
    axes[1, 0].set_xlabel("Epochs")
    axes[1, 0].set_ylabel("Log Loss")
    axes[1, 0].set_title("Gradient Descent - Loss Convergence")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    axes[1, 0].set_yscale("log")

    # Gradient descent - Accuracy convergence
    axes[1, 1].plot(
        gdHistory["epochs"],
        gdHistory["validation_accuracy"],
        "o-",
        linewidth=2,
        markersize=4,
        color="green",
        label="Validation Accuracy",
    )
    axes[1, 1].set_xlabel("Epochs")
    axes[1, 1].set_ylabel("Accuracy (%)")
    axes[1, 1].set_title("Gradient Descent - Accuracy Convergence")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()


def displayFinalResults(
    newtonCoefficients,
    newtonIterations,
    newtonTime,
    gdCoefficients,
    gdEpochs,
    gdTime,
    sklearnModel,
    sklearnIterations,
    sklearnTime,
    testFeatures,
    testTargets,
):
    """Display final results and comparison."""
    newtonTestAccuracy = calculateModelAccuracy(
        testFeatures, testTargets.to_frame(), newtonCoefficients
    )
    gdTestAccuracy = calculateModelAccuracy(
        testFeatures, testTargets.to_frame(), gdCoefficients
    )
    sklearnTestAccuracy = sklearnModel.score(testFeatures, testTargets) * 100

    print(f"FINAL RESULTS (Newton's Method):")
    print(f"  Iterations completed: {newtonIterations}")
    print(f"  Training time: {newtonTime:.4f} seconds")
    print(f"  Test Accuracy: {newtonTestAccuracy:.2f}%")
    print()

    print(f"FINAL RESULTS (Gradient Descent):")
    print(f"  Epochs completed: {gdEpochs}")
    print(f"  Training time: {gdTime:.4f} seconds")
    print(f"  Test Accuracy: {gdTestAccuracy:.2f}%")
    print()

    print(f"FINAL RESULTS (Scikit-Learn):")
    print(f"  Iterations completed: {sklearnIterations}")
    print(f"  Training time: {sklearnTime:.4f} seconds")
    print(f"  Test Accuracy: {sklearnTestAccuracy:.2f}%")
    print()

    print("=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Method':<20} {'Accuracy':<12} {'Time (s)':<12} {'Iterations':<12}")
    print("-" * 60)
    print(
        f"{'Newton\'s Method':<20} {newtonTestAccuracy:<12.2f} {newtonTime:<12.4f} {newtonIterations:<12}"
    )
    print(
        f"{'Gradient Descent':<20} {gdTestAccuracy:<12.2f} {gdTime:<12.4f} {gdEpochs:<12}"
    )
    print(
        f"{'Scikit-Learn':<20} {sklearnTestAccuracy:<12.2f} {sklearnTime:<12.4f} {sklearnIterations:<12}"
    )
    print("=" * 60)


def main():
    """Main function to orchestrate the heart disease prediction workflow."""
    # Load and explore data
    heartDataset = loadAndExploreData("heart.csv")

    # Visualize data distribution
    visualizeDataDistribution(heartDataset)

    # Split data
    trainingData, validationData, testData = splitData(heartDataset)

    # Prepare features and targets
    (
        trainingTargets,
        trainingFeatures,
        validationTargets,
        validationFeatures,
        testTargets,
        testFeatures,
    ) = prepareFeaturesAndTargets(trainingData, validationData, testData)

    # Train with Newton's method
    newtonCoefficients, newtonIterations, newtonHistory, newtonTime = (
        trainWithNewtonsMethod(
            trainingFeatures, trainingTargets, validationFeatures, validationTargets
        )
    )

    # Train with gradient descent
    gdCoefficients, gdEpochs, gdHistory, gdTime = trainWithGradientDescent(
        trainingFeatures, trainingTargets, validationFeatures, validationTargets
    )

    # Train with scikit-learn
    sklearnModel, sklearnIterations, sklearnTime = trainWithSklearn(
        trainingFeatures, trainingTargets, validationFeatures, validationTargets
    )

    # Display results
    displayFinalResults(
        newtonCoefficients,
        newtonIterations,
        newtonTime,
        gdCoefficients,
        gdEpochs,
        gdTime,
        sklearnModel,
        sklearnIterations,
        sklearnTime,
        testFeatures,
        testTargets,
    )

    # Plot error convergence
    plotErrorConvergence(newtonHistory, gdHistory)


if __name__ == "__main__":
    main()

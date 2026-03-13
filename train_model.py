from src.model_utils import load_and_preprocess_data, split_data, train_model, evaluate_model, save_model, plot_feature_importance

def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y, feature_names = load_and_preprocess_data('data/pcos_dataset.csv')
    
    # Split data
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train model
    print("Training Random Forest model...")
    model = train_model(X_train, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    
    print("\nModel Evaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])
    
    # Save model
    print("Saving model...")
    save_model(model, 'models/pcos_model.pkl')
    
    # Plot feature importance
    print("Generating feature importance plot...")
    plot_feature_importance(model, feature_names, 'models/feature_importance.png')
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
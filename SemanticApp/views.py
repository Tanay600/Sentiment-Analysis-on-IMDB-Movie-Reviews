from django.shortcuts import render
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer

classifier = load("./savedmodels/classifier.joblib")
vectorizer = load("./savedmodels/vectorizer.joblib")

def predictor(request): 
    return render(request, "main.html")

def formInfo(request):
    reviews = request.GET["reviews"]
    review_list = reviews.split('\n')
    
    results = []
    for review in review_list:
        # Transform the review using the loaded vectorizer
        review_vector = vectorizer.transform([review])
        
        # Predict sentiment using the classifier
        y_pred = classifier.predict(review_vector)

        results.append('Negative' if y_pred == 0 else 'Positive')
    
    return render(request, "result.html", {'results': results})

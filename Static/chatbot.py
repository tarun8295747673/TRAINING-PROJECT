from flask import Flask, render_template, request , jsonify

import google.generativeai as genai

from sklearn.linear_model import LinearRegression
import pandas as pd


app = Flask(__name__)
app.secret_key = 'your_secret_key'


API_KEY = 'AIzaSyCnHiPnc81WluNjSklL6lLR5FO_NbHRCfM'
#'AIzaSyCCrYnLhDIgToWeG4u_nPpQcB9uNJMze0U'
genai.configure(api_key=API_KEY)

model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

medical_keywords = [
"CustomerFeedback", "VehicleDesign", "SentimentAnalysis", "TopicModeling", "FeedbackAnalysis", "DesignImprovement", 
"NLPTool", "MachineLearning", "FeedbackTrends", "DataAnalytics", "FeatureAnalysis", "UserExperience", "ReviewAnalysis", 
"SurveyAnalysis", "SentimentTracking", "CustomerInsights", "DesignOptimization", "FeedbackClassification", "KeywordExtraction", 
"FeedbackLoop", "CustomerPreferences", "ProductDevelopment", "BehavioralAnalysis", "TextAnalytics", "FeedbackIntegration", 
"CustomerSentiment", "TrendAnalysis", "DesignInsights", "DataPreprocessing", "FeatureSentiment", "RealTimeAnalysis", 
"CustomerReviews", "FeedbackPatterns", "SentimentMetrics", "ProductFeedback", "VoiceOfCustomer", "DesignEnhancement", 
"CustomerExperience", "FeedbackMonitoring", "AIinDesign", "UserFeedback", "DesignStrategy", "SentimentDashboard", 
"ConsumerBehavior", "FeatureRequests", "FeedbackPrioritization", "ReviewSentiment", "FeedbackAggregation", 
"AutomotiveDesign", "DesignCustomization", "FeedbackSystem", "CustomerPainPoints", "InsightGeneration", "AIFeedbackTool", 
"ProductInsights", "CustomerNeeds", "SentimentAnalysisTool", "MarketResearch", "DesignQuality", "UserCenteredDesign", 
"FeedbackData", "SentimentClassification", "CustomerVoice", "DataVisualization", "CustomerSatisfaction", "AutomotiveUX", 
"FeatureOptimization", "PredictiveAnalytics", "TextMining", "FeedbackAutomation", "CustomerJourney", "DesignInnovation", 
"UserSentiment", "ReviewAnalytics", "AIInsights", "FeedbackSegmentation", "DesignImprovements", "CustomerResearch", 
"SentimentInsights", "FeedbackMetrics", "RealTimeFeedback", "CustomerExperienceEnhancement", "AutomotiveFeedback", 
"DesignAnalysis", "UserFeedbackDashboard", "DataDrivenInsights", "CustomerInsightsPlatform", "FeedbackInterpretation", 
"SentimentTrends", "BehavioralInsights", "FeatureAnalysisTool", "FeedbackTracking", "CustomerOpinions", "DesignFeedback", 
"InsightfulAnalysis", "ReviewFeedback", "UserPreferences", "FeedbackAnalysisTool", "CustomerInsightsAnalysis", 
"AutomotiveFeedbackInsights", "FeatureRequestsAnalysis", "DesignFeedbackSystem", "SentimentResearch", "ConsumerInsights", 
"FeedbackDataMining", "DesignData", "CustomerFeedbackPlatform", "FeedbackRecommendations", "UserFeedbackInsights", 
"SentimentTrackingTool", "ProductFeedbackAnalysis", "DesignEnhancementTool", "CustomerFeedbackMetrics", "SentimentEvaluation", 
"AutomotiveUserExperience", "FeedbackCategorization", "DesignInsightsTool", "CustomerReviewAnalysis", "SentimentAnalysisInsights", 
"FeatureSentimentAnalysis", "CustomerCentricFeedback", "FeedbackInterpretationTool", "DesignResearch", "SentimentInsightsTool", 
"AutomotiveDesignTrends", "CustomerFeedbackTrends", "FeatureOptimizationInsights", "DesignStrategyInsights", "UserExperienceFeedback", 
"FeedbackInsightsPlatform", "RealTimeFeedbackAnalysis", "CustomerSentimentMetrics", "ProductDesignFeedback", "FeedbackProcessing", 
"AutomotiveProductInsights", "SentimentAnalysisMetrics", "UserFeedbackTrends", "DesignCustomizationInsights", "FeedbackAggregationTool", 
"CustomerPreferencesAnalysis", "DesignDataInsights", "SentimentAnalysisPlatform", "AutomotiveInsights", "CustomerFeedbackDashboard", 
"FeatureInsights", "ReviewSentimentMetrics", "FeedbackSynthesis", "DesignFeedbackMetrics", "CustomerExperienceData", 
"InsightfulFeedbackAnalysis", "FeedbackAutomationTool", "UserSentimentAnalysis", "ProductImprovementInsights", "SentimentFeedback", 
"CustomerNeedsAnalysis", "DesignFeedbackInsights", "FeatureAnalytics", "FeedbackLoopInsights", "SentimentMetricsTool", 
"CustomerSentimentInsights", "AutomotiveDesignInsightsTool", "FeedbackInsightsDashboard", "UserFeedbackTrends", "DesignFeedbackAnalysis", 
"SentimentAnalysisDashboard", "CustomerFeedbackResearch", "ProductFeedbackInsights", "BehavioralFeedbackAnalysis", 
"DesignDataPlatform", "FeatureRequestsInsights", "FeedbackClassificationTool", "CustomerExperienceInsights", 
"SentimentInsightsPlatform", "AutomotiveUXInsights", "FeedbackDataAnalysis", "DesignImprovementMetrics", "CustomerOpinionInsights", 
"ProductDesignInsights", "FeedbackTrendsDashboard", "UserFeedbackAnalysisTool", "SentimentResearchTool", "CustomerInsightsResearch", 
"FeatureSentimentInsights", "FeedbackDataPlatform", "DesignResearchInsights", "CustomerReviewMetrics", "AutomotiveDesignFeedback", 
"UserExperienceInsights", "FeedbackAnalysisDashboard", "SentimentFeedbackAnalysis", "ProductInsightsPlatform", "DesignTrendsInsights", 
"CustomerFeedbackStrategy", "SentimentTrackingInsights", "FeedbackAnalysisMetrics", "AutomotiveFeedbackAnalysis", 
"FeatureFeedbackInsights", "CustomerCentricDesignInsights"
]

csv_path = 'datasetFile.csv'  
data = pd.read_csv(csv_path)

X = data[['Parameter 1', 'Parameter 2', 'Parameter 3', 'Parameter 4', 'Parameter 5', 'Parameter 6', 'Parameter 7', 'Parameter 8']]
y = data['Parameter 9']
X = X.to_numpy()

model1 = LinearRegression()
model1.fit(X, y)

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        inputs = [float(request.form[field]) for field in ['Parameter 1', 'Parameter 2', 'Parameter 3', 'Parameter 4', 'Parameter 5', 'Parameter 6', 'Parameter 7', 'Parameter 8']]
        prediction = model1.predict([inputs])
        output = "Argument 1" if prediction[0] >= 0.5 else "Argument 2"
        return render_template('index1.html', prediction_text=f'{output}')
    return render_template('index1.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chat.html')

@app.route('/coding')
def coding():
    return render_template('coding.html')

@app.route('/about_CodroidHub')
def about_CodroidHub():
    return render_template('about CodroidHub.html')

@app.route('/about_instructor')
def about_instructor():
    return render_template('about instructor.html')

@app.route('/about_developer')
def about_developer():
    return render_template('about developer.html')


@app.route('/ask', methods=['POST'])
def ask():
    user_message = str(request.form['messageText'])
    
    if not is_medical_query(user_message):
        bot_response_text = "I'm sorry, I can only answer medical-related questions. Please ask a question related to medical topics."
    else:
        bot_response = chat.send_message(user_message)
        bot_response_text = bot_response.text
    
    return jsonify({'status': 'OK', 'answer': bot_response_text})

def is_medical_query(query):
    return any(keyword.lower() in query.lower() for keyword in medical_keywords)

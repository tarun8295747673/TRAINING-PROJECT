from flask import Flask, request , jsonify

import google.generativeai as genai

app = Flask(__name__)
app.secret_key = 'your_secret_key'


API_KEY = 'AIzaSyCnHiPnc81WluNjSklL6lLR5FO_NbHRCfM'
#'AIzaSyCCrYnLhDIgToWeG4u_nPpQcB9uNJMze0U'
genai.configure(api_key=API_KEY)

model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

medical_keywords = [
"Oil", "Tires", "Brakes", "Battery", "Engine", "Fluids", "Alignment", "Coolant", "Transmission", "Suspension", "Exhaust",
"Filter", "Radiator", "Spark", "Wipers", "Pressure", "Rotors", "Pads", "Bearings", "Clutch", "Shocks", "Steering", "Ignition", 
"Axles", "Gaskets", "Pistons", "Hoses", "Belts", "Lights", "Sensors", "Plugs", "Valves", "Fuel", "Airbags", "Muffler", "Turbo", 
"Differential", "Bumper", "Alternator", "Pump", "Timing", "Battery", "Dashboard", "Transmission", "Chassis", "Cylinder", 
"Bearings", "Catalytic", "Differential", "Fuses", "Grille", "Headlights", "Heating", "Insulation", "Powertrain", "Seals", 
"Thermostat", "U-joint", "Vacuum", "Wiring","CustomerFeedback", "VehicleDesign", "SentimentAnalysis", "TopicModeling", "FeedbackAnalysis", "DesignImprovement", 
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
"FeatureFeedbackInsights", "CustomerCentricDesignInsights", "hi", "hello"
]




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

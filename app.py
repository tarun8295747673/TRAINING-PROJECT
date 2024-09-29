from flask import Flask, render_template, request , jsonify 

import google.generativeai as genai

from sklearn.linear_model import LinearRegression
import pandas as pd

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter


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

csv_path = 'datasetFile.csv'  
data = pd.read_csv(csv_path)

X = data[['Parameter 1', 'Parameter 2', 'Parameter 3', 'Parameter 4', 'Parameter 5', 'Parameter 6', 'Parameter 7', 'Parameter 8']]
y = data['Parameter 9']
X = X.to_numpy()

model1 = LinearRegression()
model1.fit(X, y)

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/wCV')
def wCV():
    return render_template('problem_statement_template_CV.html')

@app.route('/wNLP')
def wNLP():
    return render_template('problem_statement_template_NLP.html')

@app.route('/wSD')
def wSD():
    return render_template('problem_statement_template_SD.html')

@app.route('/')
def home():
    return render_template('WEBPAGE.html')

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

@app.route('/about_CodroidHub')
def about_CodroidHub():
    return render_template('about CodroidHub.html')

@app.route('/about_instructor')
def about_instructor():
    return render_template('about instructor.html')

@app.route('/about_developer')
def about_developer():
    return render_template('about developer.html')

@app.route('/main')
def main():
    return render_template('coding.html')


@app.route('/ask', methods=['POST'])
def ask():
    user_message = str(request.form['messageText'])
    
    if not is_medical_query(user_message):
        bot_response_text = "I'm sorry, I can only answer vehivle maintenance related questions. Please ask a question related to vehicle maintenance topics."
    else:
        bot_response = chat.send_message(user_message)
        bot_response_text = bot_response.text
    
    return jsonify({'status': 'OK', 'answer': bot_response_text})

def is_medical_query(query):
    return any(keyword.lower() in query.lower() for keyword in medical_keywords)

# Route to display the .py file
@app.route('/chatbotScript')
def chatbotScript():
    with open('static/chatbot.py', 'r') as f:
        code = f.read()
    lexer = PythonLexer()
    formatter = HtmlFormatter(full=True, linenos=True, style='friendly')
    highlighted_code = highlight(code, lexer, formatter)
    html_content = f"""
    <html>
    <head>
        <title>Chatbot Script</title>
        <style>{formatter.get_style_defs('.highlight')}</style>
    </head>
    <body>
        <h1>Highlighted Python Script</h1>
        <div class="highlight">{highlighted_code}</div>
    </body>
    </html>
    """
    return html_content

@app.route('/StatisticalScript')
def StatisticalScript():
    with open('static/Statistical.py', 'r') as f:
        code = f.read()
    lexer = PythonLexer()
    formatter = HtmlFormatter(full=True, linenos=True, style='friendly')
    highlighted_code = highlight(code, lexer, formatter)
    html_content = f"""
    <html>
    <head>
        <title>Chatbot Script</title>
        <style>{formatter.get_style_defs('.highlight')}</style>
    </head>
    <body>
        <h1>Statistical Script</h1>
        <div class="highlight">{highlighted_code}</div>
    </body>
    </html>
    """
    
    return html_content

@app.route('/powerBI')
def powerBI():
    return render_template('dashboard.html')

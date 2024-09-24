import requests
import google.generativeai as genai
import numpy as np
from IPython.display import display, Markdown

# Gemini API Key
genai.configure(api_key='AIzaSyCK1V7X1xWcynJGp-Ke98G_kd_pZse0f7I')

model = genai.GenerativeModel('gemini-pro')

chat = model.start_chat(history=[])

# Sample current portfolio data
portfolio = [
    {'ETF_TCK_CD': 'SPY', 'Value': 1000, 'Return': 0.02},
    {'ETF_TCK_CD': 'QQQ', 'Value': 3000, 'Return': 0.015},
    {'ETF_TCK_CD': 'IVV', 'Value': 2000, 'Return': 0.017},
]

# Sample ETF candidates evaluation
etf_candidates = [
    ('VTI', 0.57),
    ('VOO', 0.47),
    ('ARKK', 0.43),
]

# Expected returns and covariance matrix (example values)
expected_returns = np.array([0.02, 0.015, 0.017, 0.057])  # SPY, QQQ, IVV, VTI
cov_matrix = np.array([[0.0004, 0.0001, 0.0002, 0.0003],
                       [0.0001, 0.0003, 0.0002, 0.0001],
                       [0.0002, 0.0002, 0.0005, 0.0004],
                       [0.0003, 0.0001, 0.0004, 0.0006]])

def to_markdown(text):
    return Markdown(text)

def calculate_portfolio_performance(weights, expected_returns, cov_matrix):
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_risk

def get_gemini_recommendations(portfolio, etf_evaluations):
    portfolio_summary = '\n'.join([f"{p['ETF_TCK_CD']}: {p['Value']} USD with {p['Return']*100}% expected return" for p in portfolio])
    etf_summary = '\n'.join([f"{etf[0]} with score {etf[1]:.2f}" for etf in etf_evaluations])

    # Generate prompt
    prompt = f"""
    Based on the current portfolio with these allocations:
    {portfolio_summary}

    And considering the following ETF candidates:
    {etf_summary}

    Provide recommendations for rebalancing the portfolio and target allocations based on portfolio theory.
    """

    # Send message to Gemini API
    response = chat.send_message(prompt)
    
    if response and response.candidates:
        # Extract text from the response
        recommendations = response.candidates[0].content.parts[0].text  # Accessing as an attribute
        
        # Display the response in Markdown format
        display(to_markdown(recommendations))
        
        # Print the recommendations to the console
        print(recommendations)  # Print to console for clarity
        
        return recommendations  # Return the text response
    
    return {}

def perform_portfolio_evaluation(portfolio, etf_candidates):
    # Assuming the evaluation logic here returns scores based on some logic
    etf_evaluations = etf_candidates  # Just a placeholder for actual evaluation logic
    
    # Calculate weights based on current portfolio values
    total_value = sum(p['Value'] for p in portfolio)
    weights = np.array([p['Value'] / total_value for p in portfolio])  # Calculate weights
    
    # Calculate portfolio performance
    portfolio_return, portfolio_risk = calculate_portfolio_performance(weights, expected_returns[:len(weights)], cov_matrix[:len(weights), :len(weights)])
    print(f"Expected Portfolio Return: {portfolio_return:.2%}, Portfolio Risk (Std Dev): {portfolio_risk:.2%}")
    
    target_allocation = get_gemini_recommendations(portfolio, etf_evaluations)
    return target_allocation

# Main execution
if __name__ == "__main__":
    perform_portfolio_evaluation(portfolio, etf_candidates)
